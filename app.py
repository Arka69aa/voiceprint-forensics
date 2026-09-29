
import hashlib
import io
from typing import Dict, List, Tuple

import librosa
import numpy as np
import soundfile as sf
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from speechbrain.inference.speaker import SpeakerRecognition


# ============================================================
# VoicePrint Forensics
# ============================================================

st.set_page_config(
    page_title="VoicePrint Forensics",
    page_icon="🎙️",
    layout="wide",
)

SAMPLE_RATE = 16000
DETECTOR_CHUNK_SECONDS = 4
DETECTOR_CHUNK_SAMPLES = SAMPLE_RATE * DETECTOR_CHUNK_SECONDS

SPEAKER_MODEL = "speechbrain/spkrec-ecapa-voxceleb"

# Lightweight Wav2Vec2 detector suitable for Streamlit deployment.
# The previous 0xmola checkpoint had an invalid/missing model_type.
DEEPFAKE_MODEL = "mo-thecreator/Deepfake-audio-detection"


# -----------------------------
# Audio utilities
# -----------------------------
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_audio_bytes(data: bytes) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(io.BytesIO(data), always_2d=False)
    audio = np.asarray(audio)

    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float32)

    peak = np.max(np.abs(audio)) if audio.size else 0.0
    if peak > 1.5:
        audio = audio / peak

    if sr != SAMPLE_RATE:
        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=SAMPLE_RATE,
        )

    audio = np.asarray(audio, dtype=np.float32)

    if audio.size == 0:
        raise ValueError("No decodable audio was found.")

    return audio, SAMPLE_RATE


def audio_stats(audio: np.ndarray, sr: int) -> Dict[str, float]:
    duration = len(audio) / sr
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio**2) + 1e-12))

    zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio)[0]))
    centroid = float(
        np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)[0])
    )
    rolloff = float(
        np.mean(
            librosa.feature.spectral_rolloff(
                y=audio,
                sr=sr,
                roll_percent=0.85,
            )[0]
        )
    )

    frame_rms = librosa.feature.rms(y=audio)[0]
    silence_ratio = float(
        np.mean(frame_rms < max(1e-5, rms * 0.12))
    )

    return {
        "duration": duration,
        "peak": peak,
        "rms": rms,
        "zcr": zcr,
        "spectral_centroid": centroid,
        "spectral_rolloff": rolloff,
        "silence_ratio": silence_ratio,
    }


def acoustic_vector(audio: np.ndarray, sr: int) -> np.ndarray:
    """Diagnostic only; this is not a speaker/authenticity embedding."""
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(audio)
    rms = librosa.feature.rms(y=audio)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)

    features = []
    for x in [mfcc, contrast, chroma, zcr, rms, rolloff]:
        features.extend(np.mean(x, axis=1).tolist())
        features.extend(np.std(x, axis=1).tolist())

    return np.asarray(features, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        np.dot(a, b)
        / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12)
    )


def prepare_4_seconds(audio: np.ndarray) -> np.ndarray:
    if len(audio) >= DETECTOR_CHUNK_SAMPLES:
        return audio[:DETECTOR_CHUNK_SAMPLES].astype(np.float32)

    return np.pad(
        audio,
        (0, DETECTOR_CHUNK_SAMPLES - len(audio)),
        mode="constant",
    ).astype(np.float32)


def make_chunks(audio: np.ndarray) -> List[np.ndarray]:
    """
    The detector is documented for fixed-length speech segments.
    We use non-overlapping 4-second windows and pad the final window.
    """
    if len(audio) <= DETECTOR_CHUNK_SAMPLES:
        return [prepare_4_seconds(audio)]

    chunks = []

    for start in range(0, len(audio), DETECTOR_CHUNK_SAMPLES):
        chunk = audio[start:start + DETECTOR_CHUNK_SAMPLES]
        chunks.append(prepare_4_seconds(chunk))

    return chunks


# -----------------------------
# Models
# -----------------------------
@st.cache_resource(show_spinner="Loading ECAPA speaker model...")
def load_speaker_model():
    return SpeakerRecognition.from_hparams(
        source=SPEAKER_MODEL,
        savedir=None,
    )


@st.cache_resource(show_spinner="Loading synthetic-audio detector...")
def load_deepfake_model():
    extractor = AutoFeatureExtractor.from_pretrained(
        DEEPFAKE_MODEL
    )

    model = AutoModelForAudioClassification.from_pretrained(
        DEEPFAKE_MODEL
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    return extractor, model, device


# -----------------------------
# Speaker verification
# -----------------------------
def verify_speaker(
    speaker_model,
    reference_bytes: bytes,
    test_bytes: bytes,
):
    ref_audio, _ = load_audio_bytes(reference_bytes)
    test_audio, _ = load_audio_bytes(test_bytes)

    ref_path = "/tmp/vpf_reference.wav"
    test_path = "/tmp/vpf_test.wav"

    sf.write(ref_path, ref_audio, SAMPLE_RATE)
    sf.write(test_path, test_audio, SAMPLE_RATE)

    score, prediction = speaker_model.verify_files(
        ref_path,
        test_path,
    )

    score = float(score.squeeze().detach().cpu().item())
    same = bool(prediction.squeeze().detach().cpu().item())

    verdict = (
        "LIKELY SAME SPEAKER"
        if same
        else "LIKELY DIFFERENT SPEAKERS"
    )

    return score, verdict


# -----------------------------
# Deepfake detector
# -----------------------------
def resolve_fake_index(model):
    """
    Read the model's own label configuration rather than assuming
    LABEL_0/LABEL_1 ordering.
    """
    label2id = getattr(model.config, "label2id", {}) or {}
    id2label = getattr(model.config, "id2label", {}) or {}

    for label, idx in label2id.items():
        text = str(label).lower()
        if any(
            word in text
            for word in ["fake", "spoof", "synthetic", "deepfake"]
        ):
            return int(idx)

    for idx, label in id2label.items():
        text = str(label).lower()
        if any(
            word in text
            for word in ["fake", "spoof", "synthetic", "deepfake"]
        ):
            return int(idx)

    # Fallback for the common binary fine-tuned Wav2Vec2 convention.
    if len(id2label) == 2:
        return 1

    raise RuntimeError(
        "Could not identify the detector's fake/spoof class. "
        f"Model labels: {id2label}"
    )


def detector_chunk(
    audio: np.ndarray,
    extractor,
    model,
    device,
):
    audio = prepare_4_seconds(audio)

    inputs = extractor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        logits = model(**inputs).logits
        probabilities = F.softmax(logits, dim=-1)[0]

    fake_index = resolve_fake_index(model)
    fake_probability = float(
        probabilities[fake_index].detach().cpu().item()
    )

    return fake_probability


def analyze_deepfake(
    audio: np.ndarray,
    extractor,
    model,
    device,
):
    chunks = make_chunks(audio)
    scores = []

    progress = st.progress(
        0,
        text="Analyzing synthetic/spoof evidence...",
    )

    for i, chunk in enumerate(chunks):
        scores.append(
            detector_chunk(
                chunk,
                extractor,
                model,
                device,
            )
        )

        progress.progress(
            (i + 1) / len(chunks),
            text=f"Analyzing chunk {i + 1}/{len(chunks)}...",
        )

    progress.empty()

    scores = np.asarray(scores, dtype=np.float32)

    mean_fake = float(np.mean(scores))
    median_fake = float(np.median(scores))
    max_fake = float(np.max(scores))

    if mean_fake >= 0.80:
        verdict = "HIGH SYNTHETIC / SPOOF EVIDENCE"
    elif mean_fake >= 0.55:
        verdict = "POSSIBLE SYNTHETIC / SPOOF EVIDENCE"
    elif mean_fake <= 0.20:
        verdict = "LOW SYNTHETIC EVIDENCE"
    else:
        verdict = "INCONCLUSIVE"

    return {
        "mean": mean_fake,
        "median": median_fake,
        "maximum": max_fake,
        "scores": scores,
        "verdict": verdict,
        "chunks": len(chunks),
    }


# -----------------------------
# UI
# -----------------------------
st.title("🎙️ VoicePrint Forensics")
st.caption(
    "Reference-based speaker verification + synthetic-audio analysis"
)

st.warning(
    "This is an assistive forensic tool, not a courtroom-grade "
    "authentication system. A speaker match does not prove that a "
    "recording is genuine. Detector results can be wrong, especially "
    "for generators or recording conditions not represented in training."
)

with st.expander("How this version works"):
    st.markdown(
        """
        **Speaker verification**

        ECAPA-TDNN independently estimates whether the reference and
        test recordings are likely to belong to the same speaker.

        **Synthetic-audio detection**

        A Wav2Vec2 classifier analyzes 4-second windows and estimates
        whether they resemble its learned fake/spoof class.

        **Acoustic diagnostics**

        Spectral/MFCC statistics are displayed separately. They are
        diagnostic measurements and are deliberately NOT used as an
        authenticity percentage.
        """
    )

col1, col2 = st.columns(2)

with col1:
    st.subheader("Reference recording")
    reference_file = st.file_uploader(
        "Upload the known/reference voice",
        type=["wav", "mp3", "flac", "m4a", "ogg", "aac"],
        key="reference",
    )

with col2:
    st.subheader("Test recording")
    test_file = st.file_uploader(
        "Upload the voice you want to examine",
        type=["wav", "mp3", "flac", "m4a", "ogg", "aac"],
        key="test",
    )


if reference_file and test_file:

    reference_bytes = reference_file.getvalue()
    test_bytes = test_file.getvalue()

    reference_hash = sha256_bytes(reference_bytes)
    test_hash = sha256_bytes(test_bytes)

    st.divider()
    st.subheader("File information")

    info1, info2 = st.columns(2)

    with info1:
        st.markdown("### Reference")
        st.write(f"File: **{reference_file.name}**")
        st.write(f"SHA-256: `{reference_hash}`")

    with info2:
        st.markdown("### Test")
        st.write(f"File: **{test_file.name}**")
        st.write(f"SHA-256: `{test_hash}`")

    if reference_hash == test_hash:
        st.info(
            "The files are byte-for-byte identical."
        )

    try:
        reference_audio, reference_sr = load_audio_bytes(
            reference_bytes
        )
        test_audio, test_sr = load_audio_bytes(
            test_bytes
        )

        reference_stats = audio_stats(
            reference_audio,
            reference_sr,
        )
        test_stats = audio_stats(
            test_audio,
            test_sr,
        )

        st.subheader("Audio diagnostics")

        d1, d2 = st.columns(2)

        with d1:
            st.markdown("### Reference")
            st.write(
                f"Duration: **{reference_stats['duration']:.2f} s**"
            )
            st.write(
                f"RMS: **{reference_stats['rms']:.5f}**"
            )
            st.write(
                f"Spectral centroid: "
                f"**{reference_stats['spectral_centroid']:.1f} Hz**"
            )
            st.write(
                f"Silence ratio: "
                f"**{reference_stats['silence_ratio']:.1%}**"
            )

        with d2:
            st.markdown("### Test")
            st.write(
                f"Duration: **{test_stats['duration']:.2f} s**"
            )
            st.write(
                f"RMS: **{test_stats['rms']:.5f}**"
            )
            st.write(
                f"Spectral centroid: "
                f"**{test_stats['spectral_centroid']:.1f} Hz**"
            )
            st.write(
                f"Silence ratio: "
                f"**{test_stats['silence_ratio']:.1%}**"
            )

        st.divider()

        if st.button(
            "🔎 Analyze recordings",
            type="primary",
            use_container_width=True,
        ):

            try:
                with st.spinner("Loading forensic models..."):
                    speaker_model = load_speaker_model()
                    extractor, deepfake_model, detector_device = (
                        load_deepfake_model()
                    )

                # -------------------------
                # Speaker
                # -------------------------
                st.subheader("1. Speaker verification")

                with st.spinner(
                    "Comparing speaker characteristics..."
                ):
                    speaker_score, speaker_verdict = verify_speaker(
                        speaker_model,
                        reference_bytes,
                        test_bytes,
                    )

                if speaker_verdict == "LIKELY SAME SPEAKER":
                    st.success(speaker_verdict)
                else:
                    st.error(speaker_verdict)

                st.metric(
                    "ECAPA verification score",
                    f"{speaker_score:.4f}",
                )

                st.caption(
                    "Speaker-verification score only; it is NOT "
                    "an authenticity percentage."
                )

                # -------------------------
                # Synthetic audio
                # -------------------------
                st.subheader("2. Synthetic / spoof evidence")

                reference_result = analyze_deepfake(
                    reference_audio,
                    extractor,
                    deepfake_model,
                    detector_device,
                )

                test_result = analyze_deepfake(
                    test_audio,
                    extractor,
                    deepfake_model,
                    detector_device,
                )

                s1, s2 = st.columns(2)

                with s1:
                    st.markdown("### Reference")
                    st.metric(
                        "Mean fake probability",
                        f"{reference_result['mean']:.1%}",
                    )
                    st.write(
                        f"Median: **{reference_result['median']:.1%}**"
                    )
                    st.write(
                        f"Maximum chunk: "
                        f"**{reference_result['maximum']:.1%}**"
                    )
                    st.write(
                        f"Verdict: **{reference_result['verdict']}**"
                    )

                with s2:
                    st.markdown("### Test")
                    st.metric(
                        "Mean fake probability",
                        f"{test_result['mean']:.1%}",
                    )
                    st.write(
                        f"Median: **{test_result['median']:.1%}**"
                    )
                    st.write(
                        f"Maximum chunk: "
                        f"**{test_result['maximum']:.1%}**"
                    )
                    st.write(
                        f"Verdict: **{test_result['verdict']}**"
                    )

                st.caption(
                    "The detector is a learned classifier, not a universal "
                    "AI-voice detector. Its validation performance does "
                    "not guarantee performance on every voice generator."
                )

                # -------------------------
                # Acoustic diagnostic
                # -------------------------
                st.subheader("3. Acoustic diagnostics")

                ref_vector = acoustic_vector(
                    reference_audio,
                    reference_sr,
                )
                test_vector = acoustic_vector(
                    test_audio,
                    test_sr,
                )

                acoustic_similarity = cosine_similarity(
                    ref_vector,
                    test_vector,
                )

                st.metric(
                    "Diagnostic acoustic-vector similarity",
                    f"{acoustic_similarity:.4f}",
                )

                st.caption(
                    "Diagnostic only. High acoustic similarity is not "
                    "evidence that a recording is human or authentic."
                )

                # -------------------------
                # Summary
                # -------------------------
                st.subheader("Forensic evidence summary")

                st.write(
                    f"**Speaker:** {speaker_verdict}"
                )
                st.write(
                    f"**Reference synthetic evidence:** "
                    f"{reference_result['verdict']}"
                )
                st.write(
                    f"**Test synthetic evidence:** "
                    f"{test_result['verdict']}"
                )

                if (
                    speaker_verdict == "LIKELY SAME SPEAKER"
                    and test_result["mean"] >= 0.55
                ):
                    st.warning(
                        "The test recording is consistent with the "
                        "reference speaker while also showing synthetic/"
                        "spoof evidence. This combination can occur with "
                        "AI voice cloning, but the detector result alone "
                        "does not establish how the audio was produced."
                    )
                elif test_result["mean"] <= 0.20:
                    st.info(
                        "The current detector found low synthetic evidence. "
                        "This does NOT prove that the recording is genuine."
                    )
                else:
                    st.info(
                        "The synthetic-audio evidence is inconclusive. "
                        "Use the individual model outputs rather than "
                        "treating them as a single authenticity score."
                    )

            except Exception as exc:
                st.error("Analysis failed.")
                st.exception(exc)

    except Exception as exc:
        st.error(f"Could not decode the uploaded audio: {exc}")

else:
    st.info(
        "Upload both a reference recording and a test recording to begin."
    )

st.divider()

st.caption(
    "VoicePrint Forensics • ECAPA-TDNN speaker verification + "
    "Wav2Vec2 synthetic-audio analysis"
)
