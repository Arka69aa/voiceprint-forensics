
import hashlib
import io
import os
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
# Reference-based speaker verification + synthetic-audio analysis
# ============================================================

st.set_page_config(
    page_title="VoicePrint Forensics",
    page_icon="🎙️",
    layout="wide",
)

SAMPLE_RATE = 16000
CHUNK_SECONDS = 4.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SECONDS)

SPEAKER_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
DEEPFAKE_MODEL = "MelodyMachine/Deepfake-audio-detection-V2"


# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .metric-card {
        padding: 1rem;
        border: 1px solid rgba(255,255,255,.10);
        border-radius: 12px;
        background: rgba(255,255,255,.035);
        margin-bottom: .75rem;
    }
    .small-muted {
        color: #9aa7bd;
        font-size: .9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Utility functions
# -----------------------------
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_audio_bytes(data: bytes) -> Tuple[np.ndarray, int]:
    """Decode an uploaded audio file to mono 16 kHz float32."""
    audio, sr = sf.read(io.BytesIO(data), always_2d=False)

    audio = np.asarray(audio)

    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float32)

    # soundfile normally gives normalized floating point for common formats.
    # Protect against unusual integer-like ranges.
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
        raise ValueError("The uploaded file contains no decodable audio.")

    return audio, SAMPLE_RATE


def audio_stats(audio: np.ndarray, sr: int) -> Dict[str, float]:
    duration = len(audio) / sr
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio)) + 1e-12))

    zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio)[0]))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)[0]))
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
    rms_std = float(np.std(frame_rms))

    silence_ratio = float(np.mean(frame_rms < max(1e-5, rms * 0.12)))

    return {
        "duration": duration,
        "peak": peak,
        "rms": rms,
        "rms_std": rms_std,
        "zcr": zcr,
        "spectral_centroid": centroid,
        "spectral_rolloff": rolloff,
        "silence_ratio": silence_ratio,
    }


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def acoustic_vector(audio: np.ndarray, sr: int) -> np.ndarray:
    """Diagnostic acoustic representation. This is NOT a speaker embedding."""
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(audio)
    rms = librosa.feature.rms(y=audio)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)

    parts = []
    for x in [mfcc, contrast, chroma, zcr, rms, rolloff]:
        parts.extend(np.mean(x, axis=1).tolist())
        parts.extend(np.std(x, axis=1).tolist())

    return np.asarray(parts, dtype=np.float32)


def prepare_4s(audio: np.ndarray) -> np.ndarray:
    """Pad/truncate to the 4-second input length used by the detector."""
    if len(audio) >= CHUNK_SAMPLES:
        return audio[:CHUNK_SAMPLES].astype(np.float32)

    return np.pad(
        audio,
        (0, CHUNK_SAMPLES - len(audio)),
        mode="constant",
    ).astype(np.float32)


def make_chunks(audio: np.ndarray) -> List[np.ndarray]:
    """Create non-overlapping 4-second chunks, padding the final chunk."""
    if len(audio) <= CHUNK_SAMPLES:
        return [prepare_4s(audio)]

    chunks = []
    for start in range(0, len(audio), CHUNK_SAMPLES):
        chunk = audio[start:start + CHUNK_SAMPLES]
        if len(chunk) < CHUNK_SAMPLES:
            chunk = prepare_4s(chunk)
        chunks.append(chunk)

    return chunks


# -----------------------------
# Model loading
# -----------------------------
@st.cache_resource(show_spinner="Loading ECAPA-TDNN speaker model...")
def load_speaker_model():
    # savedir=None avoids the Windows symlink problem encountered during local use.
    return SpeakerRecognition.from_hparams(
        source=SPEAKER_MODEL,
        savedir=None,
    )


@st.cache_resource(show_spinner="Loading deepfake detection model...")
def load_deepfake_model():
    """
    MelodyMachine's checkpoint has a normal Wav2Vec2 config with
    model_type='wav2vec2', so it can be loaded directly through
    Transformers instead of the broken 0xmola pipeline configuration.
    """
    extractor = AutoFeatureExtractor.from_pretrained(DEEPFAKE_MODEL)
    model = AutoModelForAudioClassification.from_pretrained(DEEPFAKE_MODEL)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return extractor, model, device


# -----------------------------
# Speaker verification
# -----------------------------
def verify_speaker(
    speaker_model,
    ref_bytes: bytes,
    test_bytes: bytes,
) -> Tuple[float, str]:
    """
    ECAPA speaker verification.

    SpeechBrain's verify_batch returns a verification score and a
    thresholded prediction. We expose the score only as a speaker
    similarity/verification signal, not as an authenticity percentage.
    """
    ref_path = "/tmp/vpf_reference.wav"
    test_path = "/tmp/vpf_test.wav"

    ref_audio, _ = load_audio_bytes(ref_bytes)
    test_audio, _ = load_audio_bytes(test_bytes)

    sf.write(ref_path, ref_audio, SAMPLE_RATE)
    sf.write(test_path, test_audio, SAMPLE_RATE)

    score, prediction = speaker_model.verify_files(ref_path, test_path)

    score_value = float(score.squeeze().detach().cpu().item())
    same_speaker = bool(prediction.squeeze().detach().cpu().item())

    verdict = (
        "LIKELY SAME SPEAKER"
        if same_speaker
        else "LIKELY DIFFERENT SPEAKERS"
    )

    return score_value, verdict


# -----------------------------
# Deepfake detection
# -----------------------------
def find_label_index(model, wanted: str):
    wanted = wanted.lower()

    label2id = getattr(model.config, "label2id", {}) or {}

    for label, idx in label2id.items():
        if str(label).lower() == wanted:
            return int(idx)

    id2label = getattr(model.config, "id2label", {}) or {}

    for idx, label in id2label.items():
        if str(label).lower() == wanted:
            return int(idx)

    return None


def deepfake_chunk_score(
    audio_chunk: np.ndarray,
    extractor,
    model,
    device,
) -> Dict[str, float]:
    audio_chunk = prepare_4s(audio_chunk)

    inputs = extractor(
        audio_chunk,
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
        probs = F.softmax(logits, dim=-1)[0]

    spoof_idx = find_label_index(model, "fake")

    if spoof_idx is None:
        spoof_idx = find_label_index(model, "spoof")

    real_idx = find_label_index(model, "real")

    if real_idx is None:
        real_idx = find_label_index(model, "bonafide")

    if spoof_idx is None:
        raise RuntimeError(
            f"Could not identify the fake/spoof label. "
            f"Model labels: {getattr(model.config, 'id2label', {})}"
        )

    fake_probability = float(probs[spoof_idx].item())

    real_probability = (
        float(probs[real_idx].item())
        if real_idx is not None
        else float(1.0 - fake_probability)
    )

    return {
        "fake_probability": fake_probability,
        "real_probability": real_probability,
    }


def analyze_deepfake(
    audio: np.ndarray,
    extractor,
    model,
    device,
) -> Dict[str, object]:
    chunks = make_chunks(audio)

    scores = []

    progress = st.progress(
        0,
        text="Analyzing audio for synthetic/spoof evidence...",
    )

    for i, chunk in enumerate(chunks):
        result = deepfake_chunk_score(
            chunk,
            extractor,
            model,
            device,
        )
        scores.append(result)

        progress.progress(
            (i + 1) / len(chunks),
            text=f"Analyzing chunk {i + 1}/{len(chunks)}...",
        )

    progress.empty()

    fake_probs = np.asarray(
        [x["fake_probability"] for x in scores],
        dtype=np.float32,
    )

    # We intentionally report the mean model output rather than inventing
    # a calibrated "authenticity percentage".
    mean_fake = float(np.mean(fake_probs))
    max_fake = float(np.max(fake_probs))
    median_fake = float(np.median(fake_probs))

    if mean_fake >= 0.80:
        verdict = "HIGH SYNTHETIC / SPOOF EVIDENCE"
    elif mean_fake >= 0.55:
        verdict = "POSSIBLE SYNTHETIC / SPOOF EVIDENCE"
    elif mean_fake <= 0.20:
        verdict = "LOW SYNTHETIC EVIDENCE"
    else:
        verdict = "INCONCLUSIVE"

    return {
        "mean_fake_probability": mean_fake,
        "median_fake_probability": median_fake,
        "max_fake_probability": max_fake,
        "chunk_scores": fake_probs,
        "verdict": verdict,
        "chunk_count": len(chunks),
    }


# -----------------------------
# UI
# -----------------------------
st.title("🎙️ VoicePrint Forensics")
st.caption(
    "Reference-based speaker verification + synthetic-audio analysis"
)

st.warning(
    "This tool provides forensic indicators, not courtroom-grade "
    "authentication. Speaker similarity does not prove that an audio "
    "recording is genuine, and synthetic-audio detectors can produce "
    "false positives and false negatives."
)

with st.expander("How the analysis works", expanded=False):
    st.markdown(
        """
        **1. Speaker verification**

        ECAPA-TDNN compares the reference and test recordings for speaker
        characteristics. This answers **whether the recordings are likely
        from the same speaker**.

        **2. Synthetic/spoof analysis**

        A Wav2Vec2-based classifier analyzes 4-second audio windows and
        estimates whether each window resembles the **fake** or **real**
        classes it was trained on.

        **3. Acoustic diagnostics**

        Basic signal statistics are shown separately. They are supporting
        diagnostics and are not treated as proof of identity or authenticity.
        """
    )

col1, col2 = st.columns(2)

with col1:
    st.subheader("Reference recording")
    ref_file = st.file_uploader(
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

if ref_file is not None and test_file is not None:
    ref_bytes = ref_file.getvalue()
    test_bytes = test_file.getvalue()

    ref_hash = sha256_bytes(ref_bytes)
    test_hash = sha256_bytes(test_bytes)

    st.divider()

    st.subheader("File information")

    info1, info2 = st.columns(2)

    with info1:
        st.markdown("### Reference")
        st.write(f"**File:** {ref_file.name}")
        st.write(f"**SHA-256:** `{ref_hash}`")

    with info2:
        st.markdown("### Test")
        st.write(f"**File:** {test_file.name}")
        st.write(f"**SHA-256:** `{test_hash}`")

    if ref_hash == test_hash:
        st.info(
            "The two uploaded files are byte-for-byte identical. "
            "This is a file identity result, not a speaker/authenticity result."
        )

    try:
        ref_audio, ref_sr = load_audio_bytes(ref_bytes)
        test_audio, test_sr = load_audio_bytes(test_bytes)

        ref_stats = audio_stats(ref_audio, ref_sr)
        test_stats = audio_stats(test_audio, test_sr)

        st.subheader("Audio diagnostics")

        diag1, diag2 = st.columns(2)

        with diag1:
            st.markdown("### Reference")
            st.write(f"Duration: **{ref_stats['duration']:.2f} s**")
            st.write(f"RMS: **{ref_stats['rms']:.5f}**")
            st.write(f"Peak: **{ref_stats['peak']:.5f}**")
            st.write(
                f"Spectral centroid: **{ref_stats['spectral_centroid']:.1f} Hz**"
            )
            st.write(
                f"Spectral rolloff: **{ref_stats['spectral_rolloff']:.1f} Hz**"
            )
            st.write(f"Silence ratio: **{ref_stats['silence_ratio']:.1%}**")

        with diag2:
            st.markdown("### Test")
            st.write(f"Duration: **{test_stats['duration']:.2f} s**")
            st.write(f"RMS: **{test_stats['rms']:.5f}**")
            st.write(f"Peak: **{test_stats['peak']:.5f}**")
            st.write(
                f"Spectral centroid: **{test_stats['spectral_centroid']:.1f} Hz**"
            )
            st.write(
                f"Spectral rolloff: **{test_stats['spectral_rolloff']:.1f} Hz**"
            )
            st.write(f"Silence ratio: **{test_stats['silence_ratio']:.1%}**")

        st.divider()

        if st.button(
            "🔎 Analyze recordings",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner("Loading forensic models..."):
                    speaker_model = load_speaker_model()
                    extractor, deepfake_model, device = load_deepfake_model()

                # -------------------------
                # Speaker verification
                # -------------------------
                st.subheader("1. Speaker verification")

                with st.spinner("Comparing speaker characteristics..."):
                    speaker_score, speaker_verdict = verify_speaker(
                        speaker_model,
                        ref_bytes,
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
                    "This score is a speaker-verification signal. "
                    "It is not a percentage of authenticity."
                )

                # -------------------------
                # Deepfake analysis
                # -------------------------
                st.subheader("2. Synthetic / spoof evidence")

                with st.spinner("Running chunk-level synthetic-audio analysis..."):
                    ref_fake = analyze_deepfake(
                        ref_audio,
                        extractor,
                        deepfake_model,
                        device,
                    )

                    test_fake = analyze_deepfake(
                        test_audio,
                        extractor,
                        deepfake_model,
                        device,
                    )

                d1, d2 = st.columns(2)

                with d1:
                    st.markdown("### Reference")
                    st.metric(
                        "Mean fake probability",
                        f"{ref_fake['mean_fake_probability']:.1%}",
                    )
                    st.write(
                        f"Median: **{ref_fake['median_fake_probability']:.1%}**"
                    )
                    st.write(
                        f"Maximum chunk: **{ref_fake['max_fake_probability']:.1%}**"
                    )
                    st.write(f"Verdict: **{ref_fake['verdict']}**")

                with d2:
                    st.markdown("### Test")
                    st.metric(
                        "Mean fake probability",
                        f"{test_fake['mean_fake_probability']:.1%}",
                    )
                    st.write(
                        f"Median: **{test_fake['median_fake_probability']:.1%}**"
                    )
                    st.write(
                        f"Maximum chunk: **{test_fake['max_fake_probability']:.1%}**"
                    )
                    st.write(f"Verdict: **{test_fake['verdict']}**")

                st.caption(
                    "The detector was trained around ASVspoof-style spoofing "
                    "data. Performance can change substantially on codecs, "
                    "re-recordings, languages, microphones, noise, and "
                    "generative systems not represented in training data."
                )

                # -------------------------
                # Acoustic comparison
                # -------------------------
                st.subheader("3. Acoustic diagnostics")

                ref_vec = acoustic_vector(ref_audio, ref_sr)
                test_vec = acoustic_vector(test_audio, test_sr)

                acoustic_sim = cosine_similarity(ref_vec, test_vec)

                st.metric(
                    "Diagnostic acoustic-vector similarity",
                    f"{acoustic_sim:.4f}",
                )

                st.caption(
                    "This is only a broad acoustic diagnostic. Do not use it "
                    "as a speaker identity or authenticity score."
                )

                # -------------------------
                # Final evidence summary
                # -------------------------
                st.subheader("Forensic evidence summary")

                summary = [
                    ("Speaker verification", speaker_verdict),
                    ("Reference synthetic evidence", ref_fake["verdict"]),
                    ("Test synthetic evidence", test_fake["verdict"]),
                ]

                for name, value in summary:
                    st.write(f"**{name}:** {value}")

                st.info(
                    "Interpret the three outputs independently. A recording "
                    "can be from the same speaker and still be synthetic, "
                    "edited, replayed, or otherwise manipulated."
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
