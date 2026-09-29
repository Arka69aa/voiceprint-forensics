
import gc
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
# VoicePrint Forensics — v3
# Reference-based speaker verification + specialist TTS detector
# ============================================================

st.set_page_config(
    page_title="VoicePrint Forensics",
    page_icon="🎙️",
    layout="wide",
)

SR = 16000

# ECAPA is used for speaker verification.
SPEAKER_MODEL = "speechbrain/spkrec-ecapa-voxceleb"

# Specialist model trained specifically on human vs AI-generated speech.
# Its training set includes ElevenLabs, Amazon Polly, Kokoro, Hume AI,
# Speechify and Luvvoice.
DEEPFAKE_MODEL = "garystafford/wav2vec2-deepfake-voice-detector"

# The detector was trained on 2.5–13 second clips.
MIN_CLIP_SECONDS = 2.5
MAX_CLIP_SECONDS = 13.0
WINDOW_SECONDS = 8.0
HOP_SECONDS = 4.0


# -----------------------------
# Page style
# -----------------------------
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Audio
# -----------------------------
def load_audio(data: bytes) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(io.BytesIO(data), always_2d=False)
    audio = np.asarray(audio)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float32)

    if audio.size == 0:
        raise ValueError("The file contains no audio.")

    peak = float(np.max(np.abs(audio)))
    if peak > 1.5:
        audio = audio / peak

    if sr != SR:
        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=SR,
        )

    return np.asarray(audio, dtype=np.float32), SR


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def audio_stats(audio: np.ndarray, sr: int) -> Dict[str, float]:
    rms = float(np.sqrt(np.mean(audio * audio) + 1e-12))
    frame_rms = librosa.feature.rms(y=audio)[0]

    return {
        "duration": len(audio) / sr,
        "rms": rms,
        "peak": float(np.max(np.abs(audio))),
        "centroid": float(
            np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)[0])
        ),
        "rolloff": float(
            np.mean(
                librosa.feature.spectral_rolloff(
                    y=audio,
                    sr=sr,
                    roll_percent=0.85,
                )[0]
            )
        ),
        "silence": float(
            np.mean(frame_rms < max(1e-5, rms * 0.12))
        ),
    }


def make_detector_windows(audio: np.ndarray) -> List[np.ndarray]:
    """
    Use 8-second windows with 4-second overlap.

    This stays inside the specialist detector's documented
    2.5–13 second training range and gives us multiple observations
    rather than relying on a single beginning-of-file crop.
    """
    n = len(audio)
    window = int(WINDOW_SECONDS * SR)
    hop = int(HOP_SECONDS * SR)

    if n <= window:
        if n < int(MIN_CLIP_SECONDS * SR):
            # Too short for the detector to be meaningful.
            return []

        padded = np.pad(
            audio,
            (0, window - n),
            mode="constant",
        )
        return [padded.astype(np.float32)]

    windows = []

    start = 0
    while start < n:
        end = start + window
        chunk = audio[start:end]

        if len(chunk) < int(MIN_CLIP_SECONDS * SR):
            break

        if len(chunk) < window:
            chunk = np.pad(
                chunk,
                (0, window - len(chunk)),
                mode="constant",
            )

        windows.append(chunk.astype(np.float32))

        if end >= n:
            break

        start += hop

    return windows


# -----------------------------
# Specialist detector
# -----------------------------
def load_deepfake_detector():
    # IMPORTANT:
    # This repository contains preprocessor_config.json but no tokenizer.
    # AutoProcessor tries to construct a Wav2Vec2CTCTokenizer and therefore
    # fails. This is audio classification, so we only need the feature
    # extractor.
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        DEEPFAKE_MODEL
    )

    model = AutoModelForAudioClassification.from_pretrained(
        DEEPFAKE_MODEL,
        low_cpu_mem_usage=True,
    )

    model.eval()

    return feature_extractor, model


def fake_index(model) -> int:
    """
    Gary Stafford's model card defines:
      class 0 = real
      class 1 = fake
    """
    label2id = getattr(model.config, "label2id", {}) or {}
    id2label = getattr(model.config, "id2label", {}) or {}

    for label, idx in label2id.items():
        label = str(label).lower()
        if any(x in label for x in ["fake", "synthetic", "spoof"]):
            return int(idx)

    for idx, label in id2label.items():
        label = str(label).lower()
        if any(x in label for x in ["fake", "synthetic", "spoof"]):
            return int(idx)

    # Documented fallback for this checkpoint.
    return 1


def detector_score(
    audio: np.ndarray,
    feature_extractor,
    model,
) -> float:
    # Normalize per clip as recommended for robustness.
    audio = audio.astype(np.float32)
    audio = audio - np.mean(audio)
    std = float(np.std(audio))

    if std > 1e-7:
        audio = audio / std

    inputs = feature_extractor(
        audio,
        sampling_rate=SR,
        return_tensors="pt",
        padding=True,
    )

    with torch.inference_mode():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0]

    return float(
        probs[fake_index(model)].cpu().item()
    )


def run_deepfake_detection(
    audio: np.ndarray,
    feature_extractor,
    model,
) -> Dict[str, object]:

    windows = make_detector_windows(audio)

    if not windows:
        return {
            "status": "too_short",
            "scores": [],
            "mean": None,
            "median": None,
            "maximum": None,
            "verdict": "INSUFFICIENT AUDIO",
        }

    scores = []

    progress = st.progress(
        0,
        text="Running specialist synthetic-speech detector...",
    )

    for i, window in enumerate(windows):
        scores.append(
            detector_score(
                window,
                feature_extractor,
                model,
            )
        )

        progress.progress(
            (i + 1) / len(windows),
            text=f"Detector window {i + 1}/{len(windows)}...",
        )

    progress.empty()

    scores = np.asarray(scores, dtype=np.float32)

    mean_score = float(np.mean(scores))
    median_score = float(np.median(scores))
    max_score = float(np.max(scores))

    # Conservative interpretation:
    # A low score does NOT mean "proven real".
    if mean_score >= 0.90:
        verdict = "STRONG SYNTHETIC EVIDENCE"
    elif mean_score >= 0.70:
        verdict = "SYNTHETIC EVIDENCE"
    elif mean_score >= 0.45:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "NO SYNTHETIC EVIDENCE FROM THIS MODEL"

    return {
        "status": "ok",
        "scores": scores,
        "mean": mean_score,
        "median": median_score,
        "maximum": max_score,
        "verdict": verdict,
    }


# -----------------------------
# Speaker verification
# -----------------------------
def verify_speaker(
    reference_bytes: bytes,
    test_bytes: bytes,
):
    # Write normalized 16-kHz mono WAV files for SpeechBrain.
    ref_audio, _ = load_audio(reference_bytes)
    test_audio, _ = load_audio(test_bytes)

    ref_path = "/tmp/vpf_reference.wav"
    test_path = "/tmp/vpf_test.wav"

    sf.write(ref_path, ref_audio, SR)
    sf.write(test_path, test_audio, SR)

    verifier = SpeakerRecognition.from_hparams(
        source=SPEAKER_MODEL,
        savedir=None,
    )

    score, prediction = verifier.verify_files(
        ref_path,
        test_path,
    )

    score = float(score.squeeze().detach().cpu().item())
    same = bool(prediction.squeeze().detach().cpu().item())

    # Explicitly release the model after the operation.
    del verifier
    gc.collect()

    return score, (
        "LIKELY SAME SPEAKER"
        if same
        else "LIKELY DIFFERENT SPEAKERS"
    )


# -----------------------------
# Acoustic diagnostics
# -----------------------------
def acoustic_similarity(
    a: np.ndarray,
    b: np.ndarray,
    sr: int,
) -> float:
    def vec(x):
        mfcc = librosa.feature.mfcc(
            y=x,
            sr=sr,
            n_mfcc=20,
        )

        contrast = librosa.feature.spectral_contrast(
            y=x,
            sr=sr,
        )

        zcr = librosa.feature.zero_crossing_rate(x)
        rms = librosa.feature.rms(y=x)

        pieces = []
        for f in [mfcc, contrast, zcr, rms]:
            pieces.extend(np.mean(f, axis=1))
            pieces.extend(np.std(f, axis=1))

        return np.asarray(pieces, dtype=np.float32)

    x = vec(a)
    y = vec(b)

    return float(
        np.dot(x, y)
        / ((np.linalg.norm(x) * np.linalg.norm(y)) + 1e-12)
    )


# -----------------------------
# UI
# -----------------------------
st.title("🎙️ VoicePrint Forensics")
st.caption(
    "Reference-based speaker verification + specialist synthetic-speech detection"
)

st.warning(
    "Important: no current detector can guarantee detection of every "
    "AI-generated or voice-cloned recording. A low fake probability means "
    "only that this model did not find sufficient evidence."
)

with st.expander("What this version fixes"):
    st.markdown(
        """
        **Speaker identity and authenticity are separated.**

        • ECAPA-TDNN answers whether the two recordings are likely from the
        same speaker.

        • The specialist Wav2Vec2 detector evaluates whether speech resembles
        AI-generated speech.

        • Multiple overlapping windows are analyzed rather than one arbitrary
        beginning-of-file crop.

        • The application never converts acoustic similarity into an
        authenticity percentage.

        • "No synthetic evidence" is deliberately NOT labelled "AUTHENTIC."
        """
    )

left, right = st.columns(2)

with left:
    st.subheader("Reference recording")
    reference_file = st.file_uploader(
        "Known/reference voice",
        type=["wav", "mp3", "flac", "m4a", "ogg", "aac"],
        key="reference",
    )

with right:
    st.subheader("Test recording")
    test_file = st.file_uploader(
        "Voice to examine",
        type=["wav", "mp3", "flac", "m4a", "ogg", "aac"],
        key="test",
    )


if reference_file and test_file:

    reference_bytes = reference_file.getvalue()
    test_bytes = test_file.getvalue()

    if sha256(reference_bytes) == sha256(test_bytes):
        st.error(
            "The two files are byte-for-byte identical. "
            "Upload separate reference and test recordings."
        )
        st.stop()

    try:
        reference_audio, _ = load_audio(reference_bytes)
        test_audio, _ = load_audio(test_bytes)

        reference_stats = audio_stats(
            reference_audio,
            SR,
        )
        test_stats = audio_stats(
            test_audio,
            SR,
        )

        st.divider()
        st.subheader("Audio diagnostics")

        a, b = st.columns(2)

        with a:
            st.markdown("### Reference")
            st.write(
                f"Duration: **{reference_stats['duration']:.2f} s**"
            )
            st.write(
                f"RMS: **{reference_stats['rms']:.5f}**"
            )
            st.write(
                f"Spectral centroid: "
                f"**{reference_stats['centroid']:.1f} Hz**"
            )

        with b:
            st.markdown("### Test")
            st.write(
                f"Duration: **{test_stats['duration']:.2f} s**"
            )
            st.write(
                f"RMS: **{test_stats['rms']:.5f}**"
            )
            st.write(
                f"Spectral centroid: "
                f"**{test_stats['centroid']:.1f} Hz**"
            )

        if st.button(
            "🔎 Run forensic analysis",
            type="primary",
            use_container_width=True,
        ):

            # ==================================================
            # 1. Synthetic detector FIRST
            # ==================================================
            st.subheader("1. Synthetic / AI-voice detection")

            try:
                feature_extractor, deepfake_model = load_deepfake_detector()

                reference_result = run_deepfake_detection(
                    reference_audio,
                    feature_extractor,
                    deepfake_model,
                )

                test_result = run_deepfake_detection(
                    test_audio,
                    feature_extractor,
                    deepfake_model,
                )

                # Release the large Wav2Vec2 model BEFORE loading ECAPA.
                del feature_extractor
                del deepfake_model
                gc.collect()

                c1, c2 = st.columns(2)

                with c1:
                    st.markdown("### Reference")
                    if reference_result["mean"] is not None:
                        st.metric(
                            "Mean AI/fake probability",
                            f"{reference_result['mean']:.1%}",
                        )
                        st.write(
                            f"Median: "
                            f"**{reference_result['median']:.1%}**"
                        )
                        st.write(
                            f"Maximum: "
                            f"**{reference_result['maximum']:.1%}**"
                        )
                    st.write(
                        f"**{reference_result['verdict']}**"
                    )

                with c2:
                    st.markdown("### Test")
                    if test_result["mean"] is not None:
                        st.metric(
                            "Mean AI/fake probability",
                            f"{test_result['mean']:.1%}",
                        )
                        st.write(
                            f"Median: "
                            f"**{test_result['median']:.1%}**"
                        )
                        st.write(
                            f"Maximum: "
                            f"**{test_result['maximum']:.1%}**"
                        )
                    st.write(
                        f"**{test_result['verdict']}**"
                    )

            except Exception as exc:
                st.error(
                    "The specialist detector could not run."
                )
                st.exception(exc)
                reference_result = None
                test_result = None

            # ==================================================
            # 2. Speaker verification
            # ==================================================
            st.subheader("2. Speaker verification")

            try:
                with st.spinner(
                    "Comparing speaker characteristics..."
                ):
                    speaker_score, speaker_verdict = verify_speaker(
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
                    "This is a speaker-verification score, not an "
                    "authenticity percentage."
                )

            except Exception as exc:
                st.error(
                    "Speaker verification failed."
                )
                st.exception(exc)
                speaker_verdict = "UNAVAILABLE"
                speaker_score = None

            # ==================================================
            # 3. Acoustic diagnostics
            # ==================================================
            st.subheader("3. Acoustic diagnostics")

            similarity = acoustic_similarity(
                reference_audio,
                test_audio,
                SR,
            )

            st.metric(
                "Diagnostic acoustic similarity",
                f"{similarity:.4f}",
            )

            st.caption(
                "Diagnostic only. This number is deliberately excluded "
                "from the synthetic/real decision."
            )

            # ==================================================
            # 4. Evidence interpretation
            # ==================================================
            st.subheader("4. Evidence interpretation")

            if test_result is None:
                st.error(
                    "INCONCLUSIVE — the synthetic detector did not run."
                )

            elif (
                speaker_verdict == "LIKELY SAME SPEAKER"
                and test_result["mean"] >= 0.70
            ):
                st.error(
                    "SAME-SPEAKER + SYNTHETIC EVIDENCE"
                )
                st.write(
                    "The test recording is consistent with the reference "
                    "speaker while the specialist detector also reports "
                    "substantial AI-generated speech evidence."
                )

            elif test_result["mean"] >= 0.70:
                st.error(
                    "SYNTHETIC EVIDENCE DETECTED"
                )

            elif test_result["mean"] >= 0.45:
                st.warning(
                    "INCONCLUSIVE"
                )
                st.write(
                    "The detector did not reach a strong decision. "
                    "This should not be interpreted as proof of genuine audio."
                )

            else:
                st.info(
                    "NO SYNTHETIC EVIDENCE FROM THIS MODEL"
                )
                st.write(
                    "The detector currently finds insufficient evidence "
                    "of AI generation. This does NOT prove the recording "
                    "is genuine."
                )

            st.divider()

            st.subheader("Forensic summary")

            st.write(
                f"**Speaker comparison:** {speaker_verdict}"
            )

            if test_result is not None:
                st.write(
                    f"**Synthetic detector:** "
                    f"{test_result['verdict']}"
                )

            st.write(
                f"**Acoustic similarity:** "
                f"{similarity:.4f} (diagnostic only)"
            )

            st.caption(
                "Model limitations: the specialist detector was trained "
                "on specific TTS/voice-cloning families and can fail on "
                "unseen generators, codecs, re-recordings, noise and "
                "voice-conversion systems."
            )

    except Exception as exc:
        st.error("Could not decode the uploaded audio.")
        st.exception(exc)

else:
    st.info(
        "Upload both a reference recording and a test recording."
    )

st.divider()
st.caption(
    "VoicePrint Forensics v3 • ECAPA-TDNN + specialist Wav2Vec2 "
    "synthetic-speech detector"
)
