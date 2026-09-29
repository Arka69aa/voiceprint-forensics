import os
import hashlib
import tempfile
import warnings
from pathlib import Path

import numpy as np
import streamlit as st
import librosa
import torch

warnings.filterwarnings("ignore")

APP_TITLE = "VoicePrint Forensics"
SAMPLE_RATE = 16000

SPEAKER_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
DEEPFAKE_MODEL = "0xmola/wavlm-deepfake-audio-forensics"

MIN_DURATION = 2.0
RECOMMENDED_DURATION = 4.0

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #0b0f1a; }
    h1 { color: #00d4ff; font-family: "Courier New", monospace;
         text-shadow: 0 0 15px rgba(0,212,255,.6); }
    h2, h3 { color: #e8f1ff; }
    .metric-card { background:#171d35; border-radius:12px; padding:18px;
                   border-left:4px solid #00d4ff; margin-bottom:10px; }
    .same-speaker { border-left-color:#00ff88; }
    .different-speaker { border-left-color:#ff5577; }
    .uncertain { border-left-color:#ffaa00; }
    .real { border-left-color:#00ff88; }
    .spoof { border-left-color:#ff3366; }
    .warning { border-left-color:#ffaa00; }
    .neutral { border-left-color:#888; }
    .small-note { color:#9ca8bd; font-size:.85rem; }
    .disclaimer { background:#151a2b; border:1px solid #29314d;
                  border-radius:10px; padding:15px; color:#aeb9cc;
                  font-size:.88rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔍 VoicePrint Forensics")
st.caption("Reference-based speaker verification + independent synthetic-audio analysis")
st.markdown(
    """
    <div class="disclaimer">
    <b>Important:</b> This tool produces forensic evidence, not courtroom-grade
    authentication. Speaker similarity does not prove that an audio recording is
    genuine, and a synthetic-audio detector can fail on unseen generators, codecs,
    languages, noise conditions, or manipulated recordings.
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")

if "results" not in st.session_state:
    st.session_state.results = None


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_uploaded_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix or ".wav"
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(uploaded_file.getvalue())
    return path


def load_audio(audio_path):
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    return audio.astype(np.float32), SAMPLE_RATE


def audio_statistics(audio, sr):
    duration = len(audio) / sr if sr else 0
    peak = float(np.max(np.abs(audio))) if len(audio) else 0
    rms = librosa.feature.rms(y=audio)[0]
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    return {
        "duration": duration,
        "peak": peak,
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "zcr": float(np.mean(zcr)),
        "centroid": float(np.mean(centroid)),
        "rolloff": float(np.mean(rolloff)),
        "silence_ratio": float(np.mean(rms < 0.01)),
    }


@st.cache_resource(show_spinner=False)
def load_speaker_model():
    try:
        from speechbrain.inference.speaker import SpeakerRecognition
    except ImportError:
        from speechbrain.pretrained import SpeakerRecognition

    # Use the HF cache directly; avoids the Windows symlink problem.
    return SpeakerRecognition.from_hparams(
        source=SPEAKER_MODEL,
        savedir=None,
    )


@st.cache_resource(show_spinner=False)
def load_deepfake_model():
    from transformers import pipeline
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "audio-classification",
        model=DEEPFAKE_MODEL,
        device=device,
    )


def verify_speaker(model, reference_path, suspicious_path):
    score, prediction = model.verify_files(reference_path, suspicious_path)
    score = float(score.squeeze().cpu().item())
    prediction = bool(prediction.item()) if hasattr(prediction, "item") else bool(prediction)
    return score, prediction


def normalize_label(label):
    return label.lower().replace("_", "").replace("-", "").replace(" ", "")


def interpret_deepfake_results(results):
    if not results:
        raise RuntimeError("The synthetic-audio model returned no predictions.")

    fake_score = None
    real_score = None

    for result in results:
        label = normalize_label(str(result.get("label", "")))
        score = float(result.get("score", 0))
        if any(x in label for x in ("fake", "spoof", "synthetic", "deepfake", "generated")):
            fake_score = max(fake_score or 0, score)
        elif any(x in label for x in ("real", "bonafide", "genuine", "human")):
            real_score = max(real_score or 0, score)

    if fake_score is None and real_score is None:
        if len(results) >= 2:
            fake_score = float(results[0]["score"])
            real_score = float(results[1]["score"])
        else:
            fake_score = 0.0
            real_score = float(results[0]["score"])
    elif fake_score is None:
        fake_score = max(0.0, 1.0 - real_score)
    elif real_score is None:
        real_score = max(0.0, 1.0 - fake_score)

    total = fake_score + real_score
    if total > 0:
        fake_probability = fake_score / total
        real_probability = real_score / total
    else:
        fake_probability = real_probability = 0.5

    if fake_probability >= 0.75:
        label = "HIGH SYNTHETIC / SPOOF EVIDENCE"
    elif fake_probability >= 0.55:
        label = "POSSIBLE SYNTHETIC / SPOOF EVIDENCE"
    elif fake_probability <= 0.25:
        label = "LOW SYNTHETIC EVIDENCE"
    else:
        label = "INCONCLUSIVE"

    return {
        "fake_probability": fake_probability,
        "real_probability": real_probability,
        "label": label,
        "raw": results,
    }


def run_deepfake_detector(detector, audio_path):
    audio, sr = load_audio(audio_path)
    chunk_length = 4 * sr
    hop_length = 2 * sr
    chunks = []

    if len(audio) <= chunk_length:
        chunks.append(audio)
    else:
        start = 0
        while start < len(audio):
            chunk = audio[start:start + chunk_length]
            if len(chunk) >= int(2.0 * sr):
                chunks.append(chunk)
            start += hop_length

    predictions = []
    for chunk in chunks:
        result = detector({"sampling_rate": sr, "raw": chunk})
        predictions.append(interpret_deepfake_results(result))

    fake_probability = float(np.mean([p["fake_probability"] for p in predictions]))
    real_probability = float(np.mean([p["real_probability"] for p in predictions]))

    if fake_probability >= 0.75:
        label = "HIGH SYNTHETIC / SPOOF EVIDENCE"
    elif fake_probability >= 0.55:
        label = "POSSIBLE SYNTHETIC / SPOOF EVIDENCE"
    elif fake_probability <= 0.25:
        label = "LOW SYNTHETIC EVIDENCE"
    else:
        label = "INCONCLUSIVE"

    return {
        "fake_probability": fake_probability,
        "real_probability": real_probability,
        "label": label,
        "chunks": len(predictions),
        "chunk_results": predictions,
    }


def compare_acoustics(a, b):
    def pct(x, y):
        return abs(x - y) / max(abs(x), abs(y), 1e-9) * 100

    duration_diff = abs(a["duration"] - b["duration"])
    rms_diff = pct(a["rms_mean"], b["rms_mean"])
    zcr_diff = pct(a["zcr"], b["zcr"])
    centroid_diff = pct(a["centroid"], b["centroid"])
    rolloff_diff = pct(a["rolloff"], b["rolloff"])
    silence_diff = abs(a["silence_ratio"] - b["silence_ratio"])

    differences = []
    if duration_diff > 5:
        differences.append(f"Recording duration differs by {duration_diff:.2f} seconds.")
    if rms_diff > 35:
        differences.append(f"Mean RMS energy differs by {rms_diff:.1f}%.")
    if zcr_diff > 20:
        differences.append(f"Zero-crossing characteristics differ by {zcr_diff:.1f}%.")
    if centroid_diff > 30:
        differences.append(f"Spectral centroid differs by {centroid_diff:.1f}%.")
    if rolloff_diff > 30:
        differences.append(f"Spectral rolloff differs by {rolloff_diff:.1f}%.")
    if silence_diff > 0.20:
        differences.append(
            f"Silence ratio differs by {silence_diff * 100:.1f} percentage points."
        )

    return {
        "duration_difference": duration_diff,
        "rms_difference": rms_diff,
        "zcr_difference": zcr_diff,
        "centroid_difference": centroid_diff,
        "rolloff_difference": rolloff_diff,
        "silence_difference": silence_diff,
        "differences": differences,
    }


def assess_quality(stats):
    warnings_list = []
    if stats["duration"] < MIN_DURATION:
        warnings_list.append("Recording is shorter than 2 seconds.")
    elif stats["duration"] < RECOMMENDED_DURATION:
        warnings_list.append("Recording is short; around 4+ seconds of clean speech is preferable.")
    if stats["peak"] < 0.01:
        warnings_list.append("Recording is nearly silent.")
    if stats["silence_ratio"] > 0.65:
        warnings_list.append("A large portion of the recording is silence.")
    if stats["rms_mean"] < 0.01:
        warnings_list.append("Speech level is very low.")
    return warnings_list


def speaker_interpretation(score, prediction):
    if prediction:
        return "LIKELY SAME SPEAKER", "The speaker-verification model accepted the pair as the same speaker."
    return "LIKELY DIFFERENT SPEAKERS", "The speaker-verification model rejected the pair as the same speaker."


with st.sidebar:
    st.header("⚙️ Analysis Settings")
    st.markdown(
        """
        **Speaker model**

        ECAPA-TDNN / VoxCeleb

        **Synthetic detector**

        WavLM-based audio anti-spoofing model
        """
    )
    st.markdown("---")
    st.info("The first analysis may take longer because the pretrained models must be downloaded.")
    st.markdown(
        """
        ### Recommended audio
        • WAV or MP3  
        • 4+ seconds  
        • Clear speech  
        • Minimal background music  
        • Avoid extremely compressed audio
        """
    )


left, right = st.columns(2)

with left:
    st.subheader("🎯 Reference Audio")
    reference_file = st.file_uploader(
        "Upload the known/reference recording",
        type=["wav", "mp3", "flac", "m4a", "ogg"],
        key="reference",
    )
    if reference_file:
        st.audio(reference_file)

with right:
    st.subheader("🚨 Suspicious Audio")
    suspicious_file = st.file_uploader(
        "Upload the recording you want to examine",
        type=["wav", "mp3", "flac", "m4a", "ogg"],
        key="suspicious",
    )
    if suspicious_file:
        st.audio(suspicious_file)


if reference_file and suspicious_file:
    reference_bytes = reference_file.getvalue()
    suspicious_bytes = suspicious_file.getvalue()

    if file_hash(reference_bytes) == file_hash(suspicious_bytes):
        st.error("❌ These are the exact same file. Upload two different recordings.")
    elif st.button("🔬 RUN FULL FORENSIC ANALYSIS", type="primary", use_container_width=True):
        reference_path = None
        suspicious_path = None

        try:
            reference_path = save_uploaded_file(reference_file)
            suspicious_path = save_uploaded_file(suspicious_file)

            with st.status("Preparing audio...", expanded=True) as status:
                st.write("Loading reference audio...")
                reference_audio, sr = load_audio(reference_path)
                st.write("Loading suspicious audio...")
                suspicious_audio, _ = load_audio(suspicious_path)
                reference_stats = audio_statistics(reference_audio, sr)
                suspicious_stats = audio_statistics(suspicious_audio, sr)
                status.update(label="Audio prepared", state="complete")

            reference_quality = assess_quality(reference_stats)
            suspicious_quality = assess_quality(suspicious_stats)

            with st.status("Loading forensic models...", expanded=True) as status:
                st.write("Loading ECAPA-TDNN speaker verifier...")
                speaker_model = load_speaker_model()
                st.write("Loading WavLM synthetic-audio detector...")
                deepfake_model = load_deepfake_model()
                status.update(label="Models ready", state="complete")

            with st.status("Running speaker verification...", expanded=False):
                speaker_score, same_speaker = verify_speaker(
                    speaker_model, reference_path, suspicious_path
                )

            with st.status("Running synthetic-audio analysis...", expanded=False):
                deepfake_result = run_deepfake_detector(
                    deepfake_model, suspicious_path
                )

            acoustic_result = compare_acoustics(reference_stats, suspicious_stats)

            st.session_state.results = {
                "speaker_score": speaker_score,
                "same_speaker": same_speaker,
                "deepfake": deepfake_result,
                "reference_stats": reference_stats,
                "suspicious_stats": suspicious_stats,
                "reference_quality": reference_quality,
                "suspicious_quality": suspicious_quality,
                "acoustic": acoustic_result,
            }

            st.success("✅ Analysis completed.")

        except Exception as e:
            st.error("❌ Analysis failed.")
            st.exception(e)

        finally:
            for path in (reference_path, suspicious_path):
                if path and os.path.exists(path):
                    os.remove(path)


results = st.session_state.results

if results:
    st.markdown("---")
    st.header("📊 Forensic Analysis Results")

    speaker_score = results["speaker_score"]
    same_speaker = results["same_speaker"]
    deepfake = results["deepfake"]
    reference_stats = results["reference_stats"]
    suspicious_stats = results["suspicious_stats"]
    reference_quality = results["reference_quality"]
    suspicious_quality = results["suspicious_quality"]
    acoustic = results["acoustic"]

    col1, col2, col3 = st.columns(3)

    with col1:
        speaker_label, _ = speaker_interpretation(speaker_score, same_speaker)
        css_class = "same-speaker" if same_speaker else "different-speaker"
        st.markdown(
            f"""
            <div class="metric-card {css_class}">
                <div class="small-note">SPEAKER VERIFICATION</div>
                <h2>{speaker_label}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.metric("Model Verification Score", f"{speaker_score:.4f}")

    with col2:
        fake_percentage = deepfake["fake_probability"] * 100
        css_class = "spoof" if fake_percentage >= 75 else ("warning" if fake_percentage >= 55 else "real")
        st.markdown(
            f"""
            <div class="metric-card {css_class}">
                <div class="small-note">SYNTHETIC-AUDIO ANALYSIS</div>
                <h2>{deepfake["label"]}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.metric("Synthetic/Spoof Model Score", f"{fake_percentage:.1f}%")

    with col3:
        st.markdown(
            f"""
            <div class="metric-card neutral">
                <div class="small-note">ANALYSIS CHUNKS</div>
                <h2>{deepfake["chunks"]}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Long recordings are analyzed in overlapping segments.")

    st.markdown("---")
    st.subheader("🧠 What the result actually means")

    if same_speaker and deepfake["fake_probability"] < 0.25:
        st.success("The recordings were accepted as likely belonging to the same speaker, while the synthetic-audio detector found relatively little evidence of spoofed/synthetic speech.")
    elif same_speaker and deepfake["fake_probability"] >= 0.75:
        st.error("The recordings were accepted as likely belonging to the same speaker, BUT the suspicious recording also received strong synthetic/spoof evidence. Speaker similarity alone is not proof of authenticity.")
    elif not same_speaker and deepfake["fake_probability"] >= 0.75:
        st.warning("The suspicious recording was rejected as the same speaker and also received strong synthetic/spoof evidence.")
    elif not same_speaker:
        st.info("The suspicious recording was not accepted as the same speaker. The synthetic-audio result should be considered independently.")
    else:
        st.warning("The speaker model accepted the pair, but the synthetic-audio detector was inconclusive.")

    st.markdown("---")
    st.subheader("🎧 Audio Diagnostics")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 🎯 Reference")
        st.metric("Duration", f"{reference_stats['duration']:.2f} s")
        st.metric("RMS Energy", f"{reference_stats['rms_mean']:.4f}")
        st.metric("Silence", f"{reference_stats['silence_ratio'] * 100:.1f}%")
        if reference_quality:
            for warning in reference_quality:
                st.warning(warning)
        else:
            st.success("Reference recording quality looks reasonable.")

    with c2:
        st.markdown("### 🚨 Suspicious")
        st.metric("Duration", f"{suspicious_stats['duration']:.2f} s")
        st.metric("RMS Energy", f"{suspicious_stats['rms_mean']:.4f}")
        st.metric("Silence", f"{suspicious_stats['silence_ratio'] * 100:.1f}%")
        if suspicious_quality:
            for warning in suspicious_quality:
                st.warning(warning)
        else:
            st.success("Suspicious recording quality looks reasonable.")

    st.markdown("---")
    st.subheader("🔬 Supporting Acoustic Evidence")
    st.caption("These measurements describe differences between recordings. They are not converted into an arbitrary AI-artifact penalty.")

    if acoustic["differences"]:
        for difference in acoustic["differences"]:
            st.warning(difference)
    else:
        st.success("No large differences were detected by the basic acoustic measurements.")

    st.markdown("---")
    st.subheader("📐 Recording Comparison")

    comparison_data = {
        "Measurement": [
            "Duration", "RMS energy", "Zero-crossing rate",
            "Spectral centroid", "Spectral rolloff", "Silence ratio"
        ],
        "Reference": [
            f"{reference_stats['duration']:.2f} s",
            f"{reference_stats['rms_mean']:.4f}",
            f"{reference_stats['zcr']:.4f}",
            f"{reference_stats['centroid']:.1f} Hz",
            f"{reference_stats['rolloff']:.1f} Hz",
            f"{reference_stats['silence_ratio'] * 100:.1f}%",
        ],
        "Suspicious": [
            f"{suspicious_stats['duration']:.2f} s",
            f"{suspicious_stats['rms_mean']:.4f}",
            f"{suspicious_stats['zcr']:.4f}",
            f"{suspicious_stats['centroid']:.1f} Hz",
            f"{suspicious_stats['rolloff']:.1f} Hz",
            f"{suspicious_stats['silence_ratio'] * 100:.1f}%",
        ],
    }
    st.table(comparison_data)

    st.markdown("---")
    st.subheader("🤖 Synthetic-Audio Model Output")
    st.write(f"**Synthetic/spoof probability:** {deepfake['fake_probability'] * 100:.2f}%")
    st.write(f"**Real/bonafide probability:** {deepfake['real_probability'] * 100:.2f}%")
    st.progress(min(max(deepfake["fake_probability"], 0.0), 1.0))

    st.markdown("---")
    st.subheader("📋 Forensic Summary")

    summary = []
    summary.append("• Speaker verification: likely same speaker." if same_speaker else "• Speaker verification: likely different speakers.")

    if fake_percentage >= 75:
        summary.append("• Synthetic-audio detector: strong spoof/synthetic evidence.")
    elif fake_percentage >= 55:
        summary.append("• Synthetic-audio detector: possible spoof/synthetic evidence.")
    elif fake_percentage <= 25:
        summary.append("• Synthetic-audio detector: relatively low synthetic evidence.")
    else:
        summary.append("• Synthetic-audio detector: inconclusive.")

    summary.append(
        "• Recording-quality limitations were detected."
        if suspicious_quality else
        "• No major recording-quality problems were detected."
    )

    for item in summary:
        st.write(item)

    st.markdown(
        """
        <div class="disclaimer">
        <b>Do not interpret this report as mathematical proof that an audio
        recording is authentic or fake.</b> Speaker verification and
        synthetic-audio detection measure different properties.
        </div>
        """,
        unsafe_allow_html=True,
    )
