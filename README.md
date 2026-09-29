# 🔍 VoicePrint Forensics

Reference-based speaker verification and synthetic-audio analysis built with Streamlit.

## Features

- ECAPA-TDNN speaker verification
- WavLM-based synthetic/spoof audio analysis
- Acoustic diagnostics
- Audio quality checks
- Separate speaker similarity and synthetic-evidence results

## Important

This is an experimental forensic analysis tool. A high speaker similarity does **not** prove that a recording is authentic, and synthetic-audio detectors can produce false positives and false negatives.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

This repository is structured for Streamlit Community Cloud.

## Recommended audio

- 4+ seconds of clear speech
- Minimal background noise/music
- WAV preferred when available
- Avoid extremely compressed recordings

## Models

- `speechbrain/spkrec-ecapa-voxceleb`
- `0xmola/wavlm-deepfake-audio-forensics`

For educational and research use.
