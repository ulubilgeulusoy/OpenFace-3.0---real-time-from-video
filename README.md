# Realtime Webcam Facial Analysis (OpenFace 3.0 Extension)

Author: Ulubilge Ulusoy <br>
Branch: realtime-gui <br>
Purpose: Add a real-time webcam GUI and Lab Streaming Layer (LSL) output workflow on top of CMU OpenFace 3.0.

## Overview

This repository is a research-oriented fork of [CMU MultiComp Lab's OpenFace 3.0](https://github.com/CMU-MultiComp-Lab/OpenFace-3.0).

This fork adds a real-time webcam pipeline through [openface_realtime_lsl.py](openface_realtime_lsl.py) for:

- Face detection
- 98-point STAR landmark detection
- Emotion recognition (AffectNet 8 classes)
- Gaze estimation
- Action Unit output
- Live GUI preview with overlayed predictions
- LSL streaming for synchronized multimodal experiments

## License

This repository remains governed by the original OpenFace 3.0 Software License Agreement for academic or non-profit noncommercial research use only. See [LICENSE](LICENSE) for details.

The real-time webcam and LSL extensions in this fork should be used only in accordance with the original OpenFace 3.0 license and the licenses of all third-party dependencies and model weights.

## Upstream OpenFace 3.0

This README documents the additions specific to this fork.

For the original OpenFace 3.0 installation guidance, core package usage, CLI examples, citation, and upstream project details, see the original repository:

https://github.com/CMU-MultiComp-Lab/OpenFace-3.0

## Fork-Specific Installation

1. Create and activate your Python environment.
2. Install the dependencies:

```sh
pip install -r requirements.txt
pip install openface-test
```

3. Download the required model weights and place them in the local `weights/` folder.

Weight sources:

- Google Drive: https://drive.google.com/drive/folders/1aBEol-zG_blHSavKFVBH9dzc9U9eJ92p
- Hugging Face: https://huggingface.co/nutPace/openface_weights

## Required Weights For This Fork

The realtime script expects these files under `./weights/`:

- `Alignment_RetinaFace.pth`
- `Landmark_98.pkl`
- `MTL_backbone.pth`

These paths are used directly in [openface_realtime_lsl.py](openface_realtime_lsl.py).

## GPU Behavior

This branch is GPU-preferred.

When CUDA is available, the realtime script automatically selects `cuda:0`. If CUDA is not available, it falls back to CPU. That runtime device selection happens in [openface_realtime_lsl.py](openface_realtime_lsl.py).

For practical use, this fork is intended to run with an NVIDIA GPU when available, especially for realtime performance.

## Realtime Demo

From your activated environment, run:

```sh
python openface_realtime_lsl.py
```

What the script does:

- Opens a webcam feed
- Detects the most prominent face in each frame
- Runs landmark, emotion, gaze, and AU inference
- Displays a live preview window
- Publishes prediction data over LSL

Press `q` in the preview window to stop the session.

## Webcam Assumption

The current script opens webcam index `1` by default in [openface_realtime_lsl.py](openface_realtime_lsl.py).

If your system uses a different camera index, you may need to change that value before running the script.

## Lab Streaming Layer (LSL) Integration

The realtime script has LSL streaming built in through `pylsl`, which is included in [requirements.txt](requirements.txt).

When the script is running, it creates an LSL outlet named `OpenFaceRealtime` and pushes samples suitable for synchronization with other experimental streams such as ECG, EEG, or behavioral event markers.

## LSL Output Contents

The stream includes:

- Emotion probabilities in percent for the 8 AffectNet classes
- Gaze yaw and pitch in radians
- Gaze yaw and pitch in degrees
- Action Unit values

The channel labels are built dynamically in [openface_realtime_lsl.py](openface_realtime_lsl.py), so the number of AU channels follows the model output used at runtime.

## Notes

- This fork is focused on realtime experimental data collection rather than replacing the full upstream OpenFace 3.0 documentation.
- If no face is detected in a frame, the realtime loop continues and LSL output behavior is handled by the script logic.
- For core OpenFace 3.0 APIs, CLI usage, and upstream research citation, use the upstream repository documentation.
