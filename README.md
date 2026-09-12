# Realtime Webcam Facial Analysis (OpenFace 3.0 Extension)

Author: Ulubilge Ulusoy <br>
Branch: realtime-gui <br>
Purpose: Add realtime webcam GUI workflows for live preview, Lab Streaming Layer (LSL) streaming, and CSV-based data collection on top of CMU OpenFace 3.0.

## Branch Matrix

| Branch | Compute | OS | LSL | CSV Export | Realtime GUI |
| --- | --- | --- | --- | --- | --- |
| `main` | ![CPU Default](https://img.shields.io/badge/Compute-CPU_Default-1f6feb) | ![Windows](https://img.shields.io/badge/OS-Windows-0078D6) ![Linux](https://img.shields.io/badge/OS-Linux-FCC624) | ![LSL No](https://img.shields.io/badge/LSL-No-6e7781) | ![CSV No](https://img.shields.io/badge/CSV-No-6e7781) | ![GUI No](https://img.shields.io/badge/GUI-No-6e7781) |
| `realtime-gui` | ![CPU Default](https://img.shields.io/badge/Compute-CPU_Default-1f6feb) | ![Windows](https://img.shields.io/badge/OS-Windows-0078D6) | ![LSL Yes](https://img.shields.io/badge/LSL-Yes-2ea44f) | ![CSV Yes](https://img.shields.io/badge/CSV-Yes-2ea44f) | ![GUI Yes](https://img.shields.io/badge/GUI-Yes-2ea44f) |
| `realtime-gui-gpu-version` | ![GPU Preferred](https://img.shields.io/badge/Compute-GPU_Preferred-238636) | ![Windows](https://img.shields.io/badge/OS-Windows-0078D6) | ![LSL Yes](https://img.shields.io/badge/LSL-Yes-2ea44f) | ![CSV No](https://img.shields.io/badge/CSV-No-6e7781) | ![GUI Yes](https://img.shields.io/badge/GUI-Yes-2ea44f) |
| `realtime-gui-linux` | ![CPU Default](https://img.shields.io/badge/Compute-CPU_Default-1f6feb) | ![Linux](https://img.shields.io/badge/OS-Linux-FCC624) | ![LSL Yes](https://img.shields.io/badge/LSL-Yes-2ea44f) | ![CSV No](https://img.shields.io/badge/CSV-No-6e7781) | ![GUI Yes](https://img.shields.io/badge/GUI-Yes-2ea44f) |

## Overview

This repository is a research-oriented fork of [CMU MultiComp Lab's OpenFace 3.0](https://github.com/CMU-MultiComp-Lab/OpenFace-3.0).

This fork adds two realtime webcam workflows:

- [openface_realtime_lsl.py](openface_realtime_lsl.py): live webcam preview with facial analysis and LSL streaming
- [openface_realtime_csv_data_export_only.py](openface_realtime_csv_data_export_only.py): live webcam preview with facial analysis and CSV logging to `logs/`

The forked realtime scripts perform:

- Face detection
- 98-point STAR landmark detection
- Emotion recognition (AffectNet 8 classes)
- Gaze estimation
- Action Unit output
- Live GUI preview with overlayed predictions

## License

This repository remains governed by the original OpenFace 3.0 Software License Agreement for academic or non-profit noncommercial research use only. See [LICENSE](LICENSE) for details.

The realtime webcam, LSL, and CSV logging extensions in this fork should be used only in accordance with the original OpenFace 3.0 license and the licenses of all third-party dependencies and model weights.

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

The realtime scripts expect these files under `./weights/`:

- `Alignment_RetinaFace.pth`
- `Landmark_98.pkl`
- `MTL_backbone.pth`

## Runtime Behavior

This branch is currently CPU-default.

Both realtime scripts are configured to use `cpu` unless you manually change the device setting in the script. For practical use, you may want to switch that to CUDA on a compatible NVIDIA system if you need better realtime performance.

## Realtime Demo

Run the LSL streaming workflow:

```sh
python openface_realtime_lsl.py
```

Run the CSV logging workflow:

```sh
python openface_realtime_csv_data_export_only.py
```

Press `q` in the preview window to stop either workflow.

## Webcam Assumption

Both realtime scripts currently open webcam index `0` by default.

If your system uses a different camera index, you may need to change that value before running the scripts.

## LSL Integration

The [openface_realtime_lsl.py](openface_realtime_lsl.py) workflow has LSL streaming built in through `pylsl`, which is included in [requirements.txt](requirements.txt).

When the script is running, it creates an LSL outlet named `OpenFaceRealtime` and pushes samples suitable for synchronization with other experimental streams such as ECG, EEG, or behavioral event markers.

## LSL Output Contents

The LSL stream includes:

- Emotion probabilities in percent for the 8 AffectNet classes
- Gaze yaw and pitch in radians
- Gaze yaw and pitch in degrees
- Action Unit values

The AU channel count is built dynamically from the model output at runtime.

## CSV Data Collection

The [openface_realtime_csv_data_export_only.py](openface_realtime_csv_data_export_only.py) workflow creates a `logs/` directory if needed and writes timestamped session files there.

Each CSV row contains:

- `frame_idx`
- `timestamp`
- Emotion probabilities in percent
- Gaze yaw and pitch in radians
- Gaze yaw and pitch in degrees
- Action Unit values

The AU column count is also inferred dynamically from the model output.

## Notes

- This fork is focused on realtime experimental data collection rather than replacing the full upstream OpenFace 3.0 documentation.
- The number of Action Unit outputs depends on the model output returned at runtime.
- For core OpenFace 3.0 APIs, CLI usage, and upstream research citation, use the upstream repository documentation.

## AI Assistance Disclosure

This repository was developed with assistance from OpenAI Codex. Codex was used to generate and refine portions of the codebase, documentation, and repository structure.
