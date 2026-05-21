# Realtime Webcam Facial Analysis (OpenFace 3.0 Extension)

Author: Ulubilge Ulusoy <br>
Branch: realtime-gui-linux <br>
Purpose: Add a Linux-oriented realtime webcam GUI workflow with Lab Streaming Layer (LSL) output on top of CMU OpenFace 3.0.

## Overview

This repository is a research-oriented fork of [CMU MultiComp Lab's OpenFace 3.0](https://github.com/CMU-MultiComp-Lab/OpenFace-3.0).

This Linux branch adds a realtime webcam workflow through [openface_realtime_lsl.py](openface_realtime_lsl.py) for:

- Face detection
- 98-point STAR landmark detection
- Emotion recognition (AffectNet 8 classes)
- Gaze estimation
- Action Unit output
- Live GUI preview with overlayed predictions
- LSL streaming for synchronized multimodal experiments

## License

This repository remains governed by the original OpenFace 3.0 Software License Agreement for academic or non-profit noncommercial research use only. See [LICENSE](LICENSE) for details.

The realtime webcam and LSL extensions in this fork should be used only in accordance with the original OpenFace 3.0 license and the licenses of all third-party dependencies and model weights.

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

## Runtime Behavior

This branch is currently CPU-default.

The realtime script is configured to use `cpu` unless you manually change the device setting in the script. For practical use, you may want to switch that to CUDA on a compatible NVIDIA system if you need better realtime performance.

## Linux Note For STAR Landmark Loading

On some Ubuntu or Linux setups, `python3 openface_realtime_lsl.py` may fail during landmark model initialization with:

```text
PermissionError: [Errno 13] Permission denied: '/work'
```

This happens because the installed `openface-test` package may contain a hardcoded STAR runtime path from the original developer machine:

```python
self.ckpt_dir = '/work/jiewenh/openFace/OpenFace-3.0/STAR'
```

If you hit this issue, update the file inside your local virtual environment:

```text
<your-venv>/lib/python3.12/site-packages/openface/STAR/conf/alignment.py
```

Replace:

```python
import os.path as osp
```

with:

```python
import os
import os.path as osp
```

Then replace:

```python
self.ckpt_dir = '/work/jiewenh/openFace/OpenFace-3.0/STAR'
```

with:

```python
default_ckpt_dir = osp.abspath(
    osp.join(osp.dirname(__file__), "..", "runtime")
)
self.ckpt_dir = os.environ.get("OPENFACE_STAR_CKPT_DIR", default_ckpt_dir)
```

This keeps STAR runtime artifacts in a writable local directory instead of `/work/...`.

## Realtime Demo

Run the LSL streaming workflow:

```sh
python3 openface_realtime_lsl.py
```

What the script does:

- Opens a webcam feed
- Detects the most prominent face in each frame
- Runs landmark, emotion, gaze, and AU inference
- Displays a live preview window
- Publishes prediction data over LSL

Press `q` in the preview window to stop the session.

## Webcam Assumption

If your system uses a different camera index than the one configured in the script, you may need to change that value before running the workflow.

## LSL Integration

The [openface_realtime_lsl.py](openface_realtime_lsl.py) workflow has LSL streaming built in through `pylsl`, which is included in [requirements.txt](requirements.txt).

When the script is running, it creates an LSL outlet named `OpenFaceRealtime` and pushes samples suitable for synchronization with other experimental streams such as ECG, EEG, or behavioral event markers.

## LSL Output Contents

The stream includes:

- Emotion probabilities in percent for the 8 AffectNet classes
- Gaze yaw and pitch in radians
- Gaze yaw and pitch in degrees
- Action Unit values

The AU channel count is built dynamically from the model output at runtime.

## Notes

- This Linux branch is focused on realtime experimental data collection rather than replacing the full upstream OpenFace 3.0 documentation.
- The number of Action Unit outputs depends on the model output returned at runtime.
- For core OpenFace 3.0 APIs, CLI usage, and upstream research citation, use the upstream repository documentation.
