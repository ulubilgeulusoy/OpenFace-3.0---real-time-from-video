import math
import os
from typing import List, Optional

import cv2
import numpy as np
import torch
from pylsl import StreamInfo, StreamOutlet, local_clock

from openface.face_detection import FaceDetector
from openface.landmark_detection import LandmarkDetector
from openface.multitask_model import MultitaskPredictor


EMOTION_LABELS = [
    "Neutral",
    "Happy",
    "Sad",
    "Surprise",
    "Fear",
    "Disgust",
    "Anger",
    "Contempt",
]


def build_channel_labels(au_count: int, include_frame_idx: bool = False) -> List[str]:
    labels: List[str] = []
    if include_frame_idx:
        labels.append("frame_idx")

    labels.extend(f"emo_{label}_pct" for label in EMOTION_LABELS)
    labels.extend(
        [
            "gaze_yaw_rad",
            "gaze_pitch_rad",
            "gaze_yaw_deg",
            "gaze_pitch_deg",
        ]
    )
    labels.extend(f"AU_{idx:02d}" for idx in range(au_count))
    return labels


def create_lsl_outlet(
    channel_labels: List[str],
    stream_name: str,
    source_id: str,
    nominal_srate: float = 0.0,  # irregular stream so viewers don’t assume 30 Hz
) -> StreamOutlet:
    info = StreamInfo(
        name=stream_name,
        type="Face",
        channel_count=len(channel_labels),
        nominal_srate=nominal_srate,
        channel_format="float32",
        source_id=source_id,
    )

    channels = info.desc().append_child("channels")
    for label in channel_labels:
        ch = channels.append_child("channel")
        ch.append_child_value("label", label)
        ch.append_child_value("type", "feature")

    outlet = StreamOutlet(info)
    print(
        f"[lsl] Outlet '{stream_name}' ready with {len(channel_labels)} channels "
        f"(source_id={source_id}, nominal_srate={nominal_srate})."
    )
    return outlet


def draw_overlay_panel(frame, emo_probs, yaw_deg, pitch_deg, au_values):
    """
    Original-style overlay panel: Emotion label + per-emotion % + gaze + first 10 AUs.
    """
    panel_x = 10
    panel_y = 40
    line_h = 20
    width = 250

    overlay = frame.copy()
    panel_height = 30 + (1 + len(EMOTION_LABELS) + 2 + min(len(au_values), 10)) * line_h
    cv2.rectangle(
        overlay,
        (panel_x - 8, panel_y - 28),
        (panel_x - 8 + width, panel_y - 28 + panel_height),
        (0, 0, 0),
        -1,
    )
    frame[:] = cv2.addWeighted(overlay, 0.40, frame, 0.60, 0)

    # Emotion label
    emo_idx = int(np.nanargmax(emo_probs)) if np.any(np.isfinite(emo_probs)) else 0
    emo_label = EMOTION_LABELS[emo_idx] if 0 <= emo_idx < len(EMOTION_LABELS) else "Unknown"

    cv2.putText(
        frame,
        f"Emotion: {emo_label}",
        (panel_x, panel_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    panel_y += line_h + 5

    # Per-emotion probabilities
    for label, prob in zip(EMOTION_LABELS, emo_probs):
        if np.isnan(prob):
            txt = f"{label:8s}:   NaN"
        else:
            txt = f"{label:8s}: {prob * 100:5.1f}%"
        cv2.putText(
            frame,
            txt,
            (panel_x, panel_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        panel_y += line_h

    panel_y += 5
    cv2.putText(
        frame,
        f"Gaze yaw: {yaw_deg:+.1f} deg" if np.isfinite(yaw_deg) else "Gaze yaw: NaN",
        (panel_x, panel_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 220, 220),
        1,
        cv2.LINE_AA,
    )
    panel_y += line_h
    cv2.putText(
        frame,
        f"Gaze pitch: {pitch_deg:+.1f} deg" if np.isfinite(pitch_deg) else "Gaze pitch: NaN",
        (panel_x, panel_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 220, 220),
        1,
        cv2.LINE_AA,
    )
    panel_y += line_h + 5

    # First 10 AUs
    for idx, value in enumerate(au_values[:10]):
        if np.isnan(value):
            txt = f"AU_{idx:02d}:   NaN"
        else:
            txt = f"AU_{idx:02d}: {value:+.2f}"
        cv2.putText(
            frame,
            txt,
            (panel_x, panel_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (180, 180, 255),
            1,
            cv2.LINE_AA,
        )
        panel_y += line_h


def main():
    base_dir = os.path.dirname(__file__)
    weights_dir = os.path.join(base_dir, "weights")
    face_model_path = os.path.join(weights_dir, "Alignment_RetinaFace.pth")
    landmark_model_path = os.path.join(weights_dir, "Landmark_98.pkl")
    multitask_model_path = os.path.join(weights_dir, "MTL_backbone.pth")

    device = "cpu"  # change to "cuda" if available

    stream_name = "OpenFaceRealtime"
    source_id = "openface_realtime_gui_nofacefound"

    # If FaceDetector only accepts a filepath, we keep this. (It slows FPS but works.)
    tmp_path = "._of_tmp.jpg"

    include_frame_idx = False  # keep False unless you explicitly want it

    print("Initializing OpenFace models...")
    face_detector = FaceDetector(model_path=face_model_path, device=device)
    landmark_detector = LandmarkDetector(model_path=landmark_model_path, device=device)
    multitask_model = MultitaskPredictor(model_path=multitask_model_path, device=device)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Streaming predictions to LSL. Press 'q' to stop the preview window.")

    frame_idx = 0
    outlet: Optional[StreamOutlet] = None
    channel_labels: Optional[List[str]] = None
    au_count: Optional[int] = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read from webcam.")
                break

            # Defaults (for when we push NaNs)
            emo_probs = np.full((len(EMOTION_LABELS),), np.nan, dtype=float)
            yaw = pitch = yaw_deg = pitch_deg = float("nan")
            au_values: Optional[List[float]] = None

            # Face box defaults (for drawing only)
            x1 = y1 = x2 = y2 = 0
            dets = None
            cropped_face = None

            # --- Face detect ---
            cv2.imwrite(tmp_path, frame)
            try:
                cropped_face, dets = face_detector.get_face(tmp_path)
            except Exception as exc:
                print(f"Face detection error: {exc}")
                cropped_face, dets = None, None

            # --- If face detected, run inference ---
            if cropped_face is not None and dets is not None and len(dets) > 0:
                dets_np = np.array(dets)
                best_idx = int(np.argmax(dets_np[:, 4]))
                x1, y1, x2, y2, _ = dets_np[best_idx][:5]
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                try:
                    with torch.no_grad():
                        emotion_logits, gaze_output, au_output = multitask_model.predict(cropped_face)

                    emo_probs = torch.softmax(emotion_logits, dim=1)[0].cpu().numpy()

                    yaw = float(gaze_output[0, 0].item())
                    pitch = float(gaze_output[0, 1].item())
                    yaw_deg = yaw * 180.0 / math.pi
                    pitch_deg = pitch * 180.0 / math.pi

                    au_vec = au_output[0] if au_output.ndim == 2 else au_output
                    au_values = au_vec.cpu().numpy().astype(float).tolist()

                except Exception as exc:
                    print(f"Multitask model error: {exc}")
                    # keep NaNs

            # --- Create outlet after first successful inference (so AU count is known) ---
            if outlet is None and au_values is not None:
                au_count = len(au_values)
                channel_labels = build_channel_labels(au_count, include_frame_idx=include_frame_idx)
                outlet = create_lsl_outlet(channel_labels, stream_name, source_id, nominal_srate=0.0)

            # --- Push sample EVERY loop (critical for full-duration recordings) ---
            if outlet is not None and channel_labels is not None:
                if au_values is None:
                    # Fill AU vector with NaNs of known length
                    if au_count is None:
                        # Outlet exists only after au_count is known, so this shouldn’t happen
                        au_count = sum(1 for lbl in channel_labels if lbl.startswith("AU_"))
                    au_values = [float("nan")] * au_count

                sample = []
                if include_frame_idx:
                    sample.append(float(frame_idx))
                sample.extend([float(p) * 100.0 for p in emo_probs])  # emotion % values
                sample.extend([yaw, pitch, yaw_deg, pitch_deg])       # gaze
                sample.extend(au_values)                               # AUs

                outlet.push_sample(sample, timestamp=local_clock())

            # --- GUI drawing (like original) ---
            # Face box and landmarks if we have dets
            if dets is not None and len(dets) > 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                try:
                    # Best-effort landmark overlay (depends on your LandmarkDetector API)
                    landmarks = landmark_detector.detect_landmarks(frame, dets)
                    dets_np = np.array(dets)
                    best_idx = int(np.argmax(dets_np[:, 4]))
                    if landmarks and len(landmarks) > best_idx:
                        for (lx, ly) in landmarks[best_idx]:
                            cv2.circle(frame, (int(lx), int(ly)), 1, (255, 255, 0), -1)
                except Exception:
                    pass

            # Panel overlay (always shown; uses NaNs when not available)
            if au_values is None:
                # If we haven’t created outlet yet, we still want a stable panel length
                fallback_aus = 8 if au_count is None else au_count
                au_values_for_panel = [float("nan")] * fallback_aus
            else:
                au_values_for_panel = au_values

            draw_overlay_panel(frame, emo_probs, yaw_deg, pitch_deg, au_values_for_panel)

            cv2.imshow("OpenFace 3.0 Realtime LSL (q to quit)", frame)

            frame_idx += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


if __name__ == "__main__":
    main()
