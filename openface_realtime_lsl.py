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


def build_channel_labels(au_count: int) -> List[str]:
    labels: List[str] = ["frame_idx"]
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
    nominal_srate: float = 30.0,
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
        f"[lsl] Outlet '{stream_name}' ready with {len(channel_labels)} channels (source_id={source_id})."
    )
    return outlet


def main():
    base_dir = os.path.dirname(__file__)
    weights_dir = os.path.join(base_dir, "weights")
    face_model_path = os.path.join(weights_dir, "Alignment_RetinaFace.pth")
    landmark_model_path = os.path.join(weights_dir, "Landmark_98.pkl")
    multitask_model_path = os.path.join(weights_dir, "MTL_backbone.pth")

    device = "cpu"  # change to "cuda" if GPU is configured

    stream_name = "OpenFaceRealtime"
    source_id = "openface_realtime_direct"
    tmp_path = "._of_tmp.jpg"

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

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read from webcam.")
                break

            cv2.imwrite(tmp_path, frame)
            try:
                cropped_face, dets = face_detector.get_face(tmp_path)
            except Exception as exc:  # pragma: no cover - realtime diagnostics
                print(f"Face detection error: {exc}")
                cropped_face, dets = None, None

            if cropped_face is None or dets is None or len(dets) == 0:
                cv2.putText(
                    frame,
                    "No face detected",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("OpenFace 3.0 Realtime LSL (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                frame_idx += 1
                continue

            dets_np = np.array(dets)
            best_idx = int(np.argmax(dets_np[:, 4]))
            x1, y1, x2, y2, _ = dets_np[best_idx][:5]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            try:
                with torch.no_grad():
                    emotion_logits, gaze_output, au_output = multitask_model.predict(cropped_face)
            except Exception as exc:  # pragma: no cover
                print(f"Multitask model error: {exc}")
                frame_idx += 1
                continue

            emo_probs = torch.softmax(emotion_logits, dim=1)[0].cpu().numpy()
            yaw = float(gaze_output[0, 0].item())
            pitch = float(gaze_output[0, 1].item())
            yaw_deg = yaw * 180.0 / math.pi
            pitch_deg = pitch * 180.0 / math.pi

            au_vec = au_output[0] if au_output.ndim == 2 else au_output
            au_values = au_vec.cpu().numpy().astype(float).tolist()

            if outlet is None:
                channel_labels = build_channel_labels(len(au_values))
                outlet = create_lsl_outlet(channel_labels, stream_name, source_id)

            sample = [
                float(frame_idx),
                *[float(p) * 100.0 for p in emo_probs],
                yaw,
                pitch,
                yaw_deg,
                pitch_deg,
                *au_values,
            ]
            outlet.push_sample(sample, timestamp=local_clock())

            # Visualization overlay mirrors original script so the user can monitor quality.
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            try:
                landmarks = landmark_detector.detect_landmarks(frame, dets)
                if landmarks and len(landmarks) > best_idx:
                    for (lx, ly) in landmarks[best_idx]:
                        cv2.circle(frame, (int(lx), int(ly)), 1, (255, 255, 0), -1)
            except Exception:
                pass

            panel_x = 10
            panel_y = 20
            line_h = 20
            overlay = frame.copy()
            panel_height = 60 + (len(EMOTION_LABELS) + 2 + min(len(au_values), 10)) * line_h
            cv2.rectangle(
                overlay,
                (panel_x - 5, panel_y - 20),
                (panel_x - 5 + 220, panel_y - 20 + panel_height),
                (0, 0, 0),
                -1,
            )
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

            emo_idx = int(np.argmax(emo_probs))
            emo_label = EMOTION_LABELS[emo_idx] if 0 <= emo_idx < len(EMOTION_LABELS) else f"Class {emo_idx}"
            cv2.putText(
                frame,
                f"Emotion: {emo_label}",
                (panel_x, panel_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            panel_y += line_h + 5

            for label, prob in zip(EMOTION_LABELS, emo_probs):
                cv2.putText(
                    frame,
                    f"{label:8s}: {prob * 100:5.1f}%",
                    (panel_x, panel_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )
                panel_y += line_h

            panel_y += 5
            cv2.putText(
                frame,
                f"Gaze yaw: {yaw_deg:+.1f} deg",
                (panel_x, panel_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )
            panel_y += line_h
            cv2.putText(
                frame,
                f"Gaze pitch: {pitch_deg:+.1f} deg",
                (panel_x, panel_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )
            panel_y += line_h + 5

            for idx, value in enumerate(au_values[:10]):
                cv2.putText(
                    frame,
                    f"AU_{idx:02d}: {value:+.2f}",
                    (panel_x, panel_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (180, 180, 255),
                    1,
                    cv2.LINE_AA,
                )
                panel_y += line_h

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
