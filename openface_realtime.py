import os
import cv2
import numpy as np
import torch
import math
import csv                    # <-- ADD THIS
from datetime import datetime # <-- already added

from openface.face_detection import FaceDetector
from openface.landmark_detection import LandmarkDetector
from openface.multitask_model import MultitaskPredictor




def main():
    # ====== CONFIG ======
    BASE_DIR = os.path.dirname(__file__)
    WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
    FACE_MODEL_PATH = os.path.join(WEIGHTS_DIR, "Alignment_RetinaFace.pth")
    LANDMARK_MODEL_PATH = os.path.join(WEIGHTS_DIR, "Landmark_98.pkl")
    MULTITASK_MODEL_PATH = os.path.join(WEIGHTS_DIR, "MTL_backbone.pth")

    DEVICE = "cpu"  # change to "cuda" later if you want GPU

    # Emotion labels from the README (AffectNet 8 classes)
    EMOTION_LABELS = [
        "Neutral", "Happy", "Sad", "Surprise",
        "Fear", "Disgust", "Anger", "Contempt"
    ]

    print("Initializing models...")
    face_detector = FaceDetector(model_path=FACE_MODEL_PATH, device=DEVICE)
    landmark_detector = LandmarkDetector(model_path=LANDMARK_MODEL_PATH, device=DEVICE)
    multitask_model = MultitaskPredictor(model_path=MULTITASK_MODEL_PATH, device=DEVICE)

    cap = cv2.VideoCapture(0)  # 0 = default webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")

    TMP_PATH = "._of_tmp.jpg"

    # ====== CSV LOGGING SETUP ======
    logs_dir = os.path.join(BASE_DIR, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    session_filename = datetime.now().strftime("session_%Y%m%d_%H%M%S.csv")
    csv_path = os.path.join(logs_dir, session_filename)
    print(f"Logging to: {csv_path}")

    # Build header: frame index, timestamp, emotions, gaze, AUs
    header = ["frame_idx", "timestamp"]
    # Emotion percentages
    for label in EMOTION_LABELS:
        header.append(f"emo_{label}_pct")
    # Gaze
    header.extend([
        "gaze_yaw_rad", "gaze_pitch_rad",
        "gaze_yaw_deg", "gaze_pitch_deg"
    ])
    # AU values (we do not know how many in advance, so we’ll name them AU_00..AU_?? dynamically later)
    # We will infer number from first frame that has a face and write AU columns once.

    frame_idx = 0
    au_header_written = False

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)

        # We will write the header after we know how many AU dimensions exist.
        # For now, store base header; expand once we see au_values length.

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read from webcam.")
                break

            # Optional: resize for speed
            # frame = cv2.resize(frame, (640, 480))

            # ================================
            # 1. FACE DETECTION (temp file)
            # ================================
            cv2.imwrite(TMP_PATH, frame)

            try:
                cropped_face, dets = face_detector.get_face(TMP_PATH)
            except Exception as e:
                print(f"Face detection error: {e}")
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
                cv2.imshow("OpenFace 3.0 Realtime (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                frame_idx += 1
                continue

            dets_np = np.array(dets)
            best_idx = np.argmax(dets_np[:, 4])
            x1, y1, x2, y2, conf = dets_np[best_idx][:5]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # ================================
            # 2. MULTITASK PREDICTIONS
            # ================================
            try:
                with torch.no_grad():
                    emotion_logits, gaze_output, au_output = multitask_model.predict(cropped_face)
            except Exception as e:
                print(f"Multitask model error: {e}")
                cv2.imshow("OpenFace 3.0 Realtime (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                frame_idx += 1
                continue

            # Emotion: probabilities for all 8 classes
            emo_probs = torch.softmax(emotion_logits, dim=1)[0].cpu().numpy()
            emo_idx = int(np.argmax(emo_probs))
            emo_label = EMOTION_LABELS[emo_idx] if 0 <= emo_idx < len(EMOTION_LABELS) else f"Class {emo_idx}"

            # Gaze: yaw, pitch (radians + degrees)
            yaw = float(gaze_output[0, 0].item())
            pitch = float(gaze_output[0, 1].item())
            yaw_deg = yaw * 180.0 / math.pi
            pitch_deg = pitch * 180.0 / math.pi

            # AU vector: show all outputs as AU_0, AU_1, ...
            if au_output.ndim == 2:
                au_vec = au_output[0]
            else:
                au_vec = au_output
            au_values = au_vec.cpu().numpy().tolist()

            # ================================
            # 2b. CSV LOGGING (per-frame)
            # ================================
            # 2b. CSV LOGGING (per-frame)
            timestamp = datetime.now().isoformat()
            
            # If we haven't written AU headers yet, do it now once we know len(au_values)
            if not au_header_written:
                au_cols = [f"AU_{i:02d}" for i in range(len(au_values))]
                full_header = header + au_cols
                writer.writerow(full_header)
                au_header_written = True
            
            # ---- Round values to 3 decimal places ----
            # Emotion percentages (0–100)
            emo_probs_pct = [round(float(p) * 100.0, 3) for p in emo_probs]
            
            # Gaze (rad and deg)
            yaw_r      = round(yaw, 3)
            pitch_r    = round(pitch, 3)
            yaw_deg_r  = round(yaw_deg, 3)
            pitch_deg_r = round(pitch_deg, 3)
            
            # AU values
            au_values_rounded = [round(float(v), 3) for v in au_values]
            
            # Build row
            row = [frame_idx, timestamp]
            row.extend(emo_probs_pct)
            row.extend([yaw_r, pitch_r, yaw_deg_r, pitch_deg_r])
            row.extend(au_values_rounded)
            
            writer.writerow(row)


            # ================================
            # 3. VISUALIZATION
            # ================================

            # 3a. Draw bounding box on the face
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Optional: draw some landmarks (using detections)
            try:
                landmarks = landmark_detector.detect_landmarks(frame, dets)
                if landmarks and len(landmarks) > 0:
                    for (lx, ly) in landmarks[best_idx]:
                        cv2.circle(frame, (int(lx), int(ly)), 1, (255, 255, 0), -1)
            except Exception:
                # Landmark failures shouldn't kill the loop
                pass

            # 3b. Side info panel (emotion, gaze, all AUs)
            panel_x = 10
            panel_y = 20
            line_h = 20

            # Draw a semi-transparent rectangle as background for text
            overlay = frame.copy()
            panel_width = 320
            # height enough for header + 8 emotions + gaze + ~min(len(AUs), 10)
            panel_height = 40 + (len(EMOTION_LABELS) + 2 + min(len(au_values), 10)) * line_h
            cv2.rectangle(
                overlay,
                (panel_x - 5, panel_y - 20),
                (panel_x - 5 + panel_width, panel_y - 20 + panel_height),
                (0, 0, 0),
                -1,
            )
            alpha = 0.4
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            # Header: main emotion
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

            # Emotion distribution
            cv2.putText(
                frame,
                "Emotion probs:",
                (panel_x, panel_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            panel_y += line_h

            for label, p in zip(EMOTION_LABELS, emo_probs):
                cv2.putText(
                    frame,
                    f"{label:8s}: {p*100:5.1f}%",
                    (panel_x, panel_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )
                panel_y += line_h

            # Gaze info
            panel_y += 5
            cv2.putText(
                frame,
                f"Gaze yaw: {yaw:+.2f} rad ({yaw_deg:+.1f} deg)",
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
                f"Gaze pitch: {pitch:+.2f} rad ({pitch_deg:+.1f} deg)",
                (panel_x, panel_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )
            panel_y += line_h

            # AU info (show all, but cap visible lines to avoid crazy tall panels)
            panel_y += 5
            cv2.putText(
                frame,
                "Action Units (raw outputs):",
                (panel_x, panel_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 255),
                1,
                cv2.LINE_AA,
            )
            panel_y += line_h

            max_aus_to_show = min(len(au_values), 10)  # show first 10 by default
            for idx in range(max_aus_to_show):
                v = au_values[idx]
                cv2.putText(
                    frame,
                    f"AU_{idx:02d}: {v:+.2f}",
                    (panel_x, panel_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (180, 180, 255),
                    1,
                    cv2.LINE_AA,
                )
                panel_y += line_h

            cv2.imshow("OpenFace 3.0 Realtime (q to quit)", frame)

            frame_idx += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    # Clean up temp file
    if os.path.exists(TMP_PATH):
        try:
            os.remove(TMP_PATH)
        except Exception:
            pass


if __name__ == "__main__":
    main()
