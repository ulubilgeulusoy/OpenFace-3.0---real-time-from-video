import os
import cv2
import numpy as np
import torch

from openface.face_detection import FaceDetector
from openface.landmark_detection import LandmarkDetector
from openface.multitask_model import MultitaskPredictor


def main():
    # ====== CONFIG ======
    WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
    FACE_MODEL_PATH = os.path.join(WEIGHTS_DIR, "Alignment_RetinaFace.pth")
    LANDMARK_MODEL_PATH = os.path.join(WEIGHTS_DIR, "Landmark_98.pkl")
    MULTITASK_MODEL_PATH = os.path.join(WEIGHTS_DIR, "MTL_backbone.pth")

    DEVICE = "cpu"  # change to "cuda" later if you want GPU

    # Emotion labels from the README (AffectNet 8 classes)
    EMOTION_LABELS = [
        "Neutral", "Happy", "Sad", "Surprise",
        "Fear", "Disgust", "Anger", "Contempt"
    ]

    # ====== INIT MODELS ======
    print("Initializing models...")
    face_detector = FaceDetector(model_path=FACE_MODEL_PATH, device=DEVICE)
    landmark_detector = LandmarkDetector(model_path=LANDMARK_MODEL_PATH, device=DEVICE)
    multitask_model = MultitaskPredictor(model_path=MULTITASK_MODEL_PATH, device=DEVICE)

    # ====== INIT CAMERA ======
    cap = cv2.VideoCapture(0)  # 0 = default webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")

    TMP_PATH = "._of_tmp.jpg"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read from webcam.")
            break

        # Optional: resize for speed
        # frame = cv2.resize(frame, (640, 480))

        # =======================================
        # 1. FACE DETECTION VIA TEMPORARY FILE
        # =======================================
        # OpenFace's FaceDetector expects a filename, not a numpy array.
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
            continue

        dets_np = np.array(dets)
        best_idx = np.argmax(dets_np[:, 4])
        x1, y1, x2, y2, conf = dets_np[best_idx][:5]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # =======================================
        # 2. MULTITASK PREDICTIONS ON CROPPED FACE
        # =======================================
        try:
            with torch.no_grad():
                emotion_logits, gaze_output, au_output = multitask_model.predict(cropped_face)
        except Exception as e:
            print(f"Multitask model error: {e}")
            cv2.imshow("OpenFace 3.0 Realtime (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # Emotion: argmax
        emo_idx = int(torch.argmax(emotion_logits, dim=1).item())
        emo_label = EMOTION_LABELS[emo_idx] if 0 <= emo_idx < len(EMOTION_LABELS) else f"Class {emo_idx}"

        # Gaze: yaw, pitch
        yaw = float(gaze_output[0, 0].item())
        pitch = float(gaze_output[0, 1].item())

        # AU vector – one example (approx AU12 ~ smile index 6)
        if au_output.ndim == 2:
            au_vec = au_output[0]
        else:
            au_vec = au_output
        au_values = au_vec.cpu().numpy().tolist()
        au_smile = au_values[6] if len(au_values) > 6 else 0.0

        # =======================================
        # 3. DRAW OVERLAYS
        # =======================================

        # Face bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Emotion label
        cv2.putText(
            frame,
            emo_label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Gaze text
        gaze_text = f"yaw={yaw:+.2f}, pitch={pitch:+.2f}"
        cv2.putText(
            frame,
            gaze_text,
            (x1, y2 + 20 if y2 + 20 < frame.shape[0] - 10 else y2 - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # AU text
        au_text = f"AU12~smile: {au_smile:.2f}"
        cv2.putText(
            frame,
            au_text,
            (x1, y2 + 40 if y2 + 40 < frame.shape[0] - 10 else y2 - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("OpenFace 3.0 Realtime (q to quit)", frame)

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
