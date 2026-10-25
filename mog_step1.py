import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. options
base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

# 2. landmarker webcam
with vision.FaceLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # 3. convert to mp image obj
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 4. Run video
        timestamp_ms = int(time.monotonic() * 1000)
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        #5. Draw all 478 landmarks
        if results.face_landmarks:
            for face_landmarks in results.face_landmarks:
                for lm in face_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)

        cv2.imshow("phase 1 : Face Mesh", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
