import time
import cv2
import mediapipe as mp
import numpy as np
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

def get_pixel_coords(landmark, width, height):
    return np.array([landmark.x * width, landmark.y * height])

mog_threshold = 0.37

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

            landmarks = results.face_landmarks[0]
            #1. extract the coordinates

            cheek_l = get_pixel_coords(landmarks[234], w, h)
            cheek_r = get_pixel_coords(landmarks[454], w, h)
            mouth_l = get_pixel_coords(landmarks[61], w, h)
            mouth_r = get_pixel_coords(landmarks[291], w, h)

            #2. euclidian distance sqrt((x2-x1)^2+(y2-y1)^2)
            cheek_width = np.linalg.norm(cheek_r - cheek_l)
            mouth_width = np.linalg.norm(mouth_r - mouth_l)

            #3. ratio
            mog_ratio = mouth_width / (cheek_width + 1e-6)
            is_mogging = mog_ratio < mog_threshold

            #4. convert to cv2 format with tuple of ints
            pt_cheek_l, pt_cheek_r = tuple(cheek_l.astype(int)), tuple(cheek_r.astype(int))
            pt_mouth_l, pt_mouth_r = tuple(mouth_l.astype(int)), tuple(mouth_r.astype(int))

            # cheekbone baseline cyan
            cv2.line(frame, pt_cheek_l, pt_cheek_r, (255, 255, 0), 2)
            cv2.circle(frame, pt_cheek_l, 4, (255, 255, 0), -1)
            cv2.circle(frame, pt_cheek_r, 4, (255, 255, 0), -1)

            # mouth vector: green if mogging, red if relaxed
            status_color = (0, 255, 0) if is_mogging else (0, 0, 255)
            cv2.line(frame, pt_mouth_l, pt_mouth_r, status_color, 3)
            cv2.circle(frame, pt_mouth_l, 4, status_color, -1)
            cv2.circle(frame, pt_mouth_r, 4, status_color, -1)

            #5. telemetry readout






            #for face_landmarks in results.face_landmarks:
                #for lm in face_landmarks:
                    #cx, cy = int(lm.x * w), int(lm.y * h)
                    #cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)

        cv2.imshow("phase 1 : Face Mesh", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
