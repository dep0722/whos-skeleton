import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import joblib

# === 模�??路�?? ===
MODEL_PATH = r"C:/mydata/sf/conda/1025_test/test_model/RFmodel_1104_normal.pkl"
rf_model = joblib.load(MODEL_PATH)

# === Mediapipe Pose ???�???? ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,  # ??��??追蹤
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# === ???�?編�?? ===
LM = {
    'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12,
    'LEFT_ELBOW': 13, 'RIGHT_ELBOW': 14,
    'LEFT_WRIST': 15, 'RIGHT_WRIST': 16,
    'LEFT_HIP': 23, 'RIGHT_HIP': 24,
    'LEFT_KNEE': 25, 'RIGHT_KNEE': 26,
    'LEFT_ANKLE': 27, 'RIGHT_ANKLE': 28,
    'LEFT_HEEL': 29, 'RIGHT_HEEL': 30
}

EDGES = [
    ('LEFT_SHOULDER', 'LEFT_ELBOW'),
    ('LEFT_ELBOW', 'LEFT_WRIST'),
    ('RIGHT_SHOULDER', 'RIGHT_ELBOW'),
    ('RIGHT_ELBOW', 'RIGHT_WRIST'),
    ('LEFT_SHOULDER', 'RIGHT_SHOULDER'),
    ('LEFT_HIP', 'RIGHT_HIP'),
    ('LEFT_SHOULDER', 'LEFT_HIP'),
    ('RIGHT_SHOULDER', 'RIGHT_HIP'),
    ('LEFT_HIP', 'LEFT_KNEE'),
    ('LEFT_KNEE', 'LEFT_ANKLE'),
    ('RIGHT_HIP', 'RIGHT_KNEE'),
    ('RIGHT_KNEE', 'RIGHT_ANKLE')
]

# === �?�???�徵 ===
def extract_features(image):
    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if not results.pose_landmarks:
        return None

    landmarks = results.pose_landmarks.landmark

    hip_width = np.sqrt((landmarks[LM['LEFT_HIP']].x*w - landmarks[LM['RIGHT_HIP']].x*w)**2 +
                        (landmarks[LM['LEFT_HIP']].y*h - landmarks[LM['RIGHT_HIP']].y*h)**2)

    features = {}
    for (p1, p2) in EDGES:
        x1, y1 = landmarks[LM[p1]].x*w, landmarks[LM[p1]].y*h
        x2, y2 = landmarks[LM[p2]].x*w, landmarks[LM[p2]].y*h
        dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        features[f"{p1}_{p2}"] = dist / hip_width

    return features

# === ?????????影�?? 1 ===
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise RuntimeError("??��???????????影�?? 1")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("???�? �????影�??失�??")
            break

        features = extract_features(frame)
        if features is not None:
            df_feat = pd.DataFrame([features])
            pred_label = rf_model.predict(df_feat)[0]
            # ??��?��?��??顯示???測�?????
            cv2.putText(frame, f"Predicted: {pred_label}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "No Pose Detected", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2, cv2.LINE_AA)
        #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        cv2.imshow("Pose Classification", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC ??��??
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
