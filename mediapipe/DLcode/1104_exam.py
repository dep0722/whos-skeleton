import cv2
import mediapipe as mp
import numpy as np
import joblib
import os

# ===== 路徑 =====
MODEL_PATH = r"C:/mydata/sf/conda/1025_test/DL/DLmodemlp_model.pkl"   # ← 你的 MLP 模型

# ===== 載入模型：包含 model + scaler + label_encoder =====
data = joblib.load(MODEL_PATH)
mlp = data["model"]
scaler = data["scaler"]
le = data["label_encoder"]

# ===== Mediapipe init =====
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2,
                    enable_segmentation=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# ===== 關節映射 =====
LM = {
    'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12,
    'LEFT_ELBOW': 13, 'RIGHT_ELBOW': 14,
    'LEFT_WRIST': 15, 'RIGHT_WRIST': 16,
    'LEFT_HIP': 23, 'RIGHT_HIP': 24,
    'LEFT_KNEE': 25, 'RIGHT_KNEE': 26,
    'LEFT_ANKLE': 27, 'RIGHT_ANKLE': 28,
}

# 12 個特徵的邊
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
    ('RIGHT_KNEE', 'RIGHT_ANKLE'),
]

def standardize(features):
    return scaler.transform(features)

# ===== 打開攝影機 =====
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        left_hip = np.array([landmarks[LM['LEFT_HIP']].x * w,
                             landmarks[LM['LEFT_HIP']].y * h])
        right_hip = np.array([landmarks[LM['RIGHT_HIP']].x * w,
                              landmarks[LM['RIGHT_HIP']].y * h])
        hip_width = np.linalg.norm(left_hip - right_hip)
        if hip_width == 0:
            continue

        features = []
        for (p1, p2) in EDGES:
            x1, y1 = landmarks[LM[p1]].x * w, landmarks[LM[p1]].y * h
            x2, y2 = landmarks[LM[p2]].x * w, landmarks[LM[p2]].y * h
            dist = np.linalg.norm([x2 - x1, y2 - y1]) / hip_width
            features.append(dist)

        features = np.array(features).reshape(1, -1)
        features_scaled = standardize(features)

        # 進行 MLP 預測
        try:
            pred_idx = mlp.predict(features_scaled)[0]
            pred_label = le.inverse_transform([pred_idx])[0]
            proba = mlp.predict_proba(features_scaled)[0].max()
        except:
            pred_label = "NaN"
            proba = 0.0

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.putText(frame, f"{pred_label} ({proba:.2f})", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("MLP Pose Real-time Classification", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
