import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import json
import os

# ===== 路徑設定 =====
MODEL_PATH = r"C:/mydata/sf/conda/1025_test/test_model/KNN1101_normal.pkl"
CSV_PATH = r"C:/mydata/sf/conda/1025_test/test_csv/pose_1104.csv"
WEIGHT_JSON = r"C:/mydata/sf/conda/1025_test/test_json/test2.json"

# ===== 判斷是否有權重 =====
use_weight = "_weighted" in os.path.basename(MODEL_PATH).lower()

# ===== 載入模型 =====
knn = joblib.load(MODEL_PATH)

# ===== 從 CSV 計算標準化 =====
df = pd.read_csv(CSV_PATH)
X_csv = df.drop(columns=["label","filename"])
mean_ = X_csv.mean().values
scale_ = X_csv.std().values

def standardize(features):
    # 避免除以0造成 NaN
    safe_scale = np.where(scale_ == 0, 1, scale_)
    return (features - mean_) / safe_scale

# ===== 讀取權重（如果有） =====
weights = None
if use_weight and os.path.exists(WEIGHT_JSON):
    with open(WEIGHT_JSON, "r", encoding="utf-8") as f:
        weight_dict = json.load(f)
    weights = np.array([weight_dict.get(col, {}).get("weight", 1.0) 
                        for col in X_csv.columns])
else:
    print("未使用權重或找不到權重資訊，將忽略權重")

# ===== Mediapipe 初始化 =====
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2,
                    enable_segmentation=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# ===== 關節點與邊定義 =====
LM = {
    'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12,
    'LEFT_ELBOW': 13, 'RIGHT_ELBOW': 14,
    'LEFT_WRIST': 15, 'RIGHT_WRIST': 16,
    'LEFT_HIP': 23, 'RIGHT_HIP': 24,
    'LEFT_KNEE': 25, 'RIGHT_KNEE': 26,
    'LEFT_ANKLE': 27, 'RIGHT_ANKLE': 28,
    'LEFT_HEEL': 29, 'RIGHT_HEEL': 30,
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
    ('RIGHT_KNEE', 'RIGHT_ANKLE'),
]

# ===== 打開攝影機 =====
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # 計算髖寬正規化邊長
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
        # 標準化
        features_scaled = standardize(features)

        # 套用權重（如果有）
        if weights is not None:
            features_scaled = features_scaled * np.sqrt(weights)

        # 預測
        try:
            pred_label = knn.predict(features_scaled)[0]
            proba = knn.predict_proba(features_scaled)[0].max()  # top 1 機率
        except ValueError:
            pred_label = "NaN"
            proba = 0.0

        # 顯示結果
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        weight_status = "Weighted" if weights is not None else "Normal"
        cv2.putText(frame, f"{pred_label} ({weight_status}, {proba:.2f})", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("KNN Pose Real-time Classification", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
