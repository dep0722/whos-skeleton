import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import joblib

BASE_DIR = r"C:/mydata/sf/conda/1025_test/test_model/RFmodel_1101_normal.pkl"
MODEL_PATH = r"C:/Users/haoti/groundhog/Coding/Vscode/Conda/1025_test/test_model/RFmodel_1030.pkl"

# === 讀取模型 ===
rf_model = joblib.load(MODEL_PATH)

# === Mediapipe Pose 初始化 ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False, min_detection_confidence=0.5)

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

def extract_features(image):
    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if not results.pose_landmarks:
        return None
    
    landmarks = results.pose_landmarks.landmark
    
    # 計算髖寬
    hip_width = np.sqrt((landmarks[LM['LEFT_HIP']].x*w - landmarks[LM['RIGHT_HIP']].x*w)**2 +
                        (landmarks[LM['LEFT_HIP']].y*h - landmarks[LM['RIGHT_HIP']].y*h)**2)
    
    features = {}
    for (p1, p2) in EDGES:
        x1, y1 = landmarks[LM[p1]].x*w, landmarks[LM[p1]].y*h
        x2, y2 = landmarks[LM[p2]].x*w, landmarks[LM[p2]].y*h
        dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        features[f"{p1}_{p2}"] = dist / hip_width  # 正規化
    
    return features

# === 批次處理資料夾 ===
rows = []
for filename in os.listdir(BASE_DIR):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
        continue
    img_path = os.path.join(BASE_DIR, filename)
    image = cv2.imread(img_path)
    if image is None:
        continue
    
    feat = extract_features(image)
    if feat is None:
        continue
    
    df_feat = pd.DataFrame([feat])
    pred_label = rf_model.predict(df_feat)[0]
    true_label = filename[0].upper()  # 第一個字母作為正確類別
    
    rows.append({
        "filename": filename,
        "true_label": true_label,
        "pred_label": pred_label
    })

# === 計算整體準確率 ===
df_result = pd.DataFrame(rows)
print(df_result.columns)
print(df_result.head())

accuracy = (df_result["true_label"] == df_result["pred_label"]).mean()
print(f"總共處理 {len(df_result)} 張圖片")
print(f"整體準確率: {accuracy:.4f}")

# === 輸出結果 CSV ===
output_csv = os.path.join(BASE_DIR, "C:/Users/haoti/groundhog/Coding/Vscode/Conda/1025_test/exam/exam_csv/1030.csv")
df_result.to_csv(output_csv, index=False, encoding="utf-8-sig")
print(f"已輸出 CSV: {output_csv}")
