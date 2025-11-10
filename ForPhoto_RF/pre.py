'''
將 p 中的各個小資料夾中的圖片依照名字做成分類 CSV
'''
import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp  # pip install mediapipe opencv-python pandas numpy

BASE_DIR = r"C:/mydata/sf/conda/1025_test/p2"

# === Mediapipe Pose 初始化 ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2,
                    enable_segmentation=False, min_detection_confidence=0.5)

# === 關節點定義 ===
LM = {
    'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12,
    'LEFT_ELBOW': 13, 'RIGHT_ELBOW': 14,
    'LEFT_WRIST': 15, 'RIGHT_WRIST': 16,
    'LEFT_HIP': 23, 'RIGHT_HIP': 24,
    'LEFT_KNEE': 25, 'RIGHT_KNEE': 26,
    'LEFT_ANKLE': 27, 'RIGHT_ANKLE': 28,
    'LEFT_HEEL': 29, 'RIGHT_HEEL': 30,
}

# === 要計算的邊 ===
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

'''
這個函數會直接把 p 內的所有小資料夾依照｢子資料夾｣名稱賦予類別
並把資料做成 csv 存在資料夾結構中的 test_csv 資料夾中
最後兩欄是類別名以及圖檔名，其他都是邊長資料(以髖寬比例正規化)
'''
def read_images_and_extract_features():
    """讀取所有資料夾圖片，提取骨架邊長特徵（以髖寬正規化）"""
    subfolders = [f for f in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, f))]
    subfolders.sort()
    data_rows = []

    for subfolder in subfolders:
        folder_path = os.path.join(BASE_DIR, subfolder)
        print(f"處理資料夾: {subfolder}")

        for filename in os.listdir(folder_path):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                continue

            img_path = os.path.join(folder_path, filename)
            image = cv2.imread(img_path)
            if image is None:
                print(f"無法讀取: {filename}")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if not results.pose_landmarks:
                print(f"未偵測到人體: {filename}")
                continue

            landmarks = results.pose_landmarks.landmark

            # --- hip width base （使用 normalized xy，更穩）---
            left_hip = np.array([landmarks[LM['LEFT_HIP']].x,
                                    landmarks[LM['LEFT_HIP']].y])
            right_hip = np.array([landmarks[LM['RIGHT_HIP']].x,
                                    landmarks[LM['RIGHT_HIP']].y])
            hip_width = np.linalg.norm(left_hip - right_hip)
            if hip_width < 1e-6:
                continue

            # 計算各邊邊長（normalized space 計算）
            feature_row = {}
            for (p1, p2) in EDGES:
                a = landmarks[LM[p1]]
                b = landmarks[LM[p2]]
                dist = np.linalg.norm(np.array([a.x - b.x, a.y - b.y])) / hip_width
                feature_row[f"{p1}_{p2}"] = dist

            feature_row["label"] = subfolder
            feature_row["filename"] = filename
            data_rows.append(feature_row)

    df = pd.DataFrame(data_rows)
    print(f"總共處理 {len(df)} 張圖片。")
    return df


if __name__ == "__main__":
    df = read_images_and_extract_features()
    output_path = r"C:/mydata/sf/conda/1025_test/test_csv/pose_1104.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"已輸出 CSV:{output_path}")
