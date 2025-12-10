import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from concurrent.futures import ProcessPoolExecutor

BASE_DIR = r"C:/mydata/sf/conda/1025_test/p1105test"

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

EDGES = [
    ('LEFT_SHOULDER', 'LEFT_ELBOW'),
    ('LEFT_ELBOW', 'LEFT_WRIST'),
    ('RIGHT_SHOULDER', 'RIGHT_ELBOW'),
    ('RIGHT_ELBOW', 'RIGHT_WRIST'),
    ('LEFT_SHOULDER', 'RIGHT_SHOULDER'),
    ('LEFT_HIP', 'RIGHT_HIP'),
    ('LEFT_HIP', 'LEFT_KNEE'),
    ('LEFT_KNEE', 'LEFT_ANKLE'),
    ('RIGHT_HIP', 'RIGHT_KNEE'),
    ('RIGHT_KNEE', 'RIGHT_ANKLE'),
]

# --- 每個進程初始化 Mediapipe Pose ---
def init_pose():
    mp_pose = mp.solutions.pose
    return mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5
    )

# --- 單張圖片處理函數 ---
def process_image(args):
    img_path, label = args
    pose = init_pose()
    image = cv2.imread(img_path)
    if image is None:
        print(f"無法讀取: {img_path}")
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        print(f"未偵測到人體: {img_path}")
        return None

    landmarks = results.pose_landmarks.landmark
    left_hip = np.array([landmarks[LM['LEFT_HIP']].x, landmarks[LM['LEFT_HIP']].y])
    right_hip = np.array([landmarks[LM['RIGHT_HIP']].x, landmarks[LM['RIGHT_HIP']].y])
    hip_width = np.linalg.norm(left_hip - right_hip)
    if hip_width < 1e-6:
        return None

    feature_row = {}
    for (p1, p2) in EDGES:
        a = landmarks[LM[p1]]
        b = landmarks[LM[p2]]
        dist = np.linalg.norm(np.array([a.x - b.x, a.y - b.y])) / hip_width
        feature_row[f"{p1}_{p2}"] = dist

    feature_row["label"] = label
    feature_row["filename"] = os.path.basename(img_path)
    return feature_row

# --- 讀取所有子資料夾圖片並平行處理 ---
def read_images_and_extract_features_parallel():
    subfolders = [f for f in os.listdir(BASE_DIR)
                    if os.path.isdir(os.path.join(BASE_DIR, f))]
    subfolders.sort()

    all_images = []
    for subfolder in subfolders:
        folder_path = os.path.join(BASE_DIR, subfolder)
        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                all_images.append((os.path.join(folder_path, filename), subfolder))

    print(f"總共 {len(all_images)} 張圖片，開始平行處理...")

    results = []
    # max_workers 可根據 CPU 核心數調整，例如 8
    with ProcessPoolExecutor(max_workers=8) as executor:
        for r in executor.map(process_image, all_images):
            if r is not None:
                results.append(r)

    df = pd.DataFrame(results)
    print(f"處理完成 {len(df)} 張圖片。")
    return df

if __name__ == "__main__":
    df = read_images_and_extract_features_parallel()
    output_path = r"C:/mydata/sf/conda/1025_test/test_csv/1106test.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"已輸出 CSV:{output_path}")
