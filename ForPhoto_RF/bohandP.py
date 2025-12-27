'''
讀取資料夾中所有子資料夾的照片並匯出關節座標CSV
每個子資料夾名稱代表一個類別（label）
使用 Mediapipe Pose 原始座標輸出
'''
import os
import csv
import cv2
from datetime import datetime
import mediapipe as mp

# === 設定 ===
BASE_IMAGE_DIR = "C:/mydata/sf/conda/1025_test/p1105test"
CSV_PATH = "C:/mydata/sf/conda/1025_test/test_csv/1113_p1105test.csv"

JOINTS = [
    "NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_EAR", "RIGHT_EAR",
    "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE",
    "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
]

# === Mediapipe 初始化 ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def extract_pose_landmarks_raw(image_path):
    """讀取圖片並回傳 JOINTS 對應的原始 (x, y) 座標 0~1"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ 無法讀取圖片: {image_path}")
        return [None] * len(JOINTS) * 2  # 無法讀取時填 None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    row = []
    if results.pose_landmarks:
        for joint_name in JOINTS:
            lm = getattr(mp_pose.PoseLandmark, joint_name, None)
            if lm is not None:
                landmark = results.pose_landmarks.landmark[lm]
                row.extend([landmark.x, landmark.y])
            else:
                row.extend([None, None])
    else:
        row = [None] * len(JOINTS) * 2

    return row

def main():
    # === 建立 CSV 欄位名稱 ===
    headers = []
    for name in JOINTS:
        headers.extend([f"{name}_x", f"{name}_y"])
    headers.extend(["label", "filename"])

    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

    with open(CSV_PATH, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        ix = 0
        # 🔁 遞迴讀取所有子資料夾與檔案
        for root, dirs, files in os.walk(BASE_IMAGE_DIR):
            # 用子資料夾名稱當作 label（相對路徑最後一層）
            label = os.path.basename(root)
            # 如果在最上層（例如 BASE_IMAGE_DIR 本身），略過
            if root == BASE_IMAGE_DIR:
                continue

            for filename in files:
                if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                filepath = os.path.join(root, filename)
                ix += 1
                print(f"[{ix}] 處理 -> {filepath} (label={label})")

                # 使用 Mediapipe 取得原始座標
                row = extract_pose_landmarks_raw(filepath)
                row.extend([label, filename])
                writer.writerow(row)

    print(f'✅ CSV 已儲存至：{CSV_PATH}')

if __name__ == '__main__':
    main()
