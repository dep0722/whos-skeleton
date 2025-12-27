import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

BASE_DIR = r"C:/mydata/sf/conda/1025_test/p1105"
OUTPUT_CSV = r"C:/mydata/sf/conda/1025_test/test_csv/1110test_3d.csv"
OUTPUT_IMG_DIR = r"C:/mydata/sf/conda/1025_test/test_photo/"

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

def process_image(img_path, label, pose, idx, total):
    try:
        image = cv2.imread(img_path)
        if image is None:
            print(f"[{idx}/{total}] 無法讀取: {img_path}")
            return None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        try:
            results = pose.process(image_rgb)
        except Exception as e:
            print(f"[{idx}/{total}] Pose 偵測失敗: {img_path}, 跳過")
            return None

        if not results.pose_landmarks:
            print(f"[{idx}/{total}] 未偵測到人體: {img_path}")
            return None

        landmarks = results.pose_landmarks.landmark
        # 計算髖部距離（3D）
        left_hip = np.array([landmarks[LM['LEFT_HIP']].x,
                             landmarks[LM['LEFT_HIP']].y,
                             landmarks[LM['LEFT_HIP']].z])
        right_hip = np.array([landmarks[LM['RIGHT_HIP']].x,
                              landmarks[LM['RIGHT_HIP']].y,
                              landmarks[LM['RIGHT_HIP']].z])
        hip_width = np.linalg.norm(left_hip - right_hip)
        if hip_width < 1e-6:
            print(f"[{idx}/{total}] 髖部距離過小，跳過: {img_path}")
            return None

        feature_row = {}
        h, w = image.shape[:2]

        # 計算骨架 3D 長度
        for (p1, p2) in EDGES:
            a = landmarks[LM[p1]]
            b = landmarks[LM[p2]]
            dist_3d = np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2) / hip_width
            feature_row[f"{p1}_{p2}"] = dist_3d

        filename = os.path.basename(img_path)

        # 只輸出檔名結尾 0001 的圖片
        if filename.endswith(("0001.jpg", "0001.png", "0001.jpeg")):
            for (p1, p2) in EDGES:
                a = landmarks[LM[p1]]
                b = landmarks[LM[p2]]
                x1, y1 = int(a.x * w), int(a.y * h)
                x2, y2 = int(b.x * w), int(b.y * h)
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.putText(image, f"{feature_row[f'{p1}_{p2}']:.2f}",
                            (mid_x, mid_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.putText(image, f"Label: {label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(image, filename, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
            save_path = os.path.join(OUTPUT_IMG_DIR, filename)
            cv2.imwrite(save_path, image)

        feature_row["filename"] = filename
        feature_row["label"] = label

        print(f"[{idx}/{total}] 處理完成: {filename}")
        return feature_row

    except Exception as e:
        print(f"[{idx}/{total}] 處理 {img_path} 發生錯誤: {e}")
        return None


def read_images_and_extract_features():
    subfolders = [f for f in os.listdir(BASE_DIR)
                  if os.path.isdir(os.path.join(BASE_DIR, f))]
    subfolders.sort()

    all_images = []
    for subfolder in subfolders:
        folder_path = os.path.join(BASE_DIR, subfolder)
        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                all_images.append((os.path.join(folder_path, filename), subfolder))

    total_images = len(all_images)
    print(f"總共 {total_images} 張圖片，開始處理...")

    results = []

    # ✅ 單次初始化 Pose 模型
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5
    )

    for idx, (img_path, label) in enumerate(all_images, start=1):
        r = process_image(img_path, label, pose, idx, total_images)
        if r is not None:
            results.append(r)

    df = pd.DataFrame(results)
    print(f"處理完成 {len(df)} 張圖片。")
    return df


if __name__ == "__main__":
    df = read_images_and_extract_features()
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"✅ 已輸出 CSV: {OUTPUT_CSV}")
    print(f"✅ 骨架標記圖片已儲存於: {OUTPUT_IMG_DIR} (只輸出檔名結尾為 0001 的圖片)")
