import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

# 設定資料夾
BASE_DIR = r"C:/mydata/sf/conda/1025_test/hands"
OUTPUT_CSV = r"C:/mydata/sf/conda/1025_test/test_csv/1110_z_stability.csv"

# 手骨架 Landmark
HAND_LM = {
    'LEFT': ['LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST'],
    'RIGHT': ['RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST']
}

LM_IDX = {
    'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12,
    'LEFT_ELBOW': 13, 'RIGHT_ELBOW': 14,
    'LEFT_WRIST': 15, 'RIGHT_WRIST': 16,
}

EDGES = [
    ('SHOULDER', 'ELBOW'),
    ('ELBOW', 'WRIST'),
]

def process_hand_landmarks(landmarks, side='LEFT'):
    """
    計算手骨架 3D 長度，正規化，並收集 Z 值
    """
    points = [landmarks[LM_IDX[l]] for l in HAND_LM[side]]
    dists = []
    zs = []

    # 以肩到手肘距離正規化
    norm_dist = np.linalg.norm(np.array([points[0].x, points[0].y, points[0].z]) -
                            np.array([points[1].x, points[1].y, points[1].z]))
    if norm_dist < 1e-6:
        norm_dist = 1.0  # 避免除零

    for i, j in [(0,1),(1,2)]:
        a = np.array([points[i].x, points[i].y, points[i].z])
        b = np.array([points[j].x, points[j].y, points[j].z])
        dist = np.linalg.norm(a-b) / norm_dist
        dists.append(dist)
        zs.append(a[2])
        if j==2:  # 最後一個點也加入 Z
            zs.append(b[2])

    return dists, zs

def analyze_hands():
    subfolders = [f for f in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, f))]
    subfolders.sort()
    results = []

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5
    )

    for person in subfolders:
        folder_path = os.path.join(BASE_DIR, person)
        images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg','.png','.jpeg'))]
        images.sort()

        left_z_all = []
        right_z_all = []
        left_dists_all = []
        right_dists_all = []

        for idx, img_name in enumerate(images, start=1):
            img_path = os.path.join(folder_path, img_name)
            image = cv2.imread(img_path)
            if image is None:
                print(f"[{person}] 無法讀取 {img_name}")
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            try:
                res = pose.process(image_rgb)
            except Exception as e:
                print(f"[{person}] Pose 偵測失敗 {img_name}")
                continue
            if not res.pose_landmarks:
                print(f"[{person}] 未偵測到人體 {img_name}")
                continue

            lm = res.pose_landmarks.landmark

            l_dists, l_zs = process_hand_landmarks(lm, 'LEFT')
            r_dists, r_zs = process_hand_landmarks(lm, 'RIGHT')

            left_z_all.extend(l_zs)
            right_z_all.extend(r_zs)
            left_dists_all.extend(l_dists)
            right_dists_all.extend(r_dists)

            print(f"[{person}] {idx}/{len(images)}: {img_name} processed")

        # 統計每個人手的 Z 值
        left_z_mean = np.mean(left_z_all) if left_z_all else np.nan
        left_z_std  = np.std(left_z_all) if left_z_all else np.nan
        right_z_mean = np.mean(right_z_all) if right_z_all else np.nan
        right_z_std  = np.std(right_z_all) if right_z_all else np.nan

        # 骨架正規化距離平均
        left_dist_mean = np.mean(left_dists_all) if left_dists_all else np.nan
        right_dist_mean = np.mean(right_dists_all) if right_dists_all else np.nan

        results.append({
            'Person': person,
            'LEFT_mean_z': left_z_mean,
            'LEFT_std_z': left_z_std,
            'RIGHT_mean_z': right_z_mean,
            'RIGHT_std_z': right_z_std,
            'LEFT_norm_bone_avg': left_dist_mean,
            'RIGHT_norm_bone_avg': right_dist_mean,
            'Num_images': len(images)
        })

        print(f"[{person}] 統計完成: LEFT z mean={left_z_mean:.4f}, std={left_z_std:.4f}, RIGHT z mean={right_z_mean:.4f}, std={right_z_std:.4f}")

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"✅ 統計結果已輸出 CSV: {OUTPUT_CSV}")

if __name__ == "__main__":
    analyze_hands()
