'''
開啟攝影機並丟骨架判斷成 CSV 的版本
'''
import time
import os
from datetime import datetime
import csv
import cv2
import mediapipe as mp

# === 設定 ===
INTERVAL = 0.5
DURATION = 30
LABEL = "AAA"  # ✅ 這裡自訂你的標籤名稱（A, B, C ...）

# === 輸出設定 ===
BASE_OUTPUT_DIR = "outputs"
timestamp_run = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_FOLDER = os.path.join(BASE_OUTPUT_DIR, timestamp_run)
IMAGE_FOLDER = os.path.join(RUN_FOLDER, "C:/mydata/sf/conda/1025_test/p")
OUTPUT_CSV = os.path.join(RUN_FOLDER, "C:/mydata/sf/conda/1025_test/test_csv/pose_1030.csv")

os.makedirs(IMAGE_FOLDER, exist_ok=True)

# === Mediapipe 標記定義 ===
LM = {
    'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12,
    'LEFT_ELBOW': 13, 'RIGHT_ELBOW': 14,
    'LEFT_WRIST': 15, 'RIGHT_WRIST': 16,
    'LEFT_HIP': 23, 'RIGHT_HIP': 24,
    'LEFT_KNEE': 25, 'RIGHT_KNEE': 26,
    'LEFT_ANKLE': 27, 'RIGHT_ANKLE': 28,
    'LEFT_HEEL': 29, 'RIGHT_HEEL': 30
}

PAIRS = [
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

HEADER = [f"{a}_{b}" for a, b in PAIRS] + ["label", "filename"]

def euclidean(a, b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** 0.5

def get_landmark_xy(landmarks, idx):
    lm = landmarks[idx]
    return (lm.x, lm.y, getattr(lm, 'visibility', 0.0))

def compute_row(landmarks):
    vals = []
    for a, b in PAIRS:
        ia = LM[a]
        ib = LM[b]
        xa, ya, va = get_landmark_xy(landmarks, ia)
        xb, yb, vb = get_landmark_xy(landmarks, ib)
        if (va is None or vb is None) or (va < 0.3 or vb < 0.3):
            vals.append(float('nan'))
        else:
            vals.append(euclidean((xa, ya), (xb, yb)))
    return vals

def main():
    # 🔹 攝影機設定（建議 Logitech C270 用這組）
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise RuntimeError('❌ 無法開啟攝影機')

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # 建立 CSV 並確保標頭存在
    file_exists = os.path.isfile(OUTPUT_CSV)
    csvfile = open(OUTPUT_CSV, 'a', newline='', encoding='utf-8')
    writer = csv.writer(csvfile)
    if not file_exists:
        writer.writerow(HEADER)
        csvfile.flush()

    start_t = time.time()
    ix = 0

    try:
        while True:
            now = time.time()
            if DURATION and (now - start_t) > DURATION:
                print('到達指定總時長，結束。')
                break

            ret, frame = cap.read()
            if not ret:
                print('⚠️ 無法讀取影格，結束。')
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)

            if results.pose_landmarks:
                row_vals = compute_row(results.pose_landmarks.landmark)
            else:
                row_vals = [float('nan')] * len(PAIRS)

            # === 儲存影像 ===
            ix += 1
            filename = f"IMG_{timestamp_run}_{ix:04d}.jpg"
            filepath = os.path.join(IMAGE_FOLDER, filename)
            cv2.imwrite(filepath, frame)

            # === 寫入 CSV ===
            writer.writerow(row_vals + [LABEL, filename])
            csvfile.flush()

            print(f"[{ix}] Captured -> {filename} | Label: {LABEL}")

            next_t = start_t + ix * INTERVAL
            sleep_time = next_t - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print('\n使用者中止 (Ctrl+C)')
    finally:
        csvfile.close()
        cap.release()
        pose.close()
        print(f'✅ 資料已儲存至：{RUN_FOLDER}')
        print('清理完成。')

if __name__ == '__main__':
    main()
