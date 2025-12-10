'''
開啟外部攝影機連拍並儲存 + 匯出關節座標CSV
'''
import time
import os
import csv
from datetime import datetime
import cv2
import random  # 模擬關節座標時使用

# === 設定 ===
INTERVAL = 0.5   # 每張照片間隔秒數
DURATION = 5     # 拍攝總時長（秒）
LABEL = "A"      # 類別標籤

# === 關節名稱 ===
JOINTS = [
    "NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_EAR", "RIGHT_EAR",
    "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE",
    "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
]

# === 輸出設定 ===
BASE_OUTPUT_DIR = "outputs"
timestamp_run = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_FOLDER = os.path.join(BASE_OUTPUT_DIR, timestamp_run)
IMAGE_FOLDER = os.path.join(RUN_FOLDER, "你的圖片")
CSV_PATH = os.path.join(RUN_FOLDER, "你的csv.csv")

# 建立資料夾
os.makedirs(IMAGE_FOLDER, exist_ok=True)

def main():
    # 攝影機設定
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    if not cap.isOpened():
        raise RuntimeError('❌ 無法開啟攝影機')

    start_t = time.time()
    ix = 0

    # === 建立 CSV 欄位名稱 ===
    headers = []
    for name in JOINTS:
        headers.extend([f"{name}_x", f"{name}_y"])
    headers.extend(["label", "filename"])

    # === 開啟 CSV 檔案 ===
    with open(CSV_PATH, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        try:
            while True:
                now = time.time()
                if DURATION and (now - start_t) > DURATION:
                    print('📸 拍攝完成')
                    break

                ret, frame = cap.read()
                if not ret:
                    print('⚠️ 無法讀取影格，結束')
                    break

                ix += 1
                filename = f"IMG_{timestamp_run}_{ix:04d}.jpg"
                filepath = os.path.join(IMAGE_FOLDER, filename)
                cv2.imwrite(filepath, frame)
                print(f"[{ix}] 已儲存 -> {filepath}")

                # === 模擬關節座標（未連YOLOv7前使用） ===
                # 之後可替換成 YOLOv7 HumanPose 的偵測輸出結果
                row = []
                for _ in JOINTS:
                    x = round(random.uniform(0.0, 640.0), 2)
                    y = round(random.uniform(0.0, 480.0), 2)
                    row.extend([x, y])
                row.extend([LABEL, filename])
                writer.writerow(row)

                next_t = start_t + ix * INTERVAL
                sleep_time = next_t - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print('🛑 使用者中止')
        finally:
            cap.release()
            print(f'✅ 所有圖片與CSV已儲存至：{RUN_FOLDER}')

if __name__ == '__main__':
    main()
