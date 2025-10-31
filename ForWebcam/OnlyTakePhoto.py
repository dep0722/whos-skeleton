'''
開啟外部攝影機連拍並儲存
'''
import time
import os
from datetime import datetime
import cv2

# === 設定 ===
INTERVAL = 0.5   # 每張照片間隔秒數
DURATION = 121    # 拍攝總時長（秒）
# === 輸出設定 ===
BASE_OUTPUT_DIR = "outputs"
timestamp_run = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_FOLDER = os.path.join(BASE_OUTPUT_DIR, timestamp_run)
#下面記得改檔案資料夾路徑
IMAGE_FOLDER = os.path.join(RUN_FOLDER, "C:/mydata/sf/conda/1025_test/p/A")

# 建立資料夾
os.makedirs(IMAGE_FOLDER, exist_ok=True)

def main():
    # 攝影機設定
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    if not cap.isOpened():
        raise RuntimeError('無法開啟攝影機')

    start_t = time.time()
    ix = 0

    try:
        while True:
            now = time.time()
            if DURATION and (now - start_t) > DURATION:
                print('拍攝完成')
                break

            ret, frame = cap.read()
            #順轉90:cv2.ROTATE_90_CLOCKWISE，逆90:cv2.ROTATE_90_COUNTERCLOCKWISE，180:cv2.ROTATE_180
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            if not ret:
                print('無法讀取影格，結束')
                break

            ix += 1
            filename = f"IMG_{timestamp_run}_{ix:04d}.jpg"
            filepath = os.path.join(IMAGE_FOLDER, filename)
            cv2.imwrite(filepath, frame)
            print(f"[{ix}] 已儲存 -> {filepath}")

            next_t = start_t + ix * INTERVAL
            sleep_time = next_t - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print('使用者中止')
    finally:
        cap.release()
        print(f'所有圖片已儲存至：{IMAGE_FOLDER}')

if __name__ == '__main__':
    main()
