'''
將 JSON 中的骨架資料轉成可視化的影片，修改 126-128 可以選擇是否要平滑化
'''

import math

class OneEuroFilter:
    def __init__(self, freq, mincutoff=1.0, beta=0.01, dcutoff=1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def alpha(self, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def filter(self, x):
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0
            return x

        dx = (x - self.x_prev) * self.freq
        alpha_d = self.alpha(self.dcutoff)
        dx_hat = alpha_d * dx + (1 - alpha_d) * self.dx_prev

        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        alpha_val = self.alpha(cutoff)
        x_hat = alpha_val * x + (1 - alpha_val) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat

import cv2
import json
import glob
import numpy as np

# ================ 設定 =================
json_dir = r"C:/mydata/sf/openpose/output_json/1208test"
video_path = r"C:/mydata/sf/openpose/examples/media/WIN_20251208_19_25_23_Pro.mp4"
output_video = r"C:/mydata/sf/openpose/output_video/1208output_smooth.mp4"

missing_threshold = 0.3   # 缺失超過 30% 的 keypoints → 丟棄該 frame
fps = 30                  # 影片 FPS（可自動讀取）
num_keypoints = 25        # BODY_25
# ========================================

# 初始化每個關節的濾波器
filters_x = [OneEuroFilter(freq=fps) for _ in range(num_keypoints)]
filters_y = [OneEuroFilter(freq=fps) for _ in range(num_keypoints)]

# 讀取影片
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(3))
h = int(cap.get(4))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

# 讀 JSON 檔案順序
json_files = sorted(glob.glob(json_dir + "/*_keypoints.json"))

def draw_skeleton(img, points):
    BODY25_PAIRS = [
        (1,8),(1,2),(1,5),(2,3),(3,4),(5,6),(6,7),
        (8,9),(9,10),(10,11),(8,12),(12,13),(13,14),
        (1,0),(0,15),(15,17),(0,16),(16,18),(14,19),(19,20),(11,22),(22,23)
    ]
    for (a,b) in BODY25_PAIRS:
        if points[a] is None or points[b] is None:
            continue
        x1,y1 = points[a]
        x2,y2 = points[b]
        cv2.line(img, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)

# ============= 主要處理迴圈 =================
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id >= len(json_files):
        break

    # 讀取 JSON
    with open(json_files[frame_id], "r") as f:
        data = json.load(f)

    if len(data["people"]) == 0:
        print("frame", frame_id, "no person detected → skip")
        frame_id += 1
        continue

    kp = data["people"][0]["pose_keypoints_2d"]
    kp = np.array(kp).reshape(-1, 3)

    # 判斷缺失比例
    visible = kp[:,2] > 0.05
    missing_ratio = 1 - np.mean(visible)

    if missing_ratio > missing_threshold:
        print(f"frame {frame_id} skipped (missing {missing_ratio*100:.1f}%)")
        frame_id += 1
        continue

    # 平滑
    smoothed_points = []
    for i in range(num_keypoints):
        x,y,conf = kp[i]
        if conf < 0.05:
            smoothed_points.append(None)
            continue

        # 底下兩行不註解會有平滑化的效果，並且 123 行改成 sx, sy
        sx = filters_x[i].filter(x)
        sy = filters_y[i].filter(y)
        smoothed_points.append((sx, sy))

    # 繪圖
    draw_skeleton(frame, smoothed_points)

    out.write(frame)

    frame_id += 1

cap.release()
out.release()

print("Done! Output saved to:", output_video)
