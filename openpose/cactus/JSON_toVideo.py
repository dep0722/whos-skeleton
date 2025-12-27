'''
將「單一 JSON（多 frame）」中的骨架資料轉成可視化影片
OneEuroFilter 可開關平滑

需要輸入原始影片、JSON
最後會產生視覺化影片
'''

import math
import cv2
import json
import numpy as np

# ================= One Euro Filter =================
class OneEuroFilter:
    def __init__(self, freq, mincutoff=1.0, beta=0.01, dcutoff=1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev = None
        self.dx_prev = None

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

def draw_keypoints(img, points, radius=4):
    for p in points:
        if p is None:
            continue
        x, y = p
        cv2.circle(img, (int(x), int(y)), radius, (0, 0, 255), -1)


# ================= 設定 =================
json_path = r"C:\mydata\sf\open\output_json\1217_3.json"
video_path = r"C:\mydata\sf\open\walking_video\1217_3.mp4"
output_video = r"C:\mydata\sf\open\output_video\1217_3_2.mp4"

num_keypoints = 25  # BODY_25
conf_threshold = 0.3
# =======================================

# 讀取 JSON（一次）
with open(json_path, "r") as f:
    frames_data = json.load(f)

num_frames_json = len(frames_data)

# 讀取影片
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

# 初始化 OneEuroFilter（每個關節各一組）
filters_x = [OneEuroFilter(freq=fps) for _ in range(num_keypoints)]
filters_y = [OneEuroFilter(freq=fps) for _ in range(num_keypoints)]

# ================= 畫骨架 =================
def draw_skeleton(img, points):
    BODY25_PAIRS = [
        (1,8),(1,2),(1,5),(2,3),(3,4),(5,6),(6,7),
        (8,9),(9,10),(10,11),(8,12),(12,13),(13,14),
        (1,0),(0,15),(15,17),(0,16),(16,18),
        (14,19),(19,20),(11,22),(22,23)
    ]

    for a, b in BODY25_PAIRS:
        if points[a] is None or points[b] is None:
            continue
        x1, y1 = points[a]
        x2, y2 = points[b]
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

# ================= 主迴圈 =================
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id >= num_frames_json:
        break

    data = frames_data[frame_id]

    # 沒人就直接輸出空 frame
    if len(data["people"]) == 0:
        out.write(frame)
        frame_id += 1
        continue

    kp = data["people"][0]["pose_keypoints_2d"]
    kp = np.array(kp).reshape(-1, 3)

    smoothed_points = []

    for i in range(num_keypoints):
        x, y, conf = kp[i]

        if conf < conf_threshold:
            smoothed_points.append(None)
            continue

        sx = x
        sy = y
        
        # 👉 要關掉平滑，只要把下面兩行改成 sx=x, sy=y
        # sx = filters_x[i].filter(x)
        # sy = filters_y[i].filter(y)

        smoothed_points.append((sx, sy))

    draw_skeleton(frame, smoothed_points)
    draw_keypoints(frame, smoothed_points)
    out.write(frame)


    frame_id += 1

cap.release()
out.release()

print("✅ Done! Output saved to:", output_video)
