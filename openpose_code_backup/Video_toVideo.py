import os
import subprocess
import json
import cv2
from glob import glob

# ====== 路徑設定 ======
openpose_bin = r"C:/mydata/sf/openpose/bin/OpenPoseDemo.exe"
video_path = r"C:/Users/User/Pictures/Camera Roll/WIN_20251210_19_00_47_Pro.mp4"
json_temp_dir = r"C:/mydata/sf/openpose/output_json_temp"  # 暫存每幀 JSON
output_video_path = r"C:/mydata/sf/openpose/output_video/1210_skeleton.mp4"

os.makedirs(json_temp_dir, exist_ok=True)
os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

# ====== 1️⃣ 呼叫 OpenPose 產生每幀 JSON & 骨架影像 ======
cmd = [
    openpose_bin,
    "--video", video_path,
    "--write_json", json_temp_dir,
    "--display", "0",
    "--write_images", json_temp_dir,  # 暫時存骨架疊圖
    "--render_pose", "1"  # 讓骨架疊上原始影像
]
subprocess.run(cmd)

# ====== 2️⃣ 讀原影片 ======
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()

# ====== 3️⃣ 讀 JSON & 疊骨架 ======
json_files = sorted(glob(os.path.join(json_temp_dir, "*.json")))

out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

for idx, jf in enumerate(json_files):
    with open(jf, "r") as f:
        data = json.load(f)

    if idx >= len(frames):
        continue
    frame = frames[idx].copy()

    # 畫骨架
    if data.get("people"):
        kp = data["people"][0]["pose_keypoints_2d"]
        BODY_25_PAIRS = [
            (1, 8), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
            (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14),
            (1, 0), (0, 15), (15, 17), (0, 16), (16, 18), (14, 19),
            (19, 20), (14, 21), (11, 22)
        ]

        for i, j in BODY_25_PAIRS:
            if len(kp) < (max(i, j)+1)*3:
                continue
            x1, y1, c1 = kp[i*3:i*3+3]
            x2, y2, c2 = kp[j*3:j*3+3]
            if c1 > 0 and c2 > 0:
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)

        # 畫關節點
        for i in range(len(kp)//3):
            x, y, c = kp[i*3:i*3+3]
            if c > 0:
                cv2.circle(frame, (int(x), int(y)), 4, (0,0,255), -1)

    out.write(frame)

out.release()
print(f"Done! 影片輸出於：{output_video_path}")
