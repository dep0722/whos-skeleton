import os
import subprocess
import json
from glob import glob

# ====== 路徑設定 ======
openpose_bin = r"C:/mydata/sf/openpose/bin/OpenPoseDemo.exe"
video_path = r"C:/Users/User/Pictures/Camera Roll/1210_none.mp4"
json_temp_dir = r"C:/mydata/sf/openpose/output_json_temp"   # 暫存每幀 JSON
output_json = r"C:/mydata/sf/openpose/output_json/comb_json/1210_2.json"

os.makedirs(json_temp_dir, exist_ok=True)
os.makedirs(os.path.dirname(output_json), exist_ok=True)

# ====== 1️⃣ 先呼叫 OpenPose 產生每幀 JSON ======
cmd = [
    openpose_bin,
    "--video", video_path,
    "--write_json", json_temp_dir,
    "--display", "0",
    "--render_pose", "0"
]

subprocess.run(cmd)

# ====== 2️⃣ 後處理：合併 JSON，不篩掉任何畫格 ======
all_frames = []
frame_id = 0

json_files = sorted(glob(os.path.join(json_temp_dir, "*.json")))

for jf in json_files:
    with open(jf, "r") as f:
        data = json.load(f)

    # 如果偵測不到骨架，仍存入空 keypoints
    if data.get("people"):
        kp = data["people"][0]["pose_keypoints_2d"]
    else:
        kp = []

    all_frames.append({
        "frame_index": frame_id,
        "keypoints": kp
    })
    frame_id += 1

# ====== 3️⃣ 輸出單一 JSON ======
with open(output_json, "w") as f:
    json.dump(all_frames, f, indent=4)

print(f"Done! 共輸出 {len(all_frames)} 個畫格")
print(f"輸出檔案：{output_json}")