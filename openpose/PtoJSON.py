import os
import subprocess
import json
from glob import glob

# ====== 路徑設定 ======
openpose_bin = r"C:/mydata/sf/openpose/bin/OpenPoseDemo.exe"
image_dir = r"C:/mydata/sf/image_data/p1105/E"  # 放圖片的資料夾
json_temp_dir = r"C:/mydata/sf/openpose/output_json/1105train/E"  # 暫存 JSON

os.makedirs(json_temp_dir, exist_ok=True)

# ====== 1️⃣ 呼叫 OpenPose 產生 JSON ======
cmd = [
    openpose_bin,
    "--image_dir", image_dir,
    "--write_json", json_temp_dir,
    "--display", "0",
    "--render_pose", "0"
]

subprocess.run(cmd)

# ====== 2️⃣ 合併所有 JSON ======
all_frames = []
image_files = sorted(glob(os.path.join(json_temp_dir, "*.json")))

for idx, jf in enumerate(image_files):
    with open(jf, "r") as f:
        data = json.load(f)

    # 🎯 data 可能是 list / dict → 自動處理
    if isinstance(data, list):
        # OpenPose 有些版本是一張圖一筆 list
        if len(data) > 0:
            people_data = data[0].get("people", [])
        else:
            people_data = []
    else:
        # OpenPose 的正常格式
        people_data = data.get("people", [])

    # 🎯 取 keypoints（若沒有偵測到人 → 空 list）
    if people_data:
        kp = people_data[0].get("pose_keypoints_2d", [])
    else:
        kp = []

    all_frames.append({
        "frame_index": idx,
        "keypoints": kp
    })

# ====== 3️⃣ 輸出最終合併檔 ======
output_json = os.path.join(json_temp_dir, "combined_images.json")
with open(output_json, "w") as f:
    json.dump(all_frames, f, indent=4)

# ====== 4️⃣ 刪除所有暫存 JSON ======
for jf in image_files:
    os.remove(jf)

print(f"Done! 共處理 {len(all_frames)} 張圖片")
print(f"唯一保留的 JSON：{output_json}")
