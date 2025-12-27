import os
import subprocess
import json
from glob import glob
import cv2

# ====== 路徑設定 ======
openpose_bin = r"C:/mydata/sf/openpose/bin/OpenPoseDemo.exe"
image_dir = r"C:/mydata/sf/conda/1025_test/p1105/A"  # 放圖片的資料夾
json_temp_dir = r"C:/mydata/sf/openpose/output_json/1105train"  # 暫存每張圖片的 JSON
output_dir = r"C:/mydata/sf/openpose/output_images/1210_test"  # 輸出圖片

os.makedirs(json_temp_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# ====== 1️⃣ 先呼叫 OpenPose 產生每張圖片的 JSON ======
cmd = [
    openpose_bin,
    "--image_dir", image_dir,
    "--write_json", json_temp_dir,
    "--display", "0",
    "--write_images", output_dir,
    "--render_pose", "1"  # 直接把骨架疊在圖片
]

subprocess.run(cmd)

# ====== 2️⃣ 後處理：合併 JSON 成單一檔案 ======
all_frames = []
image_files = sorted(glob(os.path.join(json_temp_dir, "*.json")))

for idx, jf in enumerate(image_files):
    with open(jf, "r") as f:
        data = json.load(f)

    if data.get("people"):
        kp = data["people"][0]["pose_keypoints_2d"]
    else:
        kp = []

    all_frames.append({
        "frame_index": idx,
        "keypoints": kp
    })

output_json = os.path.join(json_temp_dir, "combined_images.json")
with open(output_json, "w") as f:
    json.dump(all_frames, f, indent=4)

print(f"Done! 共處理 {len(all_frames)} 張圖片")
print(f"JSON 檔案：{output_json}")
print(f"骨架疊圖輸出於：{output_dir}")
