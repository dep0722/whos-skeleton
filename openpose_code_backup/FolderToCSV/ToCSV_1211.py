import os
import subprocess
import json
import csv
from glob import glob

# ====== 路徑設定 ======
openpose_bin = r"C:/mydata/sf/openpose/bin/OpenPoseDemo.exe"
image_root_dir = r"C:/mydata/sf/conda/1025_test/p2test"  # 這裡放有多個子資料夾，每個資料夾名是類別
temp_json_dir = r"C:/mydata/sf/openpose/output_json_temp"
output_csv = r"C:/mydata/sf/openpose/output_csv/1212_0.7ver_test.csv"

os.makedirs(temp_json_dir, exist_ok=True)
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# ====== CSV 標題 ======
num_joints = 25  # BODY_25 模型，每個關節有 x,y,confidence
header = []
for i in range(num_joints):
    header += [f"x{i}", f"y{i}", f"c{i}"]
header.append("label")

# ====== 開始遍歷每個子資料夾 ======
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)

    for class_dir in os.listdir(image_root_dir):
        class_path = os.path.join(image_root_dir, class_dir)
        if not os.path.isdir(class_path):
            continue

        # 呼叫 OpenPose 產生 JSON
        cmd = [
            openpose_bin,
            "--image_dir", class_path,
            "--write_json", temp_json_dir,
            "--display", "0",
            "--render_pose", "0"
        ]
        subprocess.run(cmd)

        # 讀取每張 JSON
        json_files = sorted(glob(os.path.join(temp_json_dir, "*.json")))
        for jf in json_files:
            with open(jf, "r") as f:
                data = json.load(f)
            if data.get("people"):
                kp = data["people"][0]["pose_keypoints_2d"]
            else:
                kp = [0]*(num_joints*3)  # 沒偵測到骨架填 0

            # 寫入 CSV
            writer.writerow(kp + [class_dir])

        # 清空暫存 JSON
        for jf in json_files:
            os.remove(jf)

print(f"完成！CSV 已輸出到 {output_csv}")
