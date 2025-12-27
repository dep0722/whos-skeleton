'''
把 JSON 轉成 CSV
'''


import os
import json
import csv

json_dir = r"C:/mydata/sf/openpose/output_json/1208test"
output_csv = r"C:/mydata/sf/openpose/output_csv/merged.csv"

# 25 個 COCO keypoints
NUM_KEYPOINTS = 25
MIN_VISIBILITY = 0.3

files = sorted([f for f in os.listdir(json_dir) if f.endswith(".json")])

rows = []
for idx, fn in enumerate(files):
    path = os.path.join(json_dir, fn)
    
    with open(path, "r") as f:
        data = json.load(f)

    # 若沒有偵測到人 → skip
    if len(data["people"]) == 0:
        continue
    
    kps = data["people"][0]["pose_keypoints_2d"]

    # reshape [x0,y0,c0,x1,y1,c1,...]
    keypoints = [kps[i:i+3] for i in range(0, len(kps), 3)]

    # --- ① 檢查可見度 ---
    valid_count = sum(1 for x, y, c in keypoints if c >= MIN_VISIBILITY)
    if valid_count < NUM_KEYPOINTS * 0.30:
        # 少於 30% 的點可見 → skip
        continue

    # --- ② 加到總表 ---
    flat = [v for triplet in keypoints for v in triplet]
    rows.append([idx] + flat)

# --- ③ 寫 CSV ---
header = ["frame"] + [f"{p}_{i}" for i in range(NUM_KEYPOINTS) for p in ["x","y","c"]]

with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print("Done! CSV saved to:", output_csv)
