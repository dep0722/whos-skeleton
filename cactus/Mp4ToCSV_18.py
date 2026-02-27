'''
Mp4 to CSV&JSON
Body18 的版本
'''

import os
import json
import glob
import csv
import subprocess
import numpy as np

# -----------------------------
# Path configuration
# -----------------------------
OPENPOSE_EXE = r"C:/mydata/sf/openpose/bin/OpenPoseDemo.exe"
VIDEO_PATH  = r"C:\mydata\sf\open\walking_video\1217_3.mp4"
name = "1223_3"
TMP_JSON_DIR = r"C:/mydata/sf/cactus/TempJson"
MERGED_JSON  = r"C:/mydata/sf/open/output_json/" + name + "_18.json"
CSV_OUTPUT   = r"C:/mydata/sf/open/output_csv/" + name + "_18.csv"

os.makedirs(TMP_JSON_DIR, exist_ok=True)

# -----------------------------
# Step 1: Run OpenPose (BODY_18)
# -----------------------------
subprocess.run(
    [
        OPENPOSE_EXE,
        "--video", VIDEO_PATH,
        "--write_json", TMP_JSON_DIR,
        "--display", "0",
        "--render_pose", "0",
        "--model_pose", "COCO"
    ],
    cwd=r"C:/mydata/sf/openpose",
    check=True
)

# -----------------------------
# Step 2: Merge per-frame JSON
# -----------------------------
frame_files = sorted(
    glob.glob(os.path.join(TMP_JSON_DIR, "*_keypoints.json"))
)

merged_frames = []

for frame_index, jf in enumerate(frame_files):
    with open(jf, "r", encoding="utf-8") as f:
        frame_data = json.load(f)

    frame_data["frame_index"] = frame_index
    merged_frames.append(frame_data)

with open(MERGED_JSON, "w", encoding="utf-8") as f:
    json.dump(merged_frames, f, indent=2)

print(f"[OK] 合併 JSON 完成，共 {len(merged_frames)} 幀")

# -----------------------------
# Step 3: Load merged JSON
# -----------------------------
with open(MERGED_JSON, "r", encoding="utf-8") as f:
    frames = json.load(f)

frames = sorted(frames, key=lambda x: x["frame_index"])

NUM_JOINTS = 18  # BODY_18
pose_sequence = []

for frame in frames:
    people = frame.get("people", [])

    if not people:
        pose_sequence.append(np.full((NUM_JOINTS, 3), np.nan))
        continue

    keypoints = people[0]["pose_keypoints_2d"]
    kp = np.array(keypoints).reshape(NUM_JOINTS, 3)
    pose_sequence.append(kp)

pose_sequence = np.stack(pose_sequence)  # (T, 18, 3)

print("[OK] pose_sequence shape:", pose_sequence.shape)

# -----------------------------
# Step 4: Export CSV
# -----------------------------
JOINT_NAMES = [
    "Nose", "Neck",
    "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist",
    "RHip", "RKnee", "RAnkle",
    "LHip", "LKnee", "LAnkle",
    "REye", "LEye", "REar", "LEar"
]

COORDS = ["x", "y", "score"]

T = pose_sequence.shape[0]

with open(CSV_OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    header = ["frame_index"]
    for joint in JOINT_NAMES:
        for c in COORDS:
            header.append(f"{joint}_{c}")
    writer.writerow(header)

    for t in range(T):
        row = [t]
        for j in range(NUM_JOINTS):
            row.extend(pose_sequence[t, j, :].tolist())
        writer.writerow(row)

print(f"[OK] CSV 輸出完成：{CSV_OUTPUT}")

# -----------------------------
# Step 5: Clean TEMP JSON
# -----------------------------
for jf in glob.glob(os.path.join(TMP_JSON_DIR, "*.json")):
    try:
        os.remove(jf)
    except Exception as e:
        print(f"[WARN] 無法刪除 {jf}: {e}")

try:
    os.rmdir(TMP_JSON_DIR)
except OSError:
    pass

print("[OK] TEMP JSON 已清除")