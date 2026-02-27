import os
import json
import glob
import csv
import subprocess
import numpy as np
from pathlib import Path

# =============================
# Path configuration
# =============================
OPENPOSE_EXE = r"C:/mydata/sf/openpose/bin/OpenPoseDemo.exe"

DATASET_ROOT   = Path(r"C:\mydata\sf\open\walking_video\0130")
TMP_JSON_ROOT  = Path(r"C:/mydata/sf/cactus/TempJson")
MERGED_JSON_ROOT = Path(r"C:\mydata\sf\open\output_json\0128")
CSV_OUTPUT_ROOT  = Path(r"C:\mydata\sf\open\output_csv\0128")

NUM_JOINTS = 25  # BODY_25

JOINT_NAMES = [
    "Nose","Neck","RShoulder","RElbow","RWrist",
    "LShoulder","LElbow","LWrist","MidHip",
    "RHip","RKnee","RAnkle",
    "LHip","LKnee","LAnkle",
    "REye","LEye","REar","LEar",
    "LBigToe","LSmallToe","LHeel",
    "RBigToe","RSmallToe","RHeel"
]

COORDS = ["x", "y", "score"]

# =============================
# Main loop (videos directly)
# =============================
for video_path in sorted(DATASET_ROOT.glob("*.mp4")):
    video_name = video_path.stem
    print(f"\n[INFO] Processing {video_name}.mp4")

    # -----------------------------
    # Parse label from filename
    # 0128_{label}_{idx}.mp4
    # -----------------------------
    try:
        _, label, _ = video_name.split("_", 2)
    except ValueError:
        print(f"[WARN] Filename format error, skip: {video_name}")
        continue

    # -----------------------------
    # Prepare folders
    # -----------------------------
    (MERGED_JSON_ROOT / label).mkdir(parents=True, exist_ok=True)
    (CSV_OUTPUT_ROOT  / label).mkdir(parents=True, exist_ok=True)

    tmp_json_dir = TMP_JSON_ROOT / label / video_name
    tmp_json_dir.mkdir(parents=True, exist_ok=True)

    merged_json_path = MERGED_JSON_ROOT / label / f"{video_name}.json"
    csv_output_path  = CSV_OUTPUT_ROOT  / label / f"{video_name}.csv"

    # -----------------------------
    # Step 1: Run OpenPose
    # -----------------------------
    subprocess.run(
        [
            OPENPOSE_EXE,
            "--video", str(video_path),
            "--write_json", str(tmp_json_dir),
            "--display", "0",
            "--render_pose", "0",
            "--model_pose", "BODY_25"
        ],
        cwd=r"C:/mydata/sf/openpose",
        check=True
    )

    # -----------------------------
    # Step 2: Merge per-frame JSON
    # -----------------------------
    frame_files = sorted(tmp_json_dir.glob("*_keypoints.json"))
    merged_frames = []

    for frame_index, jf in enumerate(frame_files):
        with open(jf, "r", encoding="utf-8") as f:
            frame_data = json.load(f)
        frame_data["frame_index"] = frame_index
        merged_frames.append(frame_data)

    with open(merged_json_path, "w", encoding="utf-8") as f:
        json.dump(merged_frames, f, indent=2)

    print(f"[OK] MERGED_JSON saved → {merged_json_path}")

    # -----------------------------
    # Step 3: Load MERGED_JSON
    # -----------------------------
    pose_sequence = []

    for frame in merged_frames:
        people = frame.get("people", [])

        if not people:
            pose_sequence.append(np.full((NUM_JOINTS, 3), np.nan))
            continue

        kp = np.array(
            people[0]["pose_keypoints_2d"]
        ).reshape(NUM_JOINTS, 3)

        pose_sequence.append(kp)

    if not pose_sequence:
        print("[WARN] No valid frames, skip.")
        continue

    pose_sequence = np.stack(pose_sequence)
    T = pose_sequence.shape[0]

    # -----------------------------
    # Step 4: Export CSV (per frame)
    # -----------------------------
    with open(csv_output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        header = ["label", "frame_index"]
        for joint in JOINT_NAMES:
            for c in COORDS:
                header.append(f"{joint}_{c}")
        writer.writerow(header)

        for t in range(T):
            row = [label, t]
            for j in range(NUM_JOINTS):
                row.extend(pose_sequence[t, j, :].tolist())
            writer.writerow(row)

    print(f"[OK] CSV saved → {csv_output_path}")

    # -----------------------------
    # Step 5: Clean TEMP JSON
    # -----------------------------
    for jf in tmp_json_dir.glob("*.json"):
        jf.unlink()

    try:
        tmp_json_dir.rmdir()
        (TMP_JSON_ROOT / label).rmdir()
    except OSError:
        pass

print("\n🎉 All videos processed successfully!")
