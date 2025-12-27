'''
mp4 轉 CSV & JSON，轉出來之後的平滑化處理
輸入的是 JSON 檔案，輸出 CSV 以及 JSON
'''

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

import os
import json
import glob
import csv
import subprocess
import numpy as np

# -----------------------------
# OneEuroFilter Implementation
# -----------------------------
class OneEuroFilter:
    def __init__(self, freq, min_cutoff=1.0, beta=0.01, d_cutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None

    def alpha(self, cutoff):
        tau = 1.0 / (2 * np.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def filter(self, x):
        x = np.array(x)
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = np.zeros_like(x)
            return x

        dx = (x - self.x_prev) * self.freq
        dx_hat = self.dx_prev + self.alpha(self.d_cutoff) * (dx - self.dx_prev)
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self.alpha(cutoff)
        x_hat = self.x_prev + a * (x - self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat

def apply_one_euro_filter(pose_seq, freq=30, min_cutoff=1.0, beta=0.01, d_cutoff=1.0):
    T, J, _ = pose_seq.shape
    filtered_seq = np.zeros_like(pose_seq)
    filters = [[OneEuroFilter(freq, min_cutoff, beta, d_cutoff) for _ in range(2)] for _ in range(J)]

    for t in range(T):
        for j in range(J):
            for k in range(2):  # only x, y
                filtered_seq[t,j,k] = filters[j][k].filter(pose_seq[t,j,k])
            filtered_seq[t,j,2] = pose_seq[t,j,2]  # score 不過濾
    return filtered_seq

# -----------------------------
# Export functions
# -----------------------------
def export_pose_json(pose_seq, json_path):
    T, J, _ = pose_seq.shape
    frames = []

    for t in range(T):
        frame = {
            "frame_index": t,
            "people": [{
                "pose_keypoints_2d": pose_seq[t].reshape(-1).tolist()
            }]
        }
        frames.append(frame)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(frames, f, indent=2)

def export_csv(pose_seq, csv_path):
    T = pose_seq.shape[0]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        header = ["frame_index"]
        for joint in JOINT_NAMES:
            for c in COORDS:
                header.append(f"{joint}_{c}")
        writer.writerow(header)

        for t in range(T):
            row = [t]
            for j in range(NUM_JOINTS):
                row.extend(pose_seq[t,j,:].tolist())
            writer.writerow(row)

# -----------------------------
# Path configuration
# -----------------------------
OPENPOSE_EXE = r"C:/mydata/sf/openpose/bin/OpenPoseDemo.exe"
name = "1217_5"
MERGED_JSON = r"C:\mydata\sf\open\output_json\1217_5.json"
CSV_FEATURE = r"C:/mydata/sf/open/output_csv/"+ name +"_feature.csv"
JSON_FEATURE = r"C:/mydata/sf/open/output_json/" + name +"_feature.json"


# os.makedirs(TMP_JSON_DIR, exist_ok=True)

# -----------------------------
# Step 1: Load merged JSON (已存在)
# -----------------------------
with open(MERGED_JSON, "r", encoding="utf-8") as f:
    frames = json.load(f)

frames = sorted(frames, key=lambda x: x["frame_index"])

pose_sequence = []

for frame in frames:
    people = frame.get("people", [])
    if not people:
        pose_sequence.append(np.full((NUM_JOINTS, 3), np.nan))
        continue

    kp = np.array(people[0]["pose_keypoints_2d"]).reshape(NUM_JOINTS, 3)
    pose_sequence.append(kp)

pose_sequence = np.stack(pose_sequence)
T = pose_sequence.shape[0]

print("[OK] Loaded merged JSON")
print("[OK] pose_sequence shape:", pose_sequence.shape)


# -----------------------------
# Step 3: Load merged JSON
# -----------------------------
pose_sequence = []

for frame in frames:
    people = frame.get("people", [])
    if not people:
        pose_sequence.append(np.full((NUM_JOINTS,3), np.nan))
        continue
    keypoints = people[0]["pose_keypoints_2d"]
    kp = np.array(keypoints).reshape(NUM_JOINTS,3)
    pose_sequence.append(kp)

pose_sequence = np.stack(pose_sequence)
T = pose_sequence.shape[0]

print("[OK] pose_sequence shape:", pose_sequence.shape)

# -----------------------------
# Step 3.5: Temporal refinement
# -----------------------------
# 關節 index（BODY_25）
L_ANKLE = 14
R_ANKLE = 11
L_KNEE  = 13
R_KNEE  = 10
L_HIP   = 12
R_HIP   = 9

pose_feature = pose_sequence.copy()

def euclid(a, b):
    return np.linalg.norm(a[:2] - b[:2])

MAX_JUMP = 80

for t in range(1, T):
    prev = pose_feature[t-1]
    curr = pose_feature[t]

    # 左右腳 swap 修正
    d_ll = euclid(curr[L_ANKLE], prev[L_ANKLE])
    d_lr = euclid(curr[L_ANKLE], prev[R_ANKLE])
    if d_lr < d_ll:
        curr[[L_ANKLE, R_ANKLE]] = curr[[R_ANKLE, L_ANKLE]]
        curr[[L_KNEE, R_KNEE]] = curr[[R_KNEE, L_KNEE]]
        curr[[L_HIP, R_HIP]] = curr[[R_HIP, L_HIP]]

    # Motion gating
    for j in [L_ANKLE, R_ANKLE, L_KNEE, R_KNEE]:
        if euclid(curr[j], prev[j]) > MAX_JUMP:
            curr[j] = prev[j]

    pose_feature[t] = curr

# -----------------------------
# Step 3.6: OneEuroFilter smoothing
# -----------------------------
pose_feature = apply_one_euro_filter(pose_feature, freq=30, min_cutoff=1.0, beta=0.01, d_cutoff=1.0)

# -----------------------------
# Step 4: Export JSON & CSV
# -----------------------------
export_pose_json(pose_feature, JSON_FEATURE)
export_csv(pose_feature, CSV_FEATURE)

print("[OK] Feature JSON / CSV 輸出完成")
