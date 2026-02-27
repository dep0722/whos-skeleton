import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# User settings
# =========================
INPUT_DIR = r"C:\mydata\sf\open\output_csv\1230One\C"   # <<<<< 資料夾
OUTPUT_DIR = r"C:\mydata\sf\open\output_images"
ANGLE_FIG_NAME = "1230C_knee_angle.png"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Helper: compute angle
# =========================
def compute_angle(A, B, C):
    BA = A - B
    BC = C - B

    dot = np.sum(BA * BC, axis=1)
    norm = np.linalg.norm(BA, axis=1) * np.linalg.norm(BC, axis=1)

    cos_theta = np.clip(dot / (norm + 1e-8), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

def joint_xy(df, name):
    return df[[f"{name}_x", f"{name}_y"]].values

# =========================
# Collect CSV files
# =========================
csv_files = sorted([
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith(".csv")
])

assert len(csv_files) > 0, "No CSV files found."

# =========================
# Prepare figure
# =========================
fig, axes = plt.subplots(
    len(csv_files), 1,
    figsize=(18, 4 * len(csv_files)),
    sharex=False
)

if len(csv_files) == 1:
    axes = [axes]

# =========================
# Process each CSV
# =========================
for ax, csv_name in zip(axes, csv_files):
    csv_path = os.path.join(INPUT_DIR, csv_name)
    df = pd.read_csv(csv_path)
    t = df["frame_index"].values

    # 判斷 L or R
    if "L" in csv_name:
        hip   = joint_xy(df, "LHip")
        knee  = joint_xy(df, "LKnee")
        ankle = joint_xy(df, "LAnkle")
        side = "Left"
    elif "R" in csv_name:
        hip   = joint_xy(df, "RHip")
        knee  = joint_xy(df, "RKnee")
        ankle = joint_xy(df, "RAnkle")
        side = "Right"
    else:
        print(f"Skip (no L/R): {csv_name}")
        continue

    angle = compute_angle(hip, knee, ankle)

    ax.plot(t, angle, linewidth=1.5)
    ax.set_title(f"{csv_name} ({side} Knee)")
    ax.set_ylabel("Angle (deg)")
    ax.grid(True, alpha=0.3)

# =========================
# Final layout & save
# =========================
axes[-1].set_xlabel("Frame")

fig.suptitle("Knee Angle vs Frame (Separated CSVs)", fontsize=18)
fig.tight_layout(rect=[0.05, 0.03, 1, 0.95])

out_path = os.path.join(OUTPUT_DIR, ANGLE_FIG_NAME)
fig.savefig(out_path, dpi=200)
plt.close(fig)

print(f"Saved angle plot: {out_path}")
print("Done.")
