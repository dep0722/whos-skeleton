import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# User settings
# =========================
INPUT_CSV  = r"C:\mydata\sf\open\output_csv\1223_3_18.csv"        # <-- change to your CSV path
OUTPUT_DIR = r"C:\mydata\sf\open\output_images"     # <-- change output directory
NUM_JOINTS = 18

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Load CSV
# =========================
df = pd.read_csv(INPUT_CSV)
t = df["frame_index"].values

# =========================
# Detect joints
# =========================
joint_names = [
    "Nose", "Neck",
    "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist",
    "RHip", "RKnee", "RAnkle",
    "LHip", "LKnee", "LAnkle",
    "REye", "LEye", "REar", "LEar"
]

joint_names.sort()

assert len(joint_names) == NUM_JOINTS

# =========================
# Helper: column plot
# =========================
def draw_column(value_suffix, ylabel, title, out_name):
    fig, axes = plt.subplots(
        NUM_JOINTS, 1,
        figsize=(18, 0.9 * NUM_JOINTS),  # 控制「密度」的關鍵
        sharex=True,
        gridspec_kw={"hspace": 0.15}     # 壓縮垂直間距
    )

    for i, joint in enumerate(joint_names):
        ax = axes[i]
        ax.plot(t, df[f"{joint}_{value_suffix}"], linewidth=1)
        ax.set_ylabel(joint, fontsize=9, rotation=0, labelpad=40)
        ax.grid(True, axis="y", alpha=0.3)

        # 不讓 y 軸太擠
        ax.tick_params(axis="y", labelsize=8)

    # 只在最下面顯示 X label
    axes[-1].set_xlabel("Frame", fontsize=11)

    fig.suptitle(title, fontsize=18)
    fig.tight_layout(rect=[0.05, 0.02, 1, 0.97])

    out_path = os.path.join(OUTPUT_DIR, out_name)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Saved: {out_path}")

# =========================
# X-T
# =========================
draw_column(
    value_suffix="x",
    ylabel="X",
    title="X-T Plots (One Joint Per Row)",
    out_name="1217_3_18-x.png"
)

# =========================
# Y-T
# =========================
draw_column(
    value_suffix="y",
    ylabel="Y",
    title="Y-T Plots (One Joint Per Row)",
    out_name="1217_3_18-y.png"
)

print("Done.")