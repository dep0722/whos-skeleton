import csv
from pathlib import Path
import matplotlib.pyplot as plt

# =============================
# Path configuration
# =============================
CSV_ROOT = Path(r"C:\mydata\sf\open\output_csv\1230One")
FIG_OUTPUT_ROOT = Path(r"C:\mydata\sf\open\output_images\1230\LAnkle")

FIG_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

LEFT_COL  = "LAnkle_y"
RIGHT_COL = "RAnkle_y"

# =============================
# Process each class folder
# =============================
for class_dir in sorted(CSV_ROOT.iterdir()):
    if not class_dir.is_dir():
        continue

    label = class_dir.name
    csv_files = sorted(class_dir.glob("L*.csv"))

    if not csv_files:
        print(f"[SKIP] {label}: no L*.csv")
        continue

    print(f"[INFO] Plotting class: {label}")

    # 所有子圖共用 x 軸
    fig, axes = plt.subplots(
        nrows=len(csv_files),
        ncols=1,
        figsize=(10, 3 * len(csv_files)),
        sharex=True
    )

    if len(csv_files) == 1:
        axes = [axes]

    # -----------------------------
    # Read & Plot
    # -----------------------------
    for ax, csv_path in zip(axes, csv_files):
        left_vals  = []
        right_vals = []

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for col in (LEFT_COL, RIGHT_COL):
                if col not in reader.fieldnames:
                    raise ValueError(f"{csv_path} missing {col}")

            for row in reader:
                lv = row[LEFT_COL]
                rv = row[RIGHT_COL]

                left_vals.append(float(lv) if lv != "nan" else float("nan"))
                right_vals.append(float(rv) if rv != "nan" else float("nan"))

        # 左右腳畫在同一個 subplot
        ax.plot(left_vals,  label="Left Ankle",  linestyle="-")
        ax.plot(right_vals, label="Right Ankle", linestyle="--")

        ax.set_ylabel("Ankle y")
        ax.set_title(csv_path.stem, fontsize=10)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Frame index")

    fig.suptitle(
        f"{label} – LHeel_y & RHeel_y (per CSV, shared x-axis)",
        fontsize=14
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = FIG_OUTPUT_ROOT / f"{label}_sharedX_0106.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"[OK] Saved figure → {out_path}")

print("\n🎉 All classes done (per CSV subplot, L/R together).")
