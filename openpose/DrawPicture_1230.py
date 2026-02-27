import csv
from pathlib import Path
import matplotlib.pyplot as plt

# =============================
# Path configuration
# =============================
CSV_ROOT = Path(r"C:\mydata\sf\open\output_csv\1230One")
FIG_OUTPUT_ROOT = Path(r"C:\mydata\sf\open\output_images\1230\LHeel")

FIG_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

TARGET_COL = "LHeel_y"

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

    max_len = 0
    all_series = []

    # -----------------------------
    # Read all CSV first
    # -----------------------------
    for csv_path in csv_files:
        y_vals = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            if TARGET_COL not in reader.fieldnames:
                raise ValueError(f"{csv_path} missing {TARGET_COL}")

            for row in reader:
                v = row[TARGET_COL]
                y_vals.append(float(v) if v != "nan" else float("nan"))

        all_series.append(y_vals)
        max_len = max(max_len, len(y_vals))

    # -----------------------------
    # Plot
    # -----------------------------
    for ax, csv_path, y_vals in zip(axes, csv_files, all_series):
        ax.plot(y_vals)
        ax.set_ylabel("LAnkle_y")
        ax.set_title(csv_path.stem, fontsize=10)

    axes[-1].set_xlabel("Frame index")

    fig.suptitle(
        f"{label} – LAnkle_y (L*.csv, shared x-axis)",
        fontsize=14
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = FIG_OUTPUT_ROOT / f"{label}_{TARGET_COL}_sharedX.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"[OK] Saved figure → {out_path}")

print("\n🎉 All classes done (shared x-axis).")
