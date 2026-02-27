import csv
import math
from pathlib import Path

# =============================
# 使用者設定
# =============================
CSV_ROOT = Path(r"C:\mydata\sf\open\output_csv\1230One\0108_B")
OUTPUT_ROOT = Path(r"C:\mydata\sf\open\output_csv\1230One_avg")

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# =============================
# Helper
# =============================
def mean_ignore_nan(values):
    vals = [v for v in values if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else float("nan")

def process_side(person_dir, side, out_dir):
    csv_files = sorted(person_dir.glob(f"{side}*.csv"))
    if not csv_files:
        print(f"  [SKIP] no {side}*.csv")
        return

    all_rows = []
    fieldnames = None
    max_len = 0

    # 讀所有 CSV
    for csv_path in csv_files:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            if fieldnames is None:
                fieldnames = reader.fieldnames
            elif reader.fieldnames != fieldnames:
                raise ValueError(f"Column mismatch: {csv_path}")

            rows = list(reader)
            all_rows.append(rows)
            max_len = max(max_len, len(rows))

    # frame-wise 平均
    avg_rows = []

    for i in range(max_len):
        out_row = {}

        for col in fieldnames:
            if col == "label":
                out_row[col] = all_rows[0][0][col]
                continue

            vals = []
            for rows in all_rows:
                if i < len(rows):
                    v = rows[i][col]
                    vals.append(float(v) if v != "nan" else float("nan"))
                else:
                    vals.append(float("nan"))

            out_row[col] = str(mean_ignore_nan(vals))

        avg_rows.append(out_row)

    # 輸出
    out_path = out_dir / f"{person_dir.name}_{side}_avg.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(avg_rows)

    print(f"  [OK] {out_path.name}")

# =============================
# 主流程：每個人
# =============================
for person_dir in sorted(CSV_ROOT.iterdir()):
    if not person_dir.is_dir():
        continue

    print(f"[INFO] Processing {person_dir.name}")

    out_person_dir = OUTPUT_ROOT / person_dir.name
    out_person_dir.mkdir(parents=True, exist_ok=True)

    process_side(person_dir, "L", out_person_dir)
    process_side(person_dir, "R", out_person_dir)

print("\n🎉 All persons done (L & R averaged).")
