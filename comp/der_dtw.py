import pandas as pd
import numpy as np
from dtw import dtw
import os

# =====================================================
# 讀取單一 CSV 中指定關節資料
# =====================================================
def get_joint_data(file_path, target_joint, axis):
    try:
        df = pd.read_csv(file_path)
        file_name = os.path.basename(file_path).upper()

        side = file_name[0]  # L or R
        if side not in ("L", "R"):
            return None

        col_name = f"{side}{target_joint}_{axis}"
        if col_name not in df.columns:
            return None

        series = df[col_name].astype(float).values

        mask = series > 0
        if not any(mask) or len(series) < 5:
            return None

        series[~mask] = series[mask].mean()

        # === Z-score normalization（保留）===
        std = np.std(series)
        if std > 0:
            series = (series - np.mean(series)) / std

        return series

    except:
        return None


# =====================================================
# Derivative DTW 用的一階導數
# =====================================================
def compute_derivative(series):
    """
    使用 gradient 而非 diff：
    - 長度不變
    - 對雜訊較穩定
    """
    return np.gradient(series)


# =====================================================
# 使用者設定
# =====================================================
PATH_A = r"C:\mydata\sf\open\output_csv\1230One\A"
PATH_B = r"C:\mydata\sf\open\output_csv\1230One\F"

JOINT_A = "Ankle"
JOINT_B = "Ankle"
AXIS    = "y"


# =====================================================
# 核心比對函式（Derivative DTW）
# =====================================================
def compare_folders_with_contrast(path_a, path_b):
    files_a = [f for f in os.listdir(path_a) if f.endswith(".csv")]
    files_b = [f for f in os.listdir(path_b) if f.endswith(".csv")]

    contrast_scores = []

    for fa in files_a:
        for fb in files_b:
            # L/R 對應
            if fa[0].upper() != fb[0].upper():
                continue

            s1 = get_joint_data(
                os.path.join(path_a, fa),
                target_joint=JOINT_A,
                axis=AXIS
            )

            s2 = get_joint_data(
                os.path.join(path_b, fb),
                target_joint=JOINT_B,
                axis=AXIS
            )

            if s1 is not None and s2 is not None:
                # ===== Derivative DTW 核心改動 =====
                ds1 = compute_derivative(s1)
                ds2 = compute_derivative(s2)

                d = dtw(
                    ds1,
                    ds2,
                    keep_internals=False
                ).normalizedDistance

                # 放大差異（原樣保留）
                score = np.exp(-3.0 * (d ** 2))
                contrast_scores.append(score)

    return contrast_scores


# =====================================================
# 主程式
# =====================================================
if not os.path.exists(PATH_A) or not os.path.exists(PATH_B):
    raise FileNotFoundError("❌ 指定的資料夾路徑不存在")

print("====================================================")
print("開始比對指定的兩個資料夾")
print(f"資料夾 A：{PATH_A}")
print(f"資料夾 B：{PATH_B}")
print(f"A 使用關節：{JOINT_A}_{AXIS}")
print(f"B 使用關節：{JOINT_B}_{AXIS}")
print("====================================================")

scores = compare_folders_with_contrast(PATH_A, PATH_B)

if not scores:
    print("❌ 沒有成功的比對資料")
else:
    print("\n✔ 比對完成")
    print(f"✔ 比對筆數：{len(scores)}")
    print(f"✔ 放大後平均相似度：{np.mean(scores):.2%}")

print("====================================================")
