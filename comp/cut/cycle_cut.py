import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import os

# =========================
# 參數(為你的資料量身訂)
# =========================
CSV_PATH = r"C:\mydata\sf\open\output_csv\1230_full\1229_A.csv"
MAX_FRAME = 500

FLAT_WINDOW = 5          # 看多少 frame 的趨勢
FLAT_SLOPE = 0.1          # 平段:每 frame 前進 < 0.1 px
MIN_FLAT_LEN = 5         # 平段至少多少 frame
RISE_TOTAL = 3.0         # 後面必須明顯前進,才算一個週期

# 儲存設定
SAVE_PATH = r"C:\mydata\sf\open\output_images\cycle_0127.jpg"  # 指定儲存位置
SAVE_DPI = 150           # 圖片解析度

NUM_COLORS = 20
cmap = colormaps["tab20"]

# =========================
# 讀資料
# =========================
df = pd.read_csv(CSV_PATH).iloc[:MAX_FRAME]

ankle_cols = {
    "L": ("LAnkle_x", "LAnkle_y"),
    "R": ("RAnkle_x", "RAnkle_y")
}

# =========================
# 建立整合圖表
# =========================
fig, axes = plt.subplots(
    3, 1, figsize=(20, 12),
    sharex=True,
    gridspec_kw={"height_ratios": [1, 1, 1]}
)

# =========================
# 主流程
# =========================
all_cycles = {}  # 儲存每側的週期資訊

for idx, (side, (col_x, col_y)) in enumerate(ankle_cols.items()):
    x = df[col_x].values
    y = df[col_y].values
    n = len(x)

    # ---------- 1️⃣ 找平段(低斜率) ----------
    is_flat = np.zeros(n, dtype=bool)

    for i in range(n - FLAT_WINDOW):
        slope = (x[i + FLAT_WINDOW] - x[i]) / FLAT_WINDOW
        if abs(slope) < FLAT_SLOPE:
            is_flat[i:i + FLAT_WINDOW] = True

    # ---------- 2️⃣ 合併平段 ----------
    flat_segments = []
    start = None

    for i, v in enumerate(is_flat):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= MIN_FLAT_LEN:
                flat_segments.append((start, i))
            start = None

    if start is not None and n - start >= MIN_FLAT_LEN:
        flat_segments.append((start, n))

    if len(flat_segments) < 2:
        print(f"{side} 平段不足")
        continue

    # ---------- 3️⃣ 平段 + 上升 = 週期 ----------
    cycles = []

    for i in range(len(flat_segments) - 1):
        fs, fe = flat_segments[i]
        ns, _ = flat_segments[i + 1]

        if abs(x[ns] - x[fe]) > RISE_TOTAL:
            cycles.append((fs, ns))

    all_cycles[side] = cycles
    print(f"{side} cycles: {len(cycles)}")

    # ---------- 4️⃣ 畫在對應的子圖 ----------
    ax = axes[idx]
    
    for i, (s, e) in enumerate(cycles):
        ax.plot(
            range(s, e),
            y[s:e],
            color=cmap(i % NUM_COLORS),
            linewidth=1.3,
            alpha=0.9
        )

    ax.set_title(f"{side} ankle – gait cycles (cycles: {len(cycles)})")
    ax.set_ylabel("y position")
    ax.grid(True)

# ---------- 5️⃣ 第三張圖: x 座標 ----------
ax_x = axes[2]

# 畫 L 和 R 的 x 座標
x_L = df["LAnkle_x"].values
x_R = df["RAnkle_x"].values
n = len(x_L)

ax_x.plot(range(n), x_L, color="blue", linewidth=1.2, label="L ankle x", alpha=0.7)
ax_x.plot(range(n), x_R, color="red", linewidth=1.2, label="R ankle x", alpha=0.7)
ax_x.set_title("X position comparison")
ax_x.set_ylabel("x position")
ax_x.set_xlabel("frame index")
ax_x.legend()
ax_x.grid(True)

plt.tight_layout()

# ---------- 6️⃣ 儲存圖片 ----------
# 確保目錄存在
save_dir = os.path.dirname(SAVE_PATH)
if save_dir and not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"建立目錄: {save_dir}")

plt.savefig(SAVE_PATH, dpi=SAVE_DPI, bbox_inches='tight')
print(f"圖片已儲存至: {SAVE_PATH}")

plt.show()