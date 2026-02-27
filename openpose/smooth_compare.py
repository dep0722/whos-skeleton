'''
同時輸入 JSON 及 CSV，輸出各四個平滑化處理之後的檔案
'''

import os
import json
import csv
import numpy as np

# =====================================================
# 使用者設定（你只需要改這裡）
# =====================================================

INPUT_JSON = r"C:\mydata\sf\open\output_json\1217_3.json"
INPUT_CSV  = r"C:\mydata\sf\open\output_csv\1217_3.csv"

JSON_ROOT = r"C:\mydata\sf\open\output_json"
CSV_ROOT  = r"C:\mydata\sf\open\output_csv"

OUTPUT_PREFIX = "0102_3_smooth_2"
LABEL = "3"

NUM_JOINTS = 25
FPS = 30

MA_WINDOW = 3
EMA_ALPHA = 0.3
CONF_EMA_BASE_ALPHA = 0.3

ONEEURO_MIN_CUTOFF = 1.0
ONEEURO_BETA = 0.01
ONEEURO_D_CUTOFF = 1.0

# =====================================================
# 讀 JSON（只當 template）
# =====================================================

def load_openpose_json_template(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "frames" in data:
        return data
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("Unknown OpenPose JSON format")

# =====================================================
# 讀 CSV（實際數據來源）
# =====================================================

def load_openpose_csv(path):
    FRAME_OFFSET = 1
    raw_rows = []
    header = None

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue

            if header is None:
                try:
                    float(row[0])
                except ValueError:
                    header = row
                    continue

            raw_rows.append(row)

    if not raw_rows:
        raise ValueError("CSV has no data rows")

    T = len(raw_rows)
    xs = np.zeros((NUM_JOINTS, T))
    ys = np.zeros((NUM_JOINTS, T))
    cs = np.zeros((NUM_JOINTS, T))

    for t, row in enumerate(raw_rows):
        for j in range(NUM_JOINTS):
            base = FRAME_OFFSET + 3*j
            xs[j, t] = float(row[base])
            ys[j, t] = float(row[base + 1])
            cs[j, t] = float(row[base + 2])

    # ✅ debug 只印一次
    print("frame[0:5]:", [raw_rows[i][0] for i in range(5)])
    print("x0 raw[0:5]:", xs[0, :5])

    return xs, ys, cs, header, raw_rows



# =====================================================
# 平滑方法
# =====================================================

def median_filter_1d(x, k=3):
    assert k % 2 == 1
    r = k // 2
    y = x.copy()
    for i in range(r, len(x) - r):
        y[i] = np.median(x[i-r:i+r+1])
    return y


def moving_average(x, k):
    kernel = np.ones(2*k + 1) / (2*k + 1)
    return np.convolve(x, kernel, mode="same")

def ema(x, alpha):
    y = np.zeros_like(x)
    y[0] = x[0]
    for t in range(1, len(x)):
        y[t] = alpha * x[t] + (1 - alpha) * y[t-1]
    return y

def conf_ema(x, c, base_alpha, min_alpha=0.05):
    y = np.zeros_like(x)
    y[0] = x[0]
    for t in range(1, len(x)):
        # 將 confidence clamp 到 [0,1]
        conf = np.clip(c[t], 0.0, 1.0)
        # alpha 有下限
        alpha = min_alpha + conf * (base_alpha - min_alpha)
        y[t] = alpha * x[t] + (1 - alpha) * y[t-1]
    return y

    
class OneEuroFilter:
    def __init__(self, freq, min_cutoff, beta, d_cutoff):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0

    def _alpha(self, cutoff):
        tau = 1.0 / (2 * np.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def filter(self, x):
        if self.x_prev is None:
            self.x_prev = x
            return x

        dx = (x - self.x_prev) * self.freq
        a_d = self._alpha(self.d_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff)

        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat

# =====================================================
# 套用平滑
# =====================================================

def apply_smoothing(xs, ys, cs, method):
    J, T = xs.shape
    xs_s = np.zeros_like(xs)
    ys_s = np.zeros_like(ys)

    for j in range(J):
        # ✅ 先去除 1~2 frame spike（這是你現在最需要的）
        x_clean = median_filter_1d(xs[j], k=3)
        y_clean = median_filter_1d(ys[j], k=3)

        if method == "MA":
            xs_s[j] = moving_average(x_clean, MA_WINDOW)
            ys_s[j] = moving_average(y_clean, MA_WINDOW)

        elif method == "EMA":
            xs_s[j] = ema(x_clean, EMA_ALPHA)
            ys_s[j] = ema(y_clean, EMA_ALPHA)

        elif method == "CONF_EMA":
            xs_s[j] = conf_ema(x_clean, cs[j], CONF_EMA_BASE_ALPHA)
            ys_s[j] = conf_ema(y_clean, cs[j], CONF_EMA_BASE_ALPHA)

        elif method == "ONE_EURO":
            fx = OneEuroFilter(FPS, ONEEURO_MIN_CUTOFF, ONEEURO_BETA, ONEEURO_D_CUTOFF)
            fy = OneEuroFilter(FPS, ONEEURO_MIN_CUTOFF, ONEEURO_BETA, ONEEURO_D_CUTOFF)
            for t in range(T):
                xs_s[j, t] = fx.filter(x_clean[t])
                ys_s[j, t] = fy.filter(y_clean[t])

    return xs_s, ys_s


# =====================================================
# 輸出
# =====================================================

def write_json(template, xs, ys, cs, out_path):
    data = json.loads(json.dumps(template))

    frames = data["frames"] if isinstance(data, dict) else data

    for t, frame in enumerate(frames):
        if "people" not in frame or not frame["people"]:
            continue
        kp = frame["people"][0]["pose_keypoints_2d"]
        for j in range(NUM_JOINTS):
            kp[3*j]   = float(xs[j, t])
            kp[3*j+1] = float(ys[j, t])
            kp[3*j+2] = float(cs[j, t])

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def write_csv(xs, ys, cs, header, raw_rows, out_path, label):
    T = xs.shape[1]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # ✅ 原樣寫回 header
        if header is not None:
            out_header = header.copy()
            if "label" not in out_header:
                out_header.append("label")
            writer.writerow(out_header)

        for t in range(T):
            row = raw_rows[t].copy()

            FRAME_OFFSET = 1

            for j in range(NUM_JOINTS):
                base = FRAME_OFFSET + 3*j
                row[base]     = f"{xs[j, t]:.6f}"
                row[base + 1] = f"{ys[j, t]:.6f}"
                row[base + 2] = f"{cs[j, t]:.6f}"

            # label 欄位處理
            if header is not None and "label" in header:
                idx = header.index("label")
                row[idx] = label
            else:
                row.append(label)

            writer.writerow(row)


# =====================================================
# 主流程
# =====================================================

def main():
    template = load_openpose_json_template(INPUT_JSON)
    xs, ys, cs, header, raw_rows = load_openpose_csv(INPUT_CSV)


    methods = ["MA", "EMA", "CONF_EMA", "ONE_EURO"]

    for method in methods:
        json_dir = os.path.join(JSON_ROOT, method)
        csv_dir  = os.path.join(CSV_ROOT, method)
        os.makedirs(json_dir, exist_ok=True)
        os.makedirs(csv_dir, exist_ok=True)

        xs_s, ys_s = apply_smoothing(xs, ys, cs, method)

        write_json(
            template,
            xs_s, ys_s, cs,
            os.path.join(json_dir, f"{OUTPUT_PREFIX}_{method}.json")
        )

        write_csv(
            xs_s, ys_s, cs,
            header, raw_rows,
            os.path.join(csv_dir, f"{OUTPUT_PREFIX}_{method}.csv"),
            LABEL
        )


        print(f"[OK] {method}")

    print("All methods finished.")

if __name__ == "__main__":
    main()
