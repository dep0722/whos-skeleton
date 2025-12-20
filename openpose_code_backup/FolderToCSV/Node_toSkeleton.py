"""
keypoint_edges_to_csv.py

讀入一個 OpenPose 輸出的 CSV（欄位為 x0,y0,c0,x1,y1,c1,...,x24,y24,c24,label）
計算指定關節對的歐氏距離（畢氏定理），輸出成一個新的 CSV：
每行一張圖，欄位為 3-4,3-2,2-1,1-5,5-6,6-7,11-10,10-9,9-8,8-12,12-13,13-14,label

規則：
- 若某端節點缺失（x==0 and y==0），該距離輸出為空字串（''）
- 輸出浮點數保留 4 位小數
"""
import csv
import math
import pandas as pd
import numpy as np

# ====== 你的輸入節點 CSV ======
input_csv = r"C:/mydata/sf/openpose/output_csv/1212_0.7ver_test.csv"

# ====== 輸出的骨架邊長 CSV ======
output_csv = r"C:/mydata/sf/openpose/output_csv/1212_0.7ver_test_nor.csv"

# ====== BODY_25 關節連接 ======
edges = [
    (3, 4), (3, 2), (2, 1), (1, 5),
    (5, 6), (6, 7), (11, 10), (10, 9),
    (9, 8), (8, 12), (12, 13), (13, 14)
]

edge_names = [f"{a}-{b}" for a, b in edges]


def safe_float(v):
    """將 None、NaN、空字串轉為 0"""
    if v is None:
        return 0.0
    if v == "":
        return 0.0
    try:
        v = float(v)
        if np.isnan(v):
            return 0.0
        return v
    except:
        return 0.0


def distance_or_empty(x1, y1, x2, y2):
    """符合你的規則：任一點 (0,0) → '' """
    if (x1 == 0 and y1 == 0) or (x2 == 0 and y2 == 0):
        return ""
    return math.hypot(x1 - x2, y1 - y2)


print("讀取 CSV…")
df = pd.read_csv(input_csv)

with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)

    # 標題
    writer.writerow(edge_names + ["label"])

    for _, row in df.iterrows():
        row_out = []

        for a, b in edges:
            ax = safe_float(row[f"x{a}"])
            ay = safe_float(row[f"y{a}"])
            bx = safe_float(row[f"x{b}"])
            by = safe_float(row[f"y{b}"])

            d = distance_or_empty(ax, ay, bx, by)
            row_out.append(d)

        row_out.append(row["label"])
        writer.writerow(row_out)

print("完成！輸出：", output_csv)
