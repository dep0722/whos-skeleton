'''
讀 CSV 以指數函數放大作為權重輸出 json
要再凸顯的話把 max_weight 跟 power 改大，但 power 一定要大於 1，max_weight 太大小邊會變成 0
'''
import pandas as pd
import numpy as np
import json

# CSV 路徑
input_csv = r"C:/mydata/sf/conda/1025_test/test_csv/pose_1101.csv"
output_json = r"C:/mydata/sf/conda/1025_test/test_json/test2.json"

# 讀 CSV
df = pd.read_csv(input_csv)

# 取得 feature column（去掉 filename 與 label）
feature_cols = [c for c in df.columns if c not in ["filename", "label"]]

# 計算每個 feature 對不同 label 的平均值
feature_means_per_label = df.groupby("label")[feature_cols].mean()

# 計算 variance（每條 feature 在不同人之間的變異程度）
feature_variances = feature_means_per_label.var(axis=0)

# --- 指數放大差異，做整數權重 ---
power = 1.5  # 可調整，數值越大凸顯差異越明顯
feature_variances_power = feature_variances ** power

# normalize 到 0~1
feature_variances_norm = feature_variances_power / feature_variances_power.max()

# 放大成整數權重，例如最大權重設為 10
max_weight = 20
feature_weights_int = (feature_variances_norm * max_weight).round().astype(int)

# 將 variance 與整數權重同時存入 dict
weight_dict_with_variance = {}
for feat in feature_cols:
    weight_dict_with_variance[feat] = {
        "variance": float(feature_variances[feat]),
        "weight": int(feature_weights_int[feat])
    }

# 存成 JSON
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(weight_dict_with_variance, f, indent=4, ensure_ascii=False)

print(f"完成！已將 variance 與整數權重輸出到: {output_json}")