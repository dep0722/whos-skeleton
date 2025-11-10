'''
跑 KNN 並做不同 K 值的比較，輸出分類報告和 pkl
'''

# pip install pandas scikit-learn joblib
import pandas as pd
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os


# ===== 路徑設定 =====
CSV_PATH = r"C:/mydata/sf/conda/1025_test/test_csv/1104_p2.csv"
WEIGHT_JSON = r"C:/mydata/sf/conda/1025_test/test_json/weight_1101.json"
MODEL_PATH = r"C:/mydata/sf/conda/1025_test/test_model/KNN1104p2_normal.pkl"

# ===== 讀取資料 =====
df = pd.read_csv(CSV_PATH)
X = df.drop(columns=["label", "filename"])
y = df["label"]

# ===== 讀取 JSON 權重 =====
with open(WEIGHT_JSON, "r", encoding="utf-8") as f:
    weight_dict = json.load(f)

# 轉成 numpy array 對應欄位，取每個特徵的 weight
weights = np.array([weight_dict.get(col, {}).get("weight", 1.0) for col in X.columns], dtype=float)

# ===== 標準化 =====
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===== 套用權重 =====
# KNN 距離計算時，每個特徵乘 sqrt(weight)
# X_weighted = X_scaled * np.sqrt(weights)
X_weighted = X_scaled

# ===== 分割訓練/測試集 =====
X_train, X_test, y_train, y_test = train_test_split(
    X_weighted, y, test_size=0.3, random_state=42, stratify=y
)

# ===== 自動找最佳 K 值 =====
k_values = [1, 3, 5, 7, 9, 11, 13, 15]
best_k = None
best_score = 0

for k in k_values:
    knn_cv = KNeighborsClassifier(n_neighbors=k, metric='euclidean', n_jobs=-1)
    scores = cross_val_score(knn_cv, X_weighted, y, cv=5)  # 5-fold 交叉驗證
    mean_score = scores.mean()
    print(f"K={k} --> 平均準確率: {mean_score:.4f}")
    if mean_score > best_score:
        best_score = mean_score
        best_k = k

print(f"\n最佳 K 值: {best_k}, 對應平均準確率: {best_score:.4f}")

# ===== 使用最佳 K 值訓練模型 =====
knn = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean', n_jobs=-1)
knn.fit(X_train, y_train)

# ===== 預測與報告 =====
y_pred = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n測試集整體準確率: {acc:.4f}")
print("\n分類報告:")
print(classification_report(y_test, y_pred))

# ===== 儲存模型 =====
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(knn, MODEL_PATH)
print(f"\nKNN 模型已儲存至: {MODEL_PATH}")