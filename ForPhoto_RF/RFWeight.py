# pip install pandas scikit-learn joblib
import pandas as pd
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ===== 路徑設定 =====
CSV_PATH = r"C:/mydata/sf/conda/1025_test/test_csv/pose_1101.csv"
WEIGHT_JSON = r"C:/mydata/sf/conda/1025_test/test_json/test2.json"
MODEL_PATH = r"C:/mydata/sf/conda/1025_test/test_model/RFmodel_1101_derived.pkl"

# ===== 讀取資料 =====
df = pd.read_csv(CSV_PATH)
X = df.drop(columns=["label", "filename"])
y = df["label"]

# ===== 讀取 JSON 權重 =====
with open(WEIGHT_JSON, "r", encoding="utf-8") as f:
    weight_dict = json.load(f)

# ===== 計算髖寬（LEFT_HIP_RIGHT_HIP） =====
if "LEFT_HIP_RIGHT_HIP" not in X.columns:
    raise ValueError("CSV 必須包含 LEFT_HIP_RIGHT_HIP 欄位")
hip_width = X["LEFT_HIP_RIGHT_HIP"]

# ===== 生成衍生特徵 =====
X_derived = X.copy()
for feat in X.columns:
    weight = weight_dict.get(feat, {}).get("weight", 1)
    # 1️⃣ 加權邊長
    X_derived[f"{feat}_weighted"] = X[feat] * weight
    # 2️⃣ 邊長比例
    X_derived[f"{feat}_ratio"] = (X[feat] / hip_width) * weight

# ===== 分割訓練/測試集 =====
X_train, X_test, y_train, y_test = train_test_split(
    X_derived, y, test_size=0.3, random_state=42, stratify=y
)

# ===== 建立並訓練模型 =====
rf = RandomForestClassifier(
    n_estimators=300,
    max_features='sqrt',       # 每棵樹隨機抽 sqrt(特徵數) 
    min_samples_split=5,        # 節點最少 5 個樣本才切
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# ===== 模型評估 =====
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"準確率: {acc:.4f}")
print("\n分類報告:")
print(classification_report(y_test, y_pred))

# ===== 查看特徵重要性 =====
importances = pd.Series(rf.feature_importances_, index=X_derived.columns)
importances = importances.sort_values(ascending=False)
print("\n前 10 個重要特徵:")
print(importances.head(10))

# ===== 儲存模型 =====
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(rf, MODEL_PATH)
print(f"\n模型已儲存至: {MODEL_PATH}")
