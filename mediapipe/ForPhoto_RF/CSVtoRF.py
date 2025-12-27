'''
讀取 CSV 然後輸出成 .pkl 模型檔 ( 隨機森林 )
'''
# pip install pandas scikit-learn joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# === 路徑設定 ===
CSV_PATH = r"C:/mydata/sf/conda/1025_test/test_csv/1104_p2.csv"
MODEL_PATH = r"C:/mydata/sf/conda/1025_test/test_model/RFmodel_1104p2_normal.pkl"

# === 讀取資料 ===
df = pd.read_csv(CSV_PATH)

# 去除無用欄位（filename）
X = df.drop(columns=["label", "filename"])
y = df["label"]

# === 分割訓練/測試集 ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# === 建立並訓練模型 ===
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# === 模型評估 ===
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"準確率: {acc:.4f}")
print("\n分類報告:")
print(classification_report(y_test, y_pred))

# === 儲存模型 ===
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(rf, MODEL_PATH)
print(f"\n模型已儲存至: {MODEL_PATH}")
