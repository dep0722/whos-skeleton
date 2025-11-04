import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# ===== 路徑設定 =====
CSV_PATH = r"C:/mydata/sf/conda/1025_test/test_csv/pose_1104.csv"     # 你的資料
MODEL_PATH = r"C:/mydata/sf/conda/1025_test/DL/DLmodemlp_model.pkl"     # 模型輸出路徑

# ===== 1. 讀取資料 =====
df = pd.read_csv(CSV_PATH)
print("原始資料筆數：", len(df))

# ===== 2. 分離特徵與標籤 =====
X = df.iloc[:, :-2].values  # 前 12 欄為特徵
y = df["label"].values

# ===== 3. Label 編碼 (A→0, B→1...) =====
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ===== 4. 特徵標準化 =====
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===== 5. 分割訓練/測試資料 =====
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ===== 6. 建立 MLP 模型 =====
# hidden_layer_sizes=(64,32)：兩層隱藏層（64與32神經元）
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
)

# ===== 7. 開始訓練 =====
print("訓練中...")
mlp.fit(X_train, y_train)

# ===== 8. 測試模型 =====
y_pred = mlp.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"測試集準確率：{acc:.4f}")
print("\n詳細報告：")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ===== 9. 保存模型與 scaler、label encoder =====
joblib.dump({
    "model": mlp,
    "scaler": scaler,
    "label_encoder": le
}, MODEL_PATH)

print(f"\n✅ 模型已保存至: {MODEL_PATH}")
