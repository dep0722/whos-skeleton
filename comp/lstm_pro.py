'''
使用 Ankle, Knee, Hip 當特徵的 LSTM 分類版
會把每個週期伸縮到 40 個 frame
'''

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# =========================
# 設定
# =========================
RESAMPLE_LEN = 40  # 每個 gait cycle 微調成 40 幀
BATCH_SIZE = 8
HIDDEN_SIZE = 64
NUM_LAYERS = 1
NUM_EPOCHS = 50
LR = 0.001

# -------------------------
# 判斷 L/R 腳
# -------------------------
def detect_side_from_filename(csv_path: Path):
    name = csv_path.stem
    if "L" in name:
        return "L"
    elif "R" in name:
        return "R"
    else:
        raise ValueError(f"[ERROR] Cannot detect L/R from filename: {name}")

# -------------------------
# 讀 CSV 並取多關節 y 座標
# -------------------------
JOINTS = ["Ankle", "Knee", "Hip"]  # 可以擴充更多關節

def load_gait_sequence(csv_path: str):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # label
    if "label" not in df.columns:
        raise ValueError(f"No 'label' column in {csv_path.name}")
    label = df["label"].iloc[0]

    side = detect_side_from_filename(csv_path)

    y_list = []
    for joint in JOINTS:
        col = f"{side}{joint}_y"
        if col not in df.columns:
            raise ValueError(f"{col} not found in {csv_path.name}")
        seq = df[col].to_numpy()
        valid_mask = (~np.isnan(seq)) & (seq != 0)
        seq = seq[valid_mask]
        if len(seq) < 10:
            raise ValueError(f"Too few valid frames: {csv_path.name}")
        y_list.append(seq.reshape(-1, 1))
    
    seq_multi = np.hstack(y_list)  # shape = (T, num_features)
    return seq_multi, label

# -------------------------
# 重採樣到固定長度
# -------------------------
def resample_sequence(seq: np.ndarray, target_len: int):
    T = len(seq)
    old_x = np.linspace(0, 1, T)
    new_x = np.linspace(0, 1, target_len)
    y_new = []
    for i in range(seq.shape[1]):
        f = interp1d(old_x, seq[:, i], kind="linear")
        y_new.append(f(new_x))
    return np.stack(y_new, axis=1)

# -------------------------
# 讀資料夾所有 CSV
# -------------------------
def folder_to_gait_inputs_labels(folder_path: str, target_len=RESAMPLE_LEN):
    folder = Path(folder_path)
    gait_list = []
    label_list = []

    for csv_file in sorted(folder.rglob("*.csv")):
        try:
            seq, label = load_gait_sequence(csv_file)
            seq = resample_sequence(seq, target_len)
            gait_list.append(seq)
            label_list.append(label)
        except Exception as e:
            print(f"[SKIP] {csv_file}: {e}")

    X = np.stack(gait_list, axis=0)  # (num_samples, seq_len, num_features)
    y = np.array(label_list)
    return X, y

# =========================
# PyTorch LSTM 分類模型
# =========================
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 最後 timestep
        out = self.fc(out)
        return out

# =========================
# 主程式
# =========================
if __name__ == "__main__":
    folder_path = r"C:\mydata\sf\open\output_csv\0128cut"
    X, y = folder_to_gait_inputs_labels(folder_path)
    print(f"Processed {X.shape[0]} samples, sequence length={X.shape[1]}, features={X.shape[2]}")

    # -------------------------
    # Label encode
    # -------------------------
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    print("Classes:", le.classes_)

    # -------------------------
    # train/test split
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42
    )

    # -------------------------
    # 標準化（每個 feature independently）
    # -------------------------
    scaler = StandardScaler()
    num_features = X.shape[2]
    for i in range(num_features):
        X_train[:,:,i] = scaler.fit_transform(X_train[:,:,i])
        X_test[:,:,i] = scaler.transform(X_test[:,:,i])

    # -------------------------
    # 轉 tensor
    # -------------------------
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # -------------------------
    # 模型
    # -------------------------
    model = LSTMClassifier(
        input_size=num_features,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=num_classes
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # -------------------------
    # 訓練
    # -------------------------
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

    # -------------------------
    # 測試準確率
    # -------------------------
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            pred = torch.argmax(outputs, dim=1)
            correct += (pred == batch_y).sum().item()
            total += batch_y.size(0)
    acc = correct / total
    print(f"Test Accuracy: {acc*100:.2f}%")

    # -------------------------
    # 每筆測試資料分類結果
    # -------------------------
    with torch.no_grad():
        outputs = model(X_test_tensor)
        preds = torch.argmax(outputs, dim=1).numpy()  # 整數編碼
        true_labels = y_test_tensor.numpy()

        # 將整數轉回原文字 label
        pred_labels = le.inverse_transform(preds)
        true_labels_text = le.inverse_transform(true_labels)

        correct_count = 0
        for i in range(len(pred_labels)):
            is_correct = pred_labels[i] == true_labels_text[i]
            if is_correct:
                correct_count += 1
            print(f"Sample {i}: Predicted={pred_labels[i]}, True={true_labels_text[i]}, Correct={is_correct}")

        print(f"\nTotal correct: {correct_count}/{len(pred_labels)}")
        print(f"Test Accuracy: {correct_count/len(pred_labels)*100:.2f}%")

