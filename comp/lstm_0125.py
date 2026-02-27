import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.preprocessing import LabelEncoder

# =========================
# Settings
# =========================
RESAMPLE_LEN = 40   # 每個 gait cycle 微調成 40 幀

# =========================
# Utilities
# =========================
def detect_side_from_filename(csv_path: Path):
    name = csv_path.stem
    if "L" in name:
        return "L"
    elif "R" in name:
        return "R"
    else:
        raise ValueError(f"[ERROR] Cannot detect L/R from filename: {name}")

# =========================
# Load single CSV
# =========================
def load_ankle_y_sequence_with_label(csv_path: str):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # ---------- 抓 label ----------
    if "label" not in df.columns:
        raise ValueError(f"[ERROR] No 'label' column in {csv_path.name}")
    label = df["label"].iloc[0]  # 假設整個 CSV label 都一樣

    # ---------- 決定腳 ----------
    side = detect_side_from_filename(csv_path)
    ankle_y_col = f"{side}Ankle_y"

    # ---------- 取 ankle_y ----------
    df = df.sort_values("frame_index")
    y = df[ankle_y_col].to_numpy()
    valid_mask = (~np.isnan(y)) & (y != 0)
    y = y[valid_mask]

    if len(y) < 10:
        raise ValueError(f"[ERROR] Too few valid frames: {csv_path.name}")

    return y.reshape(-1, 1), label

# =========================
# Resample sequence
# =========================
def resample_sequence(seq: np.ndarray, target_len: int):
    T = len(seq)
    old_x = np.linspace(0, 1, T)
    new_x = np.linspace(0, 1, target_len)
    f = interp1d(old_x, seq[:, 0], kind="linear")
    y_new = f(new_x)
    return y_new.reshape(-1, 1)

# =========================
# Folder pipeline
# =========================
def folder_to_gait_inputs_with_labels(folder_path: str, target_len=RESAMPLE_LEN):
    folder = Path(folder_path)
    gait_list = []
    label_list = []

    for csv_file in sorted(folder.rglob("*.csv")):
        try:
            seq, label = load_ankle_y_sequence_with_label(csv_file)
            seq = resample_sequence(seq, target_len)
            gait_list.append(seq)
            label_list.append(label)
        except Exception as e:
            print(f"[SKIP] {csv_file}: {e}")

    # 轉成 numpy array，方便直接丟 LSTM
    if gait_list:
        X = np.stack(gait_list, axis=0)  # shape = (num_samples, 40, 1)
        y = np.array(label_list)
        return X, y
    else:
        return np.array([]), np.array([])

# =========================
# Example usage
# =========================
if __name__ == "__main__":
    folder_path = r"C:\mydata\sf\open\output_csv\1230One"
    X, y = folder_to_gait_inputs_with_labels(folder_path, target_len=RESAMPLE_LEN)

    print(f"Processed {X.shape[0]} samples")
    if X.shape[0] > 0:
        print("X shape:", X.shape)   # -> (num_samples, 40, 1)
        print("y shape:", y.shape)   # -> (num_samples,)
        print("Example label:", y[0])
        print("Example sequence (前10幀):", X[0,:10,0])

    # -------------------------
    # 下面直接接 PyTorch LSTM 分類
    # -------------------------
    

    # 轉成 PyTorch tensor
    X_tensor = torch.tensor(X, dtype=torch.float32)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 定義 LSTM 模型
    class LSTMClassifier(nn.Module):
        def __init__(self, input_size=1, hidden_size=32, num_layers=1, num_classes=3):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, num_classes)
        
        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]  # 取最後 timestep
            out = self.fc(out)
            return out

    num_classes = len(np.unique(y))
    model = LSTMClassifier(input_size=1, hidden_size=32, num_layers=1, num_classes=num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 訓練 Loop
    num_epochs = 10
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")

    # 測試範例
    with torch.no_grad():
        sample = X_tensor[0].unsqueeze(0)  # (1,40,1)
        output = model(sample)
        pred_class_int = torch.argmax(output, dim=1).item()
        pred_label = le.inverse_transform([pred_class_int])[0]
        print("Example prediction:", pred_label, "Label:", y[0])
