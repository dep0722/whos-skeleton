'''
可以直接丟整部影片下去的 lstm
'''
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

# =========================
# 設定
# =========================
DATA_ROOT = r"C:\mydata\sf\open\output_csv\1230_full"
BATCH_SIZE = 4
HIDDEN_SIZE = 64
NUM_LAYERS = 1
EPOCHS = 40
LR = 1e-3

JOINTS = ["Ankle", "Knee", "Hip"]

# =========================
# 讀 CSV 並取左右腳多關節 y 座標
# =========================
def load_full_sequence(csv_path: Path):
    df = pd.read_csv(csv_path)

    if "label" not in df.columns:
        raise ValueError(f"No 'label' column in {csv_path.name}")
    label = df["label"].iloc[0]

    feats = []
    for side in ["L", "R"]:
        for joint in JOINTS:
            col = f"{side}{joint}_y"
            if col not in df.columns:
                raise ValueError(f"Missing {col}")
            seq = np.nan_to_num(df[col].to_numpy())
            feats.append(seq.reshape(-1, 1))

    X = np.hstack(feats)  # shape = (T, 6)
    return X, label

# =========================
# Dataset
# =========================
class GaitDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.samples[idx], dtype=torch.float32),
            self.labels[idx],
            len(self.samples[idx])
        )

def collate_fn(batch):
    xs, ys, lens = zip(*batch)
    xs_pad = pad_sequence(xs, batch_first=True)
    return xs_pad, torch.tensor(ys), torch.tensor(lens)

# =========================
# LSTM 模型
# =========================
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(
            x, lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        _, (hn, _) = self.lstm(packed)
        out = hn[-1]
        return self.fc(out)

# =========================
# 主程式
# =========================
if __name__ == "__main__":
    # 讀資料
    X_list, y_list = [], []
    for csv in Path(DATA_ROOT).rglob("*.csv"):
        try:
            x, y = load_full_sequence(csv)
            X_list.append(x)
            y_list.append(y)
        except Exception as e:
            print(f"[SKIP] {csv.name}: {e}")

    print(f"Loaded {len(X_list)} samples")

    if len(X_list) == 0:
        raise ValueError("No valid samples loaded. Check CSV and column names!")

    # Label encode
    le = LabelEncoder()
    y_enc = le.fit_transform(y_list)

    # Train/test split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_list, y_enc, test_size=0.3,
        stratify=y_enc, random_state=42
    )

    # 標準化（用 train 統計）
    scaler = StandardScaler()
    all_train = np.vstack(X_tr)
    scaler.fit(all_train)

    X_tr = [scaler.transform(x) for x in X_tr]
    X_te = [scaler.transform(x) for x in X_te]

    # Dataset / Loader
    train_ds = GaitDataset(X_tr, y_tr)
    test_ds = GaitDataset(X_te, y_te)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE,
        shuffle=False, collate_fn=collate_fn
    )

    # Model
    model = LSTMClassifier(
        input_size=X_tr[0].shape[1],
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=len(le.classes_)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # =========================
    # Training
    # =========================
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x, y, lens in train_loader:
            optimizer.zero_grad()
            out = model(x, lens)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss={total_loss/len(train_loader):.4f}")

    # =========================
    # Testing
    # =========================
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y, lens in test_loader:
            out = model(x, lens)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    print(f"Test Accuracy: {correct/total*100:.2f}%")
