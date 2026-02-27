import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim

# =========================
# Config
# =========================
ROOT_DIR = r"C:\mydata\sf\open\output_csv\1230One"
RESAMPLE_LEN = 40
TEST_RATIO = 0.2
EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-3

JOINTS = ["Ankle", "Knee", "Hip"]   # 用三個關節
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Utils
# =========================
def detect_side_from_filename(csv_path: Path):
    name = csv_path.stem
    if "L" in name:
        return "L"
    elif "R" in name:
        return "R"
    else:
        raise ValueError("Cannot detect L/R")

def resample(seq, target_len):
    x_old = np.linspace(0, 1, len(seq))
    x_new = np.linspace(0, 1, target_len)
    f = interp1d(x_old, seq, kind="linear")
    return f(x_new)

def load_csv(csv_path):
    df = pd.read_csv(csv_path).sort_values("frame_index")

    label = df["label"].iloc[0]
    side = detect_side_from_filename(csv_path)

    features = []
    for j in JOINTS:
        col = f"{side}{j}_y"
        y = df[col].to_numpy()
        mask = (~np.isnan(y)) & (y != 0)
        y = y[mask]

        if len(y) < 10:
            raise ValueError("Too short")

        y = resample(y, RESAMPLE_LEN)
        features.append(y)

    X = np.stack(features, axis=1)  # (T, F)
    return X, label

def load_from_csv_list(csv_list):
    Xs, ys = [], []
    for csv in csv_list:
        try:
            X, y = load_csv(csv)
            Xs.append(X)
            ys.append(y)
        except Exception as e:
            print(f"[SKIP] {csv}: {e}")
    return np.array(Xs), np.array(ys)

# =========================
# 1️⃣ CSV-level split
# =========================
all_csvs = list(Path(ROOT_DIR).rglob("*.csv"))

train_csvs, test_csvs = train_test_split(
    all_csvs,
    test_size=TEST_RATIO,
    random_state=42,
    shuffle=True
)

print(f"Train CSVs: {len(train_csvs)}")
print(f"Test  CSVs: {len(test_csvs)}")

# =========================
# 2️⃣ Load data
# =========================
X_train, y_train = load_from_csv_list(train_csvs)
X_test, y_test = load_from_csv_list(test_csvs)

print(f"Train samples: {len(X_train)}")
print(f"Test  samples: {len(X_test)}")

# =========================
# 3️⃣ Encode labels
# =========================
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# =========================
# 4️⃣ Torch tensors
# =========================
X_train_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
y_train_t = torch.tensor(y_train_enc, dtype=torch.long).to(DEVICE)
y_test_t = torch.tensor(y_test_enc, dtype=torch.long).to(DEVICE)

# =========================
# 5️⃣ Model
# =========================
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

model = LSTMClassifier(
    input_size=len(JOINTS),
    hidden_size=64,
    num_classes=len(le.classes_)
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# =========================
# 6️⃣ Train
# =========================
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

# =========================
# 7️⃣ Evaluate
# =========================
model.eval()
with torch.no_grad():
    preds = torch.argmax(model(X_test_t), dim=1)
    acc = (preds == y_test_t).float().mean().item()

print(f"\nTest Accuracy (CSV-level): {acc*100:.2f}%")
