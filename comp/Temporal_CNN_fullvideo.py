import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import glob
import sys
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

# ==============================
# Dataset
# ==============================
class SkeletonDataset(Dataset):
    def __init__(self, csv_list, label_list):
        self.samples = []
        self.labels = []

        for csv, label in zip(csv_list, label_list):
            df = pd.read_csv(csv)
            df = df.fillna(0)
            df = df.select_dtypes(include=["float64","float32","int64","int32"])

            if len(df) == 0:
                continue

            self.samples.append(torch.tensor(df.values, dtype=torch.float32))
            self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


# ==============================
# Collate
# ==============================
def collate_fn(batch):
    xs, ys = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs])
    xs = pad_sequence(xs, batch_first=True)
    ys = torch.tensor(ys)
    return xs, lengths, ys


# ==============================
# Temporal CNN
# ==============================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation):
        super().__init__()
        k = 3
        pad = (k - 1) // 2 * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x))) + self.skip(x)


class TemporalCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(input_dim, 64, 1),
            ConvBlock(64, 128, 2),
            ConvBlock(128, 256, 4),
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)   # (B,T,D) -> (B,D,T)
        z = self.net(x).mean(dim=2)
        return self.fc(z), z


# ==============================
# Train / Eval
# ==============================
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, _, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += x.size(0)
    return correct / max(total, 1)


def train(model, train_loader, test_loader, epochs, device):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(epochs):
        model.train()
        for x, _, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits, _ = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

        acc = evaluate(model, test_loader, device)
        print(f"Epoch {ep+1}/{epochs} | Test Acc = {acc:.4f}")


# ==============================
# Main
# ==============================
if __name__ == "__main__":

    root_train = r"C:\mydata\sf\open\output_csv\1230_full"
    root_test  = r"C:\mydata\sf\open\output_csv\1230One"

    batch_size = 4
    epochs = 50
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- classes from test ----------
    class_names = sorted([
        d for d in os.listdir(root_test)
        if os.path.isdir(os.path.join(root_test, d))
    ])

    if len(class_names) == 0:
        print("❌ test 資料夾沒有任何人名")
        sys.exit(1)

    class2idx = {c:i for i,c in enumerate(class_names)}

    # ---------- load train ----------
    train_csvs = glob.glob(os.path.join(root_train, "*.csv"))
    train_samples, train_labels = [], []

    for f in train_csvs:
        fname = os.path.basename(f)

        # 1229_A.csv -> A
        try:
            label = fname.split("_")[1].replace(".csv", "")
        except:
            continue

        if label not in class2idx:
            print("⚠️ unknown label:", label)
            continue

        train_samples.append(f)
        train_labels.append(class2idx[label])

    if len(train_samples) == 0:
        print("❌ 沒有任何訓練 CSV")
        sys.exit(1)

    # ---------- load test ----------
    test_samples, test_labels = [], []
    for person in class_names:
        csvs = glob.glob(os.path.join(root_test, person, "*.csv"))
        test_samples += csvs
        test_labels  += [class2idx[person]] * len(csvs)

    # ---------- datasets ----------
    train_ds = SkeletonDataset(train_samples, train_labels)
    test_ds  = SkeletonDataset(test_samples, test_labels)

    print("Train samples:", len(train_ds))
    print("Test  samples:", len(test_ds))

    train_loader = DataLoader(train_ds, batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds, batch_size, shuffle=False, collate_fn=collate_fn)

    # ---------- model ----------
    sample_x, _ = train_ds[0]
    model = TemporalCNN(sample_x.shape[1], len(class_names))

    # ---------- train ----------
    train(model, train_loader, test_loader, epochs, device)
