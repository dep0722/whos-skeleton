"""
TCNN 訓練程式 - Cosine 分類器 + Temporal Pyramid Pooling
段落數自適應（levels 可自行調整）
含訓練過程損失函數可視化
pip install torch numpy pandas matplotlib scipy
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


# ==================== 資料處理 ====================
class GaitDataset(Dataset):
    def __init__(self, root_dir, t_resample=40, class2idx=None):
        self.samples = []
        self.labels = []
        self.t_resample = t_resample
        self.class2idx = class2idx
        self.joints = ["RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle"]

        for cls in os.listdir(root_dir):
            cls_path = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_path):
                continue
            csv_files = self._collect_csv_recursive(cls_path)
            if len(csv_files) == 0:
                continue
            self.samples.extend(csv_files)
            if class2idx is not None:
                self.labels.extend([class2idx[cls]] * len(csv_files))
            else:
                self.labels.extend([cls] * len(csv_files))

    def _collect_csv_recursive(self, directory):
        csv_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
        return sorted(csv_files)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        csv_path = self.samples[idx]
        label = self.labels[idx]
        df = pd.read_csv(csv_path)

        features = []
        for joint in self.joints:
            for coord in ['x', 'y']:
                col = f"{joint}_{coord}"
                if col not in df.columns:
                    raise ValueError(f"CSV 缺少必要欄位: {col}")
                seq = df[col].values
                if len(seq) != self.t_resample:
                    old_x = np.linspace(0, 1, len(seq))
                    new_x = np.linspace(0, 1, self.t_resample)
                    f = interp1d(old_x, seq, kind='linear')
                    seq = f(new_x)
                features.append(seq)

        X = np.stack(features, axis=1).astype(np.float32)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)
        return X, y


# ==================== 模型定義 ====================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.dropout(out)
        return out + self.shortcut(x)


class TemporalPyramidPooling(nn.Module):
    """
    時間金字塔池化：將時間軸分成多個尺度的段落，各自做 GAP 後拼接。
    levels=[1, 2, 4] 表示分 1 段、2 段、4 段，總輸出維度 = channels * sum(levels)
    levels 可自由調整，模型會自動計算輸出維度。
    """
    def __init__(self, levels=[1, 2, 4]):
        super().__init__()
        self.levels = levels

    def forward(self, x):
        # x: (B, C, T)
        B, C, T = x.shape
        pooled = []
        for n in self.levels:
            # 用 adaptive_avg_pool1d 把時間軸縮成 n 段
            out = F.adaptive_avg_pool1d(x, n)  # (B, C, n)
            out = out.view(B, -1)              # (B, C*n)
            pooled.append(out)
        return torch.cat(pooled, dim=1)        # (B, C * sum(levels))


class TemporalCNNEncoder(nn.Module):
    def __init__(self, input_dim, channels=[64, 128, 256], kernel_size=3,
                 dropout=0.1, pyramid_levels=[1, 2, 4]):
        super().__init__()
        layers = []
        in_ch = input_dim
        dilations = [2**i for i in range(len(channels))]
        for out_ch, d in zip(channels, dilations):
            layers.append(ConvBlock(in_ch, out_ch, kernel_size=kernel_size, dilation=d, dropout=dropout))
            in_ch = out_ch
        self.net = nn.Sequential(*layers)

        self.pyramid = TemporalPyramidPooling(levels=pyramid_levels)
        pyramid_out_dim = channels[-1] * sum(pyramid_levels)
        self.proj = nn.Linear(pyramid_out_dim, channels[-1])

    def forward(self, x):
        x = x.permute(0, 2, 1)        # (B,T,D) -> (B,D,T)
        out = self.net(x)              # (B, 256, T)
        z = self.pyramid(out)          # (B, 256 * sum(levels))
        z = self.proj(z)               # (B, 256)
        return z


class CosineClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes, scale=10.0):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, embedding_dim))
        self.scale = scale

    def forward(self, z):
        z = F.normalize(z, dim=1)
        w = F.normalize(self.prototypes, dim=1)
        logits = self.scale * z @ w.t()
        return logits


class TCNN(nn.Module):
    def __init__(self, input_dim, num_classes, channels=[64, 128, 256], kernel_size=3,
                 dropout=0.1, pyramid_levels=[1, 2, 4]):
        super().__init__()
        self.encoder = TemporalCNNEncoder(input_dim, channels=channels, kernel_size=kernel_size,
                                          dropout=dropout, pyramid_levels=pyramid_levels)
        self.classifier = CosineClassifier(channels[-1], num_classes)

    def forward(self, x):
        z = self.encoder(x)
        logits = self.classifier(z)
        return logits


# ==================== 訓練函數 ====================
def train(model, loader, optimizer, criterion, device='cpu', epochs=10, save_dir='./'):
    model.to(device)
    train_losses = []
    train_accs = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            B = X.size(0)

            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += B

        avg_loss = total_loss / total
        accuracy = correct / total
        train_losses.append(avg_loss)
        train_accs.append(accuracy)
        print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | Acc: {accuracy:.4f}")

    plot_training_curves(train_losses, train_accs, save_dir, 'cosine_pyramid')
    return train_losses, train_accs


def plot_training_curves(losses, accs, save_dir, model_name):
    epochs = range(1, len(losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, losses, 'b-', linewidth=2, label='Training Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'Training Loss - {model_name}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, accs, 'g-', linewidth=2, label='Training Accuracy')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title(f'Training Accuracy - {model_name}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim([0, 1.05])

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'training_curves_{model_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved: {save_path}")


# ==================== 主程式 ====================
if __name__ == "__main__":
    DATA_DIR = r"D:\Coding\CSV\0219new\train"
    SAVE_DIR = r"D:\Coding\TCNN\cos\pyramid\results"
    MODEL_PATH = r"D:\Coding\TCNN\cos\pyramid\tcnn_cosine_pyramid.pth"

    BATCH_SIZE = 4
    EPOCHS = 50
    LEARNING_RATE = 0.001
    DEVICE = 'cpu'
    PYRAMID_LEVELS = [1, 2, 4]   # ← 可自行調整段落數

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    class2idx = {c: i for i, c in enumerate(class_names)}
    print(f"Classes: {class_names}")

    dataset = GaitDataset(DATA_DIR, t_resample=40, class2idx=class2idx)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Samples: {len(dataset)}")

    input_dim = 12
    model = TCNN(input_dim, num_classes=len(class_names), pyramid_levels=PYRAMID_LEVELS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print(f"\nModel: Cosine + Pyramid {PYRAMID_LEVELS}")
    print(f"Input dim: {input_dim}")

    train_losses, train_accs = train(model, loader, optimizer, criterion,
                                     device=DEVICE, epochs=EPOCHS, save_dir=SAVE_DIR)

    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'class2idx': class2idx,
        'input_dim': input_dim,
        'pyramid_levels': PYRAMID_LEVELS,
        'train_losses': train_losses,
        'train_accs': train_accs
    }, MODEL_PATH)
    print(f"Model saved: {MODEL_PATH}")
