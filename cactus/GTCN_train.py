import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

# ------------------------------
# 計算膝蓋角度
# ------------------------------
def compute_knee_angle(hip, knee, ankle):
    """
    hip/knee/ankle: np.array (T,2)
    回傳角度弧度
    """
    v1 = hip - knee
    v2 = ankle - knee
    dot = np.sum(v1*v2, axis=1)
    norm = np.linalg.norm(v1, axis=1)*np.linalg.norm(v2, axis=1)
    cos_angle = np.clip(dot/norm, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    return angle

# ------------------------------
# Dataset
# ------------------------------
class SkeletonDatasetSplit(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        df = pd.read_csv(path)

        # joints shape (T, num_joints, 3)
        joint_cols = [c for c in df.columns if any(k in c for k in ['_x','_y','_score'])]
        num_joints = len(joint_cols)//3
        joints = df[joint_cols].values.astype(np.float32).reshape(len(df), num_joints, 3)

        # 加膝蓋角度
        angle_dim = np.zeros((len(df), num_joints, 1), dtype=np.float32)
        joints = np.concatenate([joints, angle_dim], axis=2)

        # 左膝 11,12,13 右膝 8,9,10 (OpenPose COCO)
        if num_joints >= 14:
            L_angle = compute_knee_angle(joints[:,11,:2], joints[:,12,:2], joints[:,13,:2])
            R_angle = compute_knee_angle(joints[:,8,:2], joints[:,9,:2], joints[:,10,:2])
            joints[:,12,3] = L_angle
            joints[:,9,3] = R_angle

        # flatten joints (T, num_joints*4)
        x = joints.reshape(len(df), -1)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

# ------------------------------
# Collate function
# ------------------------------
def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded = pad_sequence(sequences, batch_first=True)
    labels = torch.stack(labels)
    return padded, lengths, labels

# ------------------------------
# 半 GTCN 卷積 (Temporal)
# ------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size-1)//2 * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout>0 else nn.Identity()
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch!=out_ch else nn.Identity()

    def forward(self,x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.dropout(out)
        return out + self.shortcut(x)

class TemporalCNNEncoder(nn.Module):
    def __init__(self, input_dim, channels=[64,128,256], kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        in_ch = input_dim
        dilations = [2**i for i in range(len(channels))]
        for out_ch,d in zip(channels, dilations):
            layers.append(ConvBlock(in_ch, out_ch, kernel_size=kernel_size, dilation=d, dropout=dropout))
            in_ch = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, x, lengths=None):
        x = x.permute(0,2,1)  # (B,T,D) -> (B,D,T)
        out = self.net(x)
        z = out.mean(dim=2)
        return z

class CosineClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes, scale=10.0):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, embedding_dim))
        self.scale = scale

    def forward(self,z):
        z = F.normalize(z, dim=1)
        w = F.normalize(self.prototypes, dim=1)
        logits = self.scale * z @ w.t()
        return logits

class TemporalCNNClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, channels=[64,128,256], kernel_size=3, dropout=0.1):
        super().__init__()
        self.encoder = TemporalCNNEncoder(input_dim, channels=channels, kernel_size=kernel_size, dropout=dropout)
        self.classifier = CosineClassifier(channels[-1], num_classes)

    def forward(self, x, lengths=None):
        z = self.encoder(x, lengths)
        logits = self.classifier(z)
        return logits, z

# ------------------------------
# 訓練 / 評估
# ------------------------------
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x,lengths,y in dataloader:
            x = x.to(device)
            y = y.to(device)
            logits,_ = model(x,lengths)
            preds = logits.argmax(dim=1)
            correct += (preds==y).sum().item()
            total += x.size(0)
    return correct/total

def train_model(model, train_loader, test_loader, num_epochs=20, lr=1e-3, device='cpu', save_path=None):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        for x,lengths,y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits,_ = model(x,lengths)
            loss = criterion(logits,y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds==y).sum().item()
            total += x.size(0)
        train_acc = correct/total
        test_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/total:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    if save_path:
        torch.save(model.state_dict(), save_path)
        print("Model saved to", save_path)

# ------------------------------
# Main
# ------------------------------
if __name__=="__main__":
    root_dir = r"C:\mydata\sf\open\output_csv\0129One"
    batch_size = 4
    num_epochs = 50
    device = 'cpu'

    class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir,d))])
    class2idx = {c:i for i,c in enumerate(class_names)}

    # CSV
    all_samples = []
    all_labels = []
    for person in class_names:
        csv_files = sorted(glob.glob(os.path.join(root_dir, person, "*.csv")))
        all_samples += csv_files
        all_labels += [class2idx[person]]*len(csv_files)

    train_samples, test_samples, train_labels, test_labels = train_test_split(
        all_samples, all_labels, test_size=0.2, stratify=all_labels, random_state=42
    )

    train_dataset = SkeletonDatasetSplit(train_samples, train_labels)
    test_dataset = SkeletonDatasetSplit(test_samples, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # 建立模型
    sample_x,_ = train_dataset[0]
    input_dim = sample_x.shape[1]
    num_classes = len(class_names)
    model = TemporalCNNClassifier(input_dim, num_classes)

    # 訓練 + 儲存
    train_model(model, train_loader, test_loader, num_epochs=num_epochs, device=device,
                save_path=r"C:\mydata\sf\comp\model\gtcn_0201.pth")
