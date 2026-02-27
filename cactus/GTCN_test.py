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

# ------------------------------
# 計算膝蓋角度
# ------------------------------
def compute_knee_angle(hip, knee, ankle):
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

        joint_cols = [c for c in df.columns if any(k in c for k in ['_x','_y','_score'])]
        num_joints = len(joint_cols)//3
        joints = df[joint_cols].values.astype(np.float32).reshape(len(df), num_joints, 3)

        # 加膝蓋角度
        angle_dim = np.zeros((len(df), num_joints, 1), dtype=np.float32)
        joints = np.concatenate([joints, angle_dim], axis=2)
        if num_joints >= 14:
            L_angle = compute_knee_angle(joints[:,11,:2], joints[:,12,:2], joints[:,13,:2])
            R_angle = compute_knee_angle(joints[:,8,:2], joints[:,9,:2], joints[:,10,:2])
            joints[:,12,3] = L_angle
            joints[:,9,3] = R_angle

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
# GTCN 模型
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
# Main
# ------------------------------
if __name__=="__main__":
    root_dir = r"C:\mydata\sf\open\output_csv\1230One"
    batch_size = 4
    device = 'cpu'

    class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir,d))])
    class2idx = {c:i for i,c in enumerate(class_names)}

    # 所有 CSV
    all_samples = []
    all_labels = []
    for person in class_names:
        csv_files = sorted(glob.glob(os.path.join(root_dir, person, "*.csv")))
        all_samples += csv_files
        all_labels += [class2idx[person]]*len(csv_files)

    # Dataset
    dataset = SkeletonDatasetSplit(all_samples, all_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 建立模型
    sample_x,_ = dataset[0]
    input_dim = sample_x.shape[1]
    num_classes = len(class_names)
    model = TemporalCNNClassifier(input_dim, num_classes)
    model.load_state_dict(torch.load(r"C:\mydata\sf\comp\model\gtcn_0127.pth", map_location=device))
    model.to(device)
    model.eval()

    # 推論
    correct = 0
    total = 0
    results = []
    with torch.no_grad():
        for x,lengths,y in dataloader:
            x = x.to(device)
            y = y.to(device)
            logits,_ = model(x,lengths)
            preds = logits.argmax(dim=1)
            results.extend(zip([all_samples[i] for i in range(total,total+x.size(0))], preds.cpu().numpy(), y.cpu().numpy()))
            correct += (preds==y).sum().item()
            total += x.size(0)

    print(f"Accuracy: {correct/total:.4f}\n")
    print("Sample predictions:")
    for path, pred, label in results:
        print(f"{os.path.basename(path)} -> Pred: {class_names[pred]}, True: {class_names[label]}")
