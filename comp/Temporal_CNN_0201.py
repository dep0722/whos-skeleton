import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import glob
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import math

# ============================================================
# Dataset with joint coordinates + knee angle features
# ============================================================
def compute_angle(a, b, c):
    """
    計算三個點 a-b-c 的夾角 (以弧度為單位)
    b 是角點
    a,b,c: (x,y)
    """
    ba = a - b
    bc = c - b
    cosine_angle = (ba*bc).sum(dim=-1) / (torch.norm(ba, dim=-1)*torch.norm(bc, dim=-1)+1e-6)
    angle = torch.acos(torch.clamp(cosine_angle, -1.0, 1.0))
    return angle  # radians

class SkeletonDataset(Dataset):
    SIDE_JOINTS = {
        "R": ["RHip","RKnee","RAnkle"],
        "L": ["LHip","LKnee","LAnkle"]
    }

    def __init__(self, root_dir, class2idx, extra_joints=None):
        self.samples=[]
        self.labels=[]
        self.extra_joints = extra_joints if extra_joints else []

        for cls in class2idx.keys():
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            # 直接抓類別資料夾下的 CSV
            csv_files = glob.glob(os.path.join(cls_dir, "*.csv"))
            for csv in csv_files:
                self.samples.append(csv)
                self.labels.append(class2idx[cls])


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        filename = os.path.basename(path)
        df = pd.read_csv(path)

        side = "R" if "R" in filename else "L" if "L" in filename else None
        if side is None:
            raise ValueError(f"檔名找不到 R 或 L: {filename}")

        joint_cols = self.SIDE_JOINTS[side] + self.extra_joints

        selected_cols=[]
        for joint in joint_cols:
            for suffix in ["_x","_y"]:
                col_name = f"{joint}{suffix}"
                if col_name in df.columns:
                    selected_cols.append(col_name)

        df = df[selected_cols].fillna(0).select_dtypes(include=['float64','float32','int64','int32'])
        x = torch.tensor(df.values, dtype=torch.float32)

        # 計算膝關節角度
        hip_idx = selected_cols.index(f"{self.SIDE_JOINTS[side][0]}_x")
        knee_idx = selected_cols.index(f"{self.SIDE_JOINTS[side][1]}_x")
        ankle_idx = selected_cols.index(f"{self.SIDE_JOINTS[side][2]}_x")

        hip = x[:, hip_idx:hip_idx+2]
        knee = x[:, knee_idx:knee_idx+2]
        ankle = x[:, ankle_idx:ankle_idx+2]

        angle = compute_angle(hip, knee, ankle).unsqueeze(1)
        x = torch.cat([x, angle], dim=1)

        # 標準化
        x = (x - x.mean(dim=0)) / (x.std(dim=0)+1e-6)

        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

# ============================================================
# Collate
# ============================================================
def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True)
    labels = torch.stack(labels)
    return padded, lengths, labels

# ============================================================
# Temporal CNN
# ============================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, dilation=1, dropout=0.3):
        super().__init__()
        padding = (kernel_size-1)//2 * dilation
        self.conv = nn.Conv1d(in_ch,out_ch,kernel_size,padding=padding,dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.shortcut = nn.Conv1d(in_ch,out_ch,1) if in_ch!=out_ch else nn.Identity()

    def forward(self,x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.dropout(out)
        res = self.shortcut(x)
        min_len = min(out.size(2), res.size(2))
        return out[:,:,:min_len] + res[:,:,:min_len]

class TemporalCNNEncoder(nn.Module):
    def __init__(self,input_dim,channels=[16,32,64,64],kernel_size=5,dropout=0.3):
        super().__init__()
        layers=[]
        in_ch=input_dim
        dilations=[2**i for i in range(len(channels))]
        for out_ch,d in zip(channels,dilations):
            layers.append(ConvBlock(in_ch,out_ch,kernel_size=d*1 if d>1 else kernel_size,dilation=d,dropout=dropout))
            in_ch=out_ch
        self.net=nn.Sequential(*layers)

    def forward(self,x,lengths=None):
        x = x.permute(0,2,1)
        out = self.net(x)
        if lengths is not None:
            lengths = lengths.to(out.device)
            mask = torch.arange(out.size(2), device=out.device).unsqueeze(0) < lengths.unsqueeze(1)
            mask = mask.unsqueeze(1)
            masked_out = out * mask
            mean_pool = masked_out.sum(dim=2)/lengths.unsqueeze(1)
            max_pool = masked_out.masked_fill(mask==0,float('-inf')).max(dim=2)[0]
            z = torch.cat([mean_pool,max_pool],dim=1)
        else:
            mean_pool = out.mean(dim=2)
            max_pool = out.max(dim=2)[0]
            z = torch.cat([mean_pool,max_pool],dim=1)
        return z

class TemporalCNNClassifier(nn.Module):
    def __init__(self,input_dim,num_classes,channels=[32,64,128,128],kernel_size=5,dropout=0.3):
        super().__init__()
        self.encoder=TemporalCNNEncoder(input_dim,channels,kernel_size,dropout)
        self.classifier=nn.Linear(channels[-1]*2,num_classes)

    def forward(self,x,lengths=None):
        z=self.encoder(x,lengths)
        logits=self.classifier(z)
        return logits,z

# ============================================================
# Train / Evaluate
# ============================================================
def evaluate(model,dataloader,device):
    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for x,lengths,y in dataloader:
            x=x.to(device,non_blocking=True)
            y=y.to(device,non_blocking=True)
            lengths=lengths.to(device)
            logits,_=model(x,lengths)
            preds=logits.argmax(dim=1)
            correct+=(preds==y).sum().item()
            total+=x.size(0)
    return correct/total

def train_model(model,train_loader,test_loader,num_epochs=20,lr=1e-3,device='cuda'):
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    for epoch in range(num_epochs):
        model.train()
        running_loss=0
        correct=0
        total=0
        for x,lengths,y in train_loader:
            x=x.to(device,non_blocking=True)
            y=y.to(device,non_blocking=True)
            lengths=lengths.to(device)
            optimizer.zero_grad()
            logits,_=model(x,lengths)
            loss=criterion(logits,y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()*x.size(0)
            preds=logits.argmax(dim=1)
            correct+=(preds==y).sum().item()
            total+=x.size(0)
        train_acc = correct/total
        test_acc = evaluate(model,test_loader,device)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss {running_loss/total:.4f} | Train {train_acc:.4f} | Test {test_acc:.4f}")

# ============================================================
# Main
# ============================================================
if __name__=="__main__":
    train_dir = r"C:\mydata\sf\open\output_csv\0129train"
    test_dir  = r"C:\mydata\sf\open\output_csv\0129test"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:",device)

    class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir,d))])
    class2idx = {c:i for i,c in enumerate(class_names)}

    train_dataset = SkeletonDataset(train_dir,class2idx,extra_joints=["Hip","Knee","Ankle"])
    test_dataset  = SkeletonDataset(test_dir,class2idx,extra_joints=["Hip","Knee","Ankle"])

    train_loader = DataLoader(train_dataset,batch_size=16,shuffle=True,collate_fn=collate_fn,num_workers=4,pin_memory=True)
    test_loader  = DataLoader(test_dataset,batch_size=16,shuffle=False,collate_fn=collate_fn,num_workers=4,pin_memory=True)

    sample_x,_ = train_dataset[0]
    input_dim = sample_x.shape[1]
    num_classes = len(class_names)

    model = TemporalCNNClassifier(input_dim,num_classes)

    train_model(model,train_loader,test_loader,num_epochs=20,device=device)
