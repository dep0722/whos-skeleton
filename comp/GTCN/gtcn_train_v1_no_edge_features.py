"""
GTCN 訓練腳本 - 版本1：不使用邊特徵
改進點：移除膝關節角度計算，簡化模型
pip install torch numpy pandas
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ==================== 資料處理 ====================
class GaitDataset(Dataset):
    """步態資料集 - 版本1：不使用邊特徵"""
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
        """遞迴收集所有 CSV 檔案"""
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

        node_list = []
        for joint in self.joints:
            x_col = f"{joint}_x"
            y_col = f"{joint}_y"
            
            if x_col not in df.columns or y_col not in df.columns:
                raise ValueError(f"CSV 缺少必要欄位: {x_col} 或 {y_col}\n檔案: {csv_path}")
            
            x = df[x_col].values
            y = df[y_col].values
            
            # 重採樣到固定長度
            if len(x) != self.t_resample:
                t_orig = np.linspace(0, 1, len(x))
                t_new = np.linspace(0, 1, self.t_resample)
                x = np.interp(t_new, t_orig, x)
                y = np.interp(t_new, t_orig, y)
            
            node_list.append(np.stack([x, y], axis=0))

        X = np.stack(node_list, axis=0).astype(np.float32)  # (6, 2, T)
        X = torch.tensor(X, dtype=torch.float32).permute(1, 2, 0)  # (C,T,V) -> (2,T,6)
        y = torch.tensor(label, dtype=torch.long)
        
        # 不返回邊特徵
        return X, y


# ==================== 模型定義 ====================
class SimpleGCNLayer(nn.Module):
    """簡化的圖卷積層 - 不使用邊特徵"""
    def __init__(self, node_in, node_out):
        super().__init__()
        self.W_node = nn.Linear(node_in, node_out, bias=False)
        self.self_loop = nn.Linear(node_in, node_out, bias=True)
        self.bn = nn.BatchNorm1d(node_out)

    def forward(self, X, edge_index):
        """
        X: (B, V, F_dim) - batch of nodes
        edge_index: list of (src, dst) tuples
        """
        B, V, F_dim = X.shape
        
        # 初始化輸出（自環連接）
        H = self.self_loop(X)  # (B, V, node_out)
        
        # 累加來自鄰居的訊息
        aggregated = torch.zeros_like(H)
        
        for (src, dst) in edge_index:
            node_feat = X[:, src, :]
            msg = self.W_node(node_feat)
            aggregated[:, dst, :] += msg
        
        H = H + aggregated
        H = H.transpose(1, 2)
        H = self.bn(H)
        H = H.transpose(1, 2)
        return F.relu(H)


class SpatialEncoder(nn.Module):
    """空間編碼器 - 不使用邊特徵"""
    def __init__(self, node_dim=2, hidden_dim=64):
        super().__init__()
        self.gcn1 = SimpleGCNLayer(node_dim, hidden_dim)
        self.gcn2 = SimpleGCNLayer(hidden_dim, hidden_dim)

    def forward(self, X, edge_index):
        B, C, T, V = X.shape
        H_out = []
        for t in range(T):
            x_t = X[:, :, t, :].permute(0, 2, 1)  # (B, V, C)
            H_t = self.gcn1(x_t, edge_index)
            H_t = self.gcn2(H_t, edge_index)
            H_out.append(H_t)
        H_out = torch.stack(H_out, dim=1)  # (B, T, V, H_dim)
        return H_out


class TemporalCNN(nn.Module):
    """時間卷積網路"""
    def __init__(self, in_ch, channels=[64, 128], kernel_size=3):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            padding = (kernel_size - 1) // 2 * dilation
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())
            in_ch = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return x.mean(dim=2)


class GTCN_V1(nn.Module):
    """圖時間卷積網路 - 版本1：不使用邊特徵"""
    def __init__(self, num_classes, node_dim=2, hidden_dim=64, joint_num=6):
        super().__init__()
        self.spatial = SpatialEncoder(node_dim=node_dim, hidden_dim=hidden_dim)
        self.temporal = TemporalCNN(in_ch=joint_num * hidden_dim, channels=[64, 128])
        self.fc = nn.Linear(128, num_classes)

    def forward(self, X, edge_index):
        H = self.spatial(X, edge_index)
        B, T, V, H_dim = H.shape
        H = H.permute(0, 2, 3, 1).reshape(B, V * H_dim, T)
        z = self.temporal(H)
        logits = self.fc(z)
        return logits


# ==================== 訓練函數 ====================
def train(model, loader, edge_index, optimizer, criterion, device='cpu', epochs=50):
    """訓練模型"""
    model.to(device)
    
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
            logits = model(X, edge_index)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += B

        avg_loss = total_loss / total
        accuracy = correct / total
        print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | Acc: {accuracy:.4f}")


# ==================== 主程式 ====================
if __name__ == "__main__":
    # ===== 配置 =====
    DATA_DIR = r"C:\mydata\sf\open\output_csv\0211_train"
    SAVE_PATH = r"C:\mydata\sf\comp\model\GTCN\gtcn_v1_0213_5min.pth"
    BATCH_SIZE = 4
    EPOCHS = 50
    LEARNING_RATE = 0.001
    DEVICE = 'cuda'
    # ================

    print("=" * 60)
    print("GTCN 版本1：不使用邊特徵（移除膝關節角度計算）")
    print("=" * 60)

    # 獲取類別
    class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    class2idx = {c: i for i, c in enumerate(class_names)}
    print(f"Classes: {class_names}")

    # 建立資料集
    try:
        dataset = GaitDataset(DATA_DIR, t_resample=40, class2idx=class2idx)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        print(f"Samples: {len(dataset)}")
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)

    # 建立模型
    edge_index = [(0, 1), (1, 2), (3, 4), (4, 5)]  # RHip→RKnee→RAnkle, LHip→LKnee→LAnkle
    model = GTCN_V1(num_classes=len(class_names))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 訓練
    print("\n開始訓練...")
    train(model, loader, edge_index, optimizer, criterion, device=DEVICE, epochs=EPOCHS)

    # 保存模型
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'class2idx': class2idx,
        'version': 'v1_no_edge_features'
    }, SAVE_PATH)
    print(f"\nSaved: {SAVE_PATH}")
