"""
GTCN 訓練腳本 - 版本3：更大的 Batch Size
改進點：BATCH_SIZE=16（從4增加到16，提升BatchNorm效果）
pip install torch numpy pandas scipy
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d


# ==================== 資料處理 ====================
def compute_knee_angle(hip, knee, ankle):
    """計算膝蓋角度（改進數值穩定性）"""
    v1 = hip - knee
    v2 = ankle - knee
    dot = np.sum(v1 * v2, axis=1)
    norm = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    norm = np.maximum(norm, 1e-8)
    cos_angle = np.clip(dot / norm, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    return angle[:, None]


class GaitDataset(Dataset):
    """步態資料集 - 版本3：配合大batch size"""
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
            
            # 使用scipy的interp1d進行高品質插值
            if len(x) != self.t_resample:
                old_x = np.linspace(0, 1, len(x))
                new_x = np.linspace(0, 1, self.t_resample)
                fx = interp1d(old_x, x, kind='linear')
                fy = interp1d(old_x, y, kind='linear')
                x = fx(new_x)
                y = fy(new_x)
            
            node_list.append(np.stack([x, y], axis=0))

        X = np.stack(node_list, axis=0).astype(np.float32)  # (6, 2, T)

        # 標準化處理
        for v in range(X.shape[0]):
            for c in range(X.shape[1]):
                seq = X[v, c, :]
                mean = seq.mean()
                std = seq.std()
                X[v, c, :] = (seq - mean) / (std + 1e-8)

        # 計算左右膝關節角度
        right_angle = compute_knee_angle(X[0].T, X[1].T, X[2].T).squeeze()
        left_angle = compute_knee_angle(X[3].T, X[4].T, X[5].T).squeeze()
        edge_attr = np.stack([right_angle, right_angle, left_angle, left_angle], axis=1).astype(np.float32)
        
        # 標準化邊特徵
        edge_attr = (edge_attr - edge_attr.mean(axis=0, keepdims=True)) / (edge_attr.std(axis=0, keepdims=True) + 1e-8)

        X = torch.tensor(X, dtype=torch.float32).permute(1, 2, 0)  # (C,T,V) -> (2,T,6)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)
        return X, edge_attr, y


# ==================== 模型定義 ====================
class EdgeGCNLayer(nn.Module):
    """邊增強圖卷積層"""
    def __init__(self, node_in, edge_in, node_out):
        super().__init__()
        self.W_node = nn.Linear(node_in, node_out, bias=False)
        self.W_edge = nn.Linear(edge_in, node_out, bias=False)
        self.self_loop = nn.Linear(node_in, node_out, bias=True)
        self.bn = nn.BatchNorm1d(node_out)

    def forward(self, X, edge_index, edge_attr):
        B, V, F_dim = X.shape
        
        H = self.self_loop(X)
        aggregated = torch.zeros_like(H)
        
        for e, (src, dst) in enumerate(edge_index):
            node_feat = X[:, src, :]
            edge_feat = edge_attr[:, e:e+1]
            msg = self.W_node(node_feat) + self.W_edge(edge_feat)
            aggregated[:, dst, :] += msg
        
        H = H + aggregated
        H = H.transpose(1, 2)
        H = self.bn(H)
        H = H.transpose(1, 2)
        return F.relu(H)


class SpatialEncoder(nn.Module):
    """空間編碼器"""
    def __init__(self, node_dim=2, edge_dim=1, hidden_dim=64):
        super().__init__()
        self.gcn1 = EdgeGCNLayer(node_dim, edge_dim, hidden_dim)
        self.gcn2 = EdgeGCNLayer(hidden_dim, edge_dim, hidden_dim)

    def forward(self, X, E, edge_index):
        B, C, T, V = X.shape
        H_out = []
        for t in range(T):
            x_t = X[:, :, t, :].permute(0, 2, 1)
            e_t = E[:, t, :]
            H_t = self.gcn1(x_t, edge_index, e_t)
            H_t = self.gcn2(H_t, edge_index, e_t)
            H_out.append(H_t)
        H_out = torch.stack(H_out, dim=1)
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


class GTCN_V3(nn.Module):
    """圖時間卷積網路 - 版本3：配合大batch size優化"""
    def __init__(self, num_classes, node_dim=2, edge_dim=1, hidden_dim=64, joint_num=6):
        super().__init__()
        self.spatial = SpatialEncoder(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim)
        self.temporal = TemporalCNN(in_ch=joint_num * hidden_dim, channels=[64, 128])
        self.fc = nn.Linear(128, num_classes)

    def forward(self, X, E, edge_index):
        H = self.spatial(X, E, edge_index)
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
        
        for X, E, y in loader:
            X = X.to(device)
            E = E.to(device)
            y = y.to(device)
            B = X.size(0)

            optimizer.zero_grad()
            logits = model(X, E, edge_index)
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
    SAVE_PATH = r"C:\mydata\sf\comp\model\GTCN\gtcn_v3_0213_5min.pth"
    BATCH_SIZE = 16  # ===== 關鍵改進：從4增加到16 =====
    EPOCHS = 50
    LEARNING_RATE = 0.001
    DEVICE = 'cuda'
    # ================

    print("=" * 60)
    print("GTCN 版本3：更大的 Batch Size (16)")
    print("改進：提升 BatchNorm 效果，訓練更穩定")
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
        print(f"Batches per epoch: {len(loader)}")
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)

    # 建立模型
    edge_index = [(0, 1), (1, 2), (3, 4), (4, 5)]
    model = GTCN_V3(num_classes=len(class_names))
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
        'version': 'v3_large_batch_size_16'
    }, SAVE_PATH)
    print(f"\nSaved: {SAVE_PATH}")
