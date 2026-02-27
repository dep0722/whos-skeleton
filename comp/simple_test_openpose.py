"""
測試腳本 - 支援 OpenPose CSV 格式
載入訓練好的模型並對測試資料進行預測
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


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
    """步態資料集 - 支援 OpenPose CSV 格式"""
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
            csv_files = sorted(glob.glob(os.path.join(cls_path, "*.csv")))
            self.samples.extend(csv_files)
            if class2idx is not None:
                self.labels.extend([class2idx[cls]] * len(csv_files))
            else:
                self.labels.extend([cls] * len(csv_files))

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
            
            if len(x) != self.t_resample:
                t_orig = np.linspace(0, 1, len(x))
                t_new = np.linspace(0, 1, self.t_resample)
                x = np.interp(t_new, t_orig, x)
                y = np.interp(t_new, t_orig, y)
            
            node_list.append(np.stack([x, y], axis=0))

        X = np.stack(node_list, axis=0).astype(np.float32)

        # 計算角度（修正：加上 squeeze）
        right_angle = compute_knee_angle(X[0].T, X[1].T, X[2].T).squeeze()  # (T,)
        left_angle = compute_knee_angle(X[3].T, X[4].T, X[5].T).squeeze()   # (T,)
        edge_attr = np.stack([right_angle, right_angle, left_angle, left_angle], axis=1).astype(np.float32)  # (T, 4)

        X = torch.tensor(X, dtype=torch.float32).permute(1, 2, 0)  # (C,T,V)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)  # (T, E)
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
        node_out = self.self_loop.out_features
        
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


class GTCN(nn.Module):
    """圖時間卷積網路"""
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


# ==================== 測試函數 ====================
def test(model, loader, edge_index, class_names, device='cpu'):
    """測試模型"""
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("\n開始測試...")
    print("=" * 60)
    
    with torch.no_grad():
        for X, E, y in loader:
            X = X.to(device)
            E = E.to(device)
            y = y.to(device)
            
            logits = model(X, E, edge_index)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # 計算準確率
    accuracy = (all_preds == all_labels).mean()
    
    # 計算每個類別的準確率
    class_acc = {}
    for i, cls in enumerate(class_names):
        mask = all_labels == i
        if mask.sum() > 0:
            class_acc[cls] = (all_preds[mask] == all_labels[mask]).mean()
    
    # 打印結果
    print(f"\n整體準確率: {accuracy:.4f} ({(all_preds == all_labels).sum()}/{len(all_labels)})")
    print("\n各類別準確率:")
    for cls, acc in class_acc.items():
        count = (all_labels == class_names.index(cls)).sum()
        print(f"  {cls}: {acc:.4f} ({count} 個樣本)")
    
    # 混淆矩陣
    print("\n混淆矩陣:")
    confusion = np.zeros((len(class_names), len(class_names)), dtype=int)
    for true, pred in zip(all_labels, all_preds):
        confusion[true, pred] += 1
    
    header = "實際\\預測"
    print(f"{header:<12}", end="")
    for cls in class_names:
        print(f"{cls:>8}", end="")
    print()
    
    for i, cls in enumerate(class_names):
        print(f"{cls:<12}", end="")
        for j in range(len(class_names)):
            print(f"{confusion[i, j]:>8}", end="")
        print()
    
    print("=" * 60)
    
    return accuracy, all_preds, all_labels, all_probs


# ==================== 主程式 ====================
if __name__ == "__main__":
    # ===== 配置區域 - 在這裡修改你的設定 =====
    MODEL_PATH = r"C:\mydata\sf\comp\model\gtcn_model_0211.pth"  # 訓練好的模型路徑
    TEST_DIR = r"C:\mydata\sf\open\output_csv\0211_test"  # 測試資料路徑（與訓練資料格式相同）
    
    BATCH_SIZE = 4
    DEVICE = 'cuda'  # 改成 'cuda' 如果有 GPU
    # ========================================

    print("\n" + "=" * 60)
    print("GTCN 步態分類測試 (OpenPose CSV 格式)")
    print("=" * 60)
    
    # 載入模型
    print(f"\n載入模型: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    class_names = checkpoint['class_names']
    class2idx = checkpoint['class2idx']
    
    print(f"類別數量: {len(class_names)}")
    print(f"類別名稱: {class_names}")
    
    # 建立模型
    edge_index = [(0, 1), (1, 2), (3, 4), (4, 5)]
    model = GTCN(num_classes=len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 載入測試資料
    print(f"\n載入測試資料: {TEST_DIR}")
    try:
        test_dataset = GaitDataset(TEST_DIR, t_resample=40, class2idx=class2idx)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        print(f"✓ 成功載入 {len(test_dataset)} 個測試樣本")
    except ValueError as e:
        print(f"\n❌ 錯誤: {e}")
        exit(1)
    
    # 測試
    accuracy, preds, labels, probs = test(model, test_loader, edge_index, class_names, device=DEVICE)
    
    print(f"\n✓ 測試完成！")