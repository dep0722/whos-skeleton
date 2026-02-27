"""
GTCN 測試程式 - 版本3：更大 Batch Size
生成混淆矩陣、分類報告、各項指標圖表
pip install torch numpy pandas matplotlib seaborn scikit-learn scipy
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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support


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
            
            if len(x) != self.t_resample:
                old_x = np.linspace(0, 1, len(x))
                new_x = np.linspace(0, 1, self.t_resample)
                fx = interp1d(old_x, x, kind='linear')
                fy = interp1d(old_x, y, kind='linear')
                x = fx(new_x)
                y = fy(new_x)
            
            node_list.append(np.stack([x, y], axis=0))

        X = np.stack(node_list, axis=0).astype(np.float32)

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

        X = torch.tensor(X, dtype=torch.float32).permute(1, 2, 0)
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


# ==================== 測試函數 ====================
def evaluate(model, loader, edge_index, device='cpu'):
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X, E, y in loader:
            X = X.to(device)
            E = E.to(device)
            y = y.to(device)
            
            logits = model(X, E, edge_index)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix (GTCN-V3: Large Batch Size)', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_metrics(y_true, y_pred, class_names, save_path):
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label='Precision', color='#3498db')
    ax.bar(x, recall, width, label='Recall', color='#2ecc71')
    ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c')
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Classification Metrics by Class (GTCN-V3)', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_support(y_true, class_names, save_path):
    unique, counts = np.unique(y_true, return_counts=True)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(class_names)), counts, color='#9b59b6', edgecolor='black')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title('Sample Distribution (GTCN-V3)', fontsize=16, fontweight='bold')
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ==================== 主程式 ====================
if __name__ == "__main__":
    # ===== 配置 =====
    MODEL_PATH = r"C:\mydata\sf\comp\model\GTCN\gtcn_v3_0213_5min.pth"
    TEST_DATA_DIR = r"C:\mydata\sf\open\output_csv\0211_test"
    OUTPUT_DIR = r"C:\mydata\sf\open\output_images\model_result\gtcn_v3_0213"
    
    BATCH_SIZE = 16  # 測試時也使用較大的batch size（與訓練一致）
    DEVICE = 'cpu'
    # ================
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("GTCN 版本3 測試：更大 Batch Size (16)")
    print("=" * 60)
    
    # 載入模型
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    class_names = checkpoint['class_names']
    class2idx = checkpoint['class2idx']
    print(f"Classes: {class_names}")
    
    # 建立測試資料集
    test_dataset = GaitDataset(TEST_DATA_DIR, t_resample=40, class2idx=class2idx)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Test samples: {len(test_dataset)}")
    
    # 建立模型並載入權重
    edge_index = [(0, 1), (1, 2), (3, 4), (4, 5)]
    model = GTCN_V3(num_classes=len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 評估
    print("\nEvaluating...")
    y_pred, y_true = evaluate(model, test_loader, edge_index, device=DEVICE)
    
    # 計算整體準確率
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    # 輸出分類報告
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    # 繪製圖表
    print("\nGenerating plots...")
    plot_confusion_matrix(y_true, y_pred, class_names, 
                         os.path.join(OUTPUT_DIR, "confusion_matrix_gtcn_v3.png"))
    plot_metrics(y_true, y_pred, class_names, 
                os.path.join(OUTPUT_DIR, "metrics_by_class_gtcn_v3.png"))
    plot_support(y_true, class_names, 
                os.path.join(OUTPUT_DIR, "sample_distribution_gtcn_v3.png"))
    
    # 儲存數值結果
    results_txt = os.path.join(OUTPUT_DIR, "results_gtcn_v3.txt")
    with open(results_txt, 'w', encoding='utf-8') as f:
        f.write("GTCN 版本3：更大 Batch Size (16)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    print(f"Saved: {results_txt}")
    
    print("\nDone!")
