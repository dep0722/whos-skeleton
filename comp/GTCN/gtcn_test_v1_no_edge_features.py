"""
GTCN 測試程式 - 版本1：不使用邊特徵
生成混淆矩陣、分類報告、各項指標圖表
pip install torch numpy pandas matplotlib seaborn scikit-learn
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support


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
                t_orig = np.linspace(0, 1, len(x))
                t_new = np.linspace(0, 1, self.t_resample)
                x = np.interp(t_new, t_orig, x)
                y = np.interp(t_new, t_orig, y)
            
            node_list.append(np.stack([x, y], axis=0))

        X = np.stack(node_list, axis=0).astype(np.float32)
        X = torch.tensor(X, dtype=torch.float32).permute(1, 2, 0)
        y = torch.tensor(label, dtype=torch.long)
        
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
        B, V, F_dim = X.shape
        H = self.self_loop(X)
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
            x_t = X[:, :, t, :].permute(0, 2, 1)
            H_t = self.gcn1(x_t, edge_index)
            H_t = self.gcn2(H_t, edge_index)
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


# ==================== 測試函數 ====================
def evaluate(model, loader, edge_index, device='cpu'):
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            
            logits = model(X, edge_index)
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
    plt.title('Confusion Matrix (GTCN-V1: No Edge Features)', fontsize=16, fontweight='bold')
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
    ax.set_title('Classification Metrics by Class (GTCN-V1)', fontsize=16, fontweight='bold')
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
    plt.title('Sample Distribution (GTCN-V1)', fontsize=16, fontweight='bold')
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ==================== 主程式 ====================
if __name__ == "__main__":
    # ===== 配置 =====
    MODEL_PATH = r"C:\mydata\sf\comp\model\GTCN\gtcn_v1_0213_5min.pth"
    TEST_DATA_DIR = r"C:\mydata\sf\open\output_csv\0211_test"
    OUTPUT_DIR = r"C:\mydata\sf\open\output_images\model_result\GTCN_V1_ResultsPicture"
    
    BATCH_SIZE = 8
    DEVICE = 'cuda'
    # ================
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("GTCN 版本1 測試：不使用邊特徵")
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
    model = GTCN_V1(num_classes=len(class_names))
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
                         os.path.join(OUTPUT_DIR, "confusion_matrix_gtcn_v1.png"))
    plot_metrics(y_true, y_pred, class_names, 
                os.path.join(OUTPUT_DIR, "metrics_by_class_gtcn_v1.png"))
    plot_support(y_true, class_names, 
                os.path.join(OUTPUT_DIR, "sample_distribution_gtcn_v1.png"))
    
    # 儲存數值結果
    results_txt = os.path.join(OUTPUT_DIR, "results_gtcn_v1.txt")
    with open(results_txt, 'w', encoding='utf-8') as f:
        f.write("GTCN 版本1：不使用邊特徵\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    print(f"Saved: {results_txt}")
    
    print("\nDone!")
