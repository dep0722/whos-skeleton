"""
TCNN 6節點測試程式
生成混淆矩陣、分類報告、各項指標圖表
pip install torch numpy pandas matplotlib seaborn scikit-learn scipy
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
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support


# ==================== 資料處理 ====================
class GaitDataset(Dataset):
    """步態資料集 - 6 節點雙側版本"""
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
                    raise ValueError(f"CSV 缺少必要欄位: {col}\n檔案: {csv_path}")
                
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


class TemporalCNNEncoder(nn.Module):
    def __init__(self, input_dim, channels=[64, 128, 256], kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        in_ch = input_dim
        dilations = [2**i for i in range(len(channels))]
        for out_ch, d in zip(channels, dilations):
            layers.append(ConvBlock(in_ch, out_ch, kernel_size=kernel_size, dilation=d, dropout=dropout))
            in_ch = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.net(x)
        z = out.mean(dim=2)
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
    def __init__(self, input_dim, num_classes, channels=[64, 128, 256], kernel_size=3, dropout=0.1):
        super().__init__()
        self.encoder = TemporalCNNEncoder(input_dim, channels=channels, kernel_size=kernel_size, dropout=dropout)
        self.classifier = CosineClassifier(channels[-1], num_classes)

    def forward(self, x):
        z = self.encoder(x)
        logits = self.classifier(z)
        return logits


# ==================== 測試函數 ====================
def evaluate(model, loader, device='cpu'):
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            
            logits = model(X)
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
    plt.title('Confusion Matrix (TCNN-6)', fontsize=16, fontweight='bold')
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
    ax.set_title('Classification Metrics by Class (TCNN-6)', fontsize=16, fontweight='bold')
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
    plt.title('Sample Distribution (TCNN-6)', fontsize=16, fontweight='bold')
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ==================== 主程式 ====================
if __name__ == "__main__":
    # ===== 配置 =====
    MODEL_PATH = r"C:\mydata\sf\comp\model\TCNN\tcnn_6node_model_0213.pth"
    TEST_DATA_DIR = r"C:\mydata\sf\open\output_csv\0219_test"
    OUTPUT_DIR = r"C:\mydata\sf\result_p\tcnn\TCNN_6node_ResultsPicture"
    
    BATCH_SIZE = 4
    DEVICE = 'cuda'
    # ================
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 載入模型
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    class_names = checkpoint['class_names']
    class2idx = checkpoint['class2idx']
    input_dim = checkpoint['input_dim']
    print(f"Classes: {class_names}")
    
    # 建立測試資料集
    test_dataset = GaitDataset(TEST_DATA_DIR, t_resample=40, class2idx=class2idx)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Test samples: {len(test_dataset)}")
    
    # 建立模型並載入權重
    model = TCNN(input_dim, num_classes=len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 評估
    print("\nEvaluating...")
    y_pred, y_true = evaluate(model, test_loader, device=DEVICE)
    
    # 計算整體準確率
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    # 輸出分類報告
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    # 繪製圖表
    print("\nGenerating plots...")
    plot_confusion_matrix(y_true, y_pred, class_names, 
                         os.path.join(OUTPUT_DIR, "confusion_matrix_tcnn6.png"))
    plot_metrics(y_true, y_pred, class_names, 
                os.path.join(OUTPUT_DIR, "metrics_by_class_tcnn6.png"))
    plot_support(y_true, class_names, 
                os.path.join(OUTPUT_DIR, "sample_distribution_tcnn6.png"))
    
    # 儲存數值結果
    results_txt = os.path.join(OUTPUT_DIR, "results_tcnn6.txt")
    with open(results_txt, 'w', encoding='utf-8') as f:
        f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    print(f"Saved: {results_txt}")
    
    print("\nDone!")
