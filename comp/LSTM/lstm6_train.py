"""
LSTM 訓練程式 - 6 節點版本（雙側）
使用雙側 RHip, RKnee, RAnkle, LHip, LKnee, LAnkle 的 x, y 座標
pip install torch numpy pandas scikit-learn scipy
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d


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
        
        # 提取雙側關節的 x, y 座標
        features = []
        for joint in self.joints:
            for coord in ['x', 'y']:
                col = f"{joint}_{coord}"
                if col not in df.columns:
                    raise ValueError(f"CSV 缺少必要欄位: {col}\n檔案: {csv_path}")
                
                seq = df[col].values
                
                # 重採樣到固定長度
                if len(seq) != self.t_resample:
                    old_x = np.linspace(0, 1, len(seq))
                    new_x = np.linspace(0, 1, self.t_resample)
                    f = interp1d(old_x, seq, kind='linear')
                    seq = f(new_x)
                
                features.append(seq)
        
        # 組合成 (T, 12) - 6關節 × 2座標
        X = np.stack(features, axis=1).astype(np.float32)  # (T, 12)
        
        # 標準化
        X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
        
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)
        return X, y


# ==================== 模型定義 ====================
class LSTMClassifier(nn.Module):
    """LSTM 分類器"""
    def __init__(self, input_size, hidden_size=64, num_layers=1, num_classes=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x: (B, T, input_size)
        out, (h_n, c_n) = self.lstm(x)
        # 取最後一個時間步
        out = out[:, -1, :]  # (B, hidden_size)
        out = self.dropout(out)
        out = self.fc(out)   # (B, num_classes)
        return out


# ==================== 訓練函數 ====================
def train(model, loader, optimizer, criterion, device='cpu', epochs=10):
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
        print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | Acc: {accuracy:.4f}")


# ==================== 主程式 ====================
if __name__ == "__main__":
    # ===== 配置 =====
    DATA_DIR = r"C:\mydata\sf\open\output_csv\0219_train"
    SAVE_PATH = r"C:\mydata\sf\comp\model\LSTM\lstm_6node_model_0213.pth"
    BATCH_SIZE = 8
    EPOCHS = 50
    LEARNING_RATE = 0.001
    HIDDEN_SIZE = 64
    NUM_LAYERS = 1
    DEVICE = 'cuda'
    # ================

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
    input_size = 12  # 6 關節 × 2 座標
    model = LSTMClassifier(input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, 
                          num_classes=len(class_names))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 訓練
    train(model, loader, optimizer, criterion, device=DEVICE, epochs=EPOCHS)

    # 保存模型
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'class2idx': class2idx,
        'input_size': input_size,
        'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS
    }, SAVE_PATH)
    print(f"Saved: {SAVE_PATH}")
