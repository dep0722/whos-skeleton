"""
深度學習步態週期自動檢測系統

方案：使用 LSTM 或 Temporal CNN 來識別週期端點
訓練資料：已手動標註好的週期 CSV

核心思想：
1. 輸入：一個滑動視窗的骨架特徵（位置 + 速度 + 加速度）
2. 輸出：當前 frame 是否為週期端點（0 或 1）
3. 模型學習：什麼樣的模式代表「腳面對鏡頭且踩在地上」
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import cv2

# =========================
# 配置
# =========================
class Config:
    # 路徑
    LABELED_DIR = r"C:\mydata\sf\open\output_csv\0128cut\A\A_1"  # 已切好的週期
    RAW_CSV = r"C:\mydata\sf\open\output_csv\0128\A\0128_A_1.csv"  # 原始完整 CSV
    VIDEO_PATH = r"C:\mydata\sf\open\walking_video\0128\0128_A_1.mp4"  # 影片（可選）
    MODEL_PATH = r"C:\mydata\sf\comp\model\0129_test.pth"
    
    # 特徵
    KEYPOINTS = ['Ankle', 'Knee', 'Hip', 'MidHip']  # 使用的關鍵點
    SIDES = ['L', 'R']
    
    # 模型參數
    WINDOW_SIZE = 30  # 使用前後 30 frames 的資訊
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    DROPOUT = 0.3
    
    # 訓練參數
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 50
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# =========================
# 資料處理
# =========================
def parse_filename(filename):
    """解析檔名：way_n_k.csv"""
    name = os.path.splitext(os.path.basename(filename))[0]
    parts = name.split('_')
    if len(parts) >= 3:
        return {
            'side': parts[0],
            'direction': int(parts[1]),
            'cycle': int(parts[2])
        }
    return None

def build_feature_columns():
    """構建特徵欄位名稱"""
    cols = []
    for side in Config.SIDES:
        for kp in Config.KEYPOINTS:
            cols.append(f"{side}{kp}_x")
            cols.append(f"{side}{kp}_y")
    return cols

def extract_features(df, feature_cols):
    """
    提取特徵：位置 + 速度 + 加速度
    """
    # 位置特徵
    positions = df[feature_cols].values
    
    # 速度特徵
    velocities = np.diff(positions, axis=0, prepend=positions[0:1])
    
    # 加速度特徵
    accelerations = np.diff(velocities, axis=0, prepend=velocities[0:1])
    
    # 組合特徵
    features = np.hstack([positions, velocities, accelerations])
    
    return features

def find_best_match(full_sequence, query_sequence, max_search_range=1000):
    """
    在完整序列中尋找查詢序列的最佳匹配位置
    使用歐氏距離
    """
    query_len = len(query_sequence)
    min_dist = float('inf')
    best_pos = None
    
    # 限制搜尋範圍以提高效率
    search_range = min(len(full_sequence) - query_len, max_search_range)
    
    for i in range(0, search_range, 5):  # 每 5 幀檢查一次
        segment = full_sequence[i:i+query_len]
        
        # 計算歐氏距離
        dist = np.sum((segment - query_sequence) ** 2)
        
        if dist < min_dist:
            min_dist = dist
            best_pos = i
    
    return best_pos

def reconstruct_labels_from_segments(raw_csv_path, labeled_dir):
    """
    根據已切好的週期，重建原始 CSV 的標籤
    
    策略：
    1. 讀取原始 CSV
    2. 找出該檔案對應的所有切好的週期
    3. 推算每個週期在原始檔案中的位置
    4. 標記端點為 1，其他為 0
    """
    # 讀取原始 CSV
    print(f"讀取原始 CSV: {raw_csv_path}")
    df = pd.read_csv(raw_csv_path)
    labels = np.zeros(len(df), dtype=np.int32)
    
    # 檢查 labeled_dir 是否存在
    if not os.path.exists(labeled_dir):
        print(f"警告: 找不到標註資料夾 {labeled_dir}")
        return df, labels
    
    # 讀取所有週期檔案
    segment_files = sorted(glob.glob(os.path.join(labeled_dir, "*.csv")))
    
    if len(segment_files) == 0:
        print(f"警告: {labeled_dir} 沒有週期檔案")
        return df, labels
    
    print(f"找到 {len(segment_files)} 個週期檔案")
    
    # 使用特徵匹配來找出每個週期在原始檔案中的位置
    feature_cols = build_feature_columns()
    
    # 檢查所需的欄位是否存在
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"警告: 原始 CSV 缺少以下欄位: {missing_cols}")
        # 嘗試自動補全缺少的欄位
        for col in missing_cols:
            df[col] = 0.0
    
    raw_features = df[feature_cols].values
    
    endpoints = []
    
    for seg_file in segment_files:
        info = parse_filename(seg_file)
        if info is None:
            continue
        
        print(f"  處理週期: {os.path.basename(seg_file)}")
        
        seg_df = pd.read_csv(seg_file)
        
        # 檢查週期檔案的欄位
        missing_seg_cols = [col for col in feature_cols if col not in seg_df.columns]
        if missing_seg_cols:
            print(f"    警告: 週期檔案缺少欄位: {missing_seg_cols}")
            for col in missing_seg_cols:
                seg_df[col] = 0.0
        
        seg_features = seg_df[feature_cols].values
        
        # 在原始資料中尋找最佳匹配位置（使用前 10 幀）
        match_len = min(10, len(seg_features))
        start_pos = find_best_match(raw_features, seg_features[:match_len])
        
        if start_pos is not None:
            end_pos = start_pos + len(seg_df)
            endpoints.append(start_pos)
            endpoints.append(end_pos - 1)  # 最後一幀
            print(f"    → 找到位置: {start_pos} - {end_pos}")
    
    # 標記端點
    print(f"\n共找到 {len(endpoints)} 個端點")
    for ep in endpoints:
        if 0 <= ep < len(labels):
            # 端點附近 ±2 幀都標記為 1
            start_idx = max(0, ep - 2)
            end_idx = min(len(labels), ep + 3)
            labels[start_idx:end_idx] = 1
    
    print(f"標記的正樣本數: {labels.sum()}")
    
    return df, labels

def prepare_training_data(raw_csv_path, labeled_dir):
    """
    準備訓練資料（單個 CSV 檔案版本）
    """
    print(f"\n處理原始 CSV: {raw_csv_path}")
    
    # 檢查檔案是否存在
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"找不到原始 CSV 檔案: {raw_csv_path}")
    
    # 重建標籤
    df, labels = reconstruct_labels_from_segments(raw_csv_path, labeled_dir)
    
    if labels.sum() == 0:
        raise ValueError("沒有找到任何標記的端點！請檢查 LABELED_DIR 路徑是否正確。")
    
    # 提取特徵
    feature_cols = build_feature_columns()
    features = extract_features(df, feature_cols)
    
    print(f"\n特徵矩陣: {features.shape}")
    print(f"端點數量: {labels.sum()}")
    print(f"正樣本比例: {labels.sum() / len(labels):.3%}")
    
    return features, labels

# =========================
# 資料集
# =========================
class GaitDataset(Dataset):
    def __init__(self, features, labels, window_size=30, scaler=None):
        self.features = features
        self.labels = labels
        self.window_size = window_size
        
        # 標準化
        if scaler is None:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(features)
        else:
            self.scaler = scaler
            self.features = self.scaler.transform(features)
    
    def __len__(self):
        return max(0, len(self.features) - self.window_size)
    
    def __getitem__(self, idx):
        # 取出一個視窗
        x = self.features[idx:idx+self.window_size]
        y = self.labels[idx+self.window_size//2]  # 預測中心點
        
        return torch.FloatTensor(x), torch.FloatTensor([y])

# =========================
# 模型
# =========================
class GaitDetectorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        
        # 取最後一個時間步
        last_out = lstm_out[:, -1, :]
        
        # 分類
        out = self.fc(last_out)
        
        return out

class GaitDetectorCNN(nn.Module):
    """使用 1D CNN 的替代方案"""
    def __init__(self, input_size, window_size=30):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.transpose(1, 2)  # → (batch, features, seq_len)
        
        x = self.conv(x)
        x = x.squeeze(-1)  # → (batch, 256)
        
        out = self.fc(x)
        
        return out

# =========================
# 訓練
# =========================
def train_model(model, train_loader, val_loader, epochs=50, device='cpu'):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # 訓練
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 驗證
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()
                
                pred_label = (pred > 0.5).float()
                correct += (pred_label == y).sum().item()
                total += y.size(0)
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc = correct / total
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.3f}")
        
        # 儲存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # 確保目錄存在
            model_dir = os.path.dirname(Config.MODEL_PATH)
            if model_dir and not os.path.exists(model_dir):
                os.makedirs(model_dir)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, Config.MODEL_PATH)
            print("  → 儲存最佳模型")
    
    return history

# =========================
# 主程式
# =========================
def main():
    print("=== 深度學習步態週期檢測系統 ===\n")
    
    # 1. 準備資料
    print("準備訓練資料...")
    try:
        X, y = prepare_training_data(Config.RAW_CSV, Config.LABELED_DIR)
    except Exception as e:
        print(f"\n錯誤: {e}")
        print("\n請檢查以下設定:")
        print(f"1. 原始 CSV 路徑: {Config.RAW_CSV}")
        print(f"   - 檔案是否存在? {os.path.exists(Config.RAW_CSV)}")
        print(f"2. 標註資料夾路徑: {Config.LABELED_DIR}")
        print(f"   - 資料夾是否存在? {os.path.exists(Config.LABELED_DIR)}")
        if os.path.exists(Config.LABELED_DIR):
            csv_files = glob.glob(os.path.join(Config.LABELED_DIR, "*.csv"))
            print(f"   - 包含 {len(csv_files)} 個 CSV 檔案")
        return
    
    # 2. 分割資料
    print(f"\n分割訓練/驗證資料...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"訓練集: {len(X_train)} 樣本, 正樣本: {y_train.sum()}")
    print(f"驗證集: {len(X_val)} 樣本, 正樣本: {y_val.sum()}")
    
    # 3. 創建資料集
    train_dataset = GaitDataset(X_train, y_train, Config.WINDOW_SIZE)
    val_dataset = GaitDataset(X_val, y_val, Config.WINDOW_SIZE, scaler=train_dataset.scaler)
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print(f"\n錯誤: 資料集長度不足！")
        print(f"訓練集長度: {len(train_dataset)}")
        print(f"驗證集長度: {len(val_dataset)}")
        print(f"請確保資料長度 > WINDOW_SIZE ({Config.WINDOW_SIZE})")
        return
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)
    
    # 4. 創建模型
    input_size = X.shape[1]
    model = GaitDetectorLSTM(
        input_size=input_size,
        hidden_size=Config.HIDDEN_SIZE,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT
    ).to(Config.DEVICE)
    
    print(f"\n=== 模型資訊 ===")
    print(f"模型: {model.__class__.__name__}")
    print(f"輸入維度: {input_size}")
    print(f"隱藏層大小: {Config.HIDDEN_SIZE}")
    print(f"LSTM 層數: {Config.NUM_LAYERS}")
    print(f"訓練資料: {len(train_dataset)} 樣本")
    print(f"驗證資料: {len(val_dataset)} 樣本")
    print(f"設備: {Config.DEVICE}")
    print(f"批次大小: {Config.BATCH_SIZE}\n")
    
    # 5. 訓練
    print("開始訓練...\n")
    history = train_model(
        model, train_loader, val_loader,
        epochs=Config.EPOCHS,
        device=Config.DEVICE
    )
    
    # 6. 視覺化
    print("\n生成訓練曲線...")
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("訓練曲線已儲存至: training_history.png")
    plt.show()
    
    print(f"\n=== 訓練完成 ===")
    print(f"最佳驗證損失: {min(history['val_loss']):.4f}")
    print(f"最佳驗證準確率: {max(history['val_acc']):.3f}")
    print(f"模型已儲存至: {Config.MODEL_PATH}")

if __name__ == "__main__":
    main()