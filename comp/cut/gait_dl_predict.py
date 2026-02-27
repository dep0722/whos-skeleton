"""
使用訓練好的深度學習模型來自動檢測步態週期

使用方式：
1. 先用 gait_dl_detector_fixed.py 訓練模型
2. 再用這個腳本對新資料進行自動檢測
"""

# =========================
# 必須在最開頭設定環境變數
# =========================
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
import torch
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# 從訓練腳本導入必要的函數和類別
import sys
import importlib.util

# 動態載入訓練模組
def load_training_module():
    """動態載入訓練模組，避免重複導入"""
    module_path = r"C:\mydata\sf\comp\cut\gait_dl_detector.py"
    
    if not os.path.exists(module_path):
        print(f"警告: 找不到訓練模組 {module_path}")
        return None
    
    spec = importlib.util.spec_from_file_location("gait_dl_detector_fixed", module_path)
    module = importlib.util.module_from_spec(spec)
    
    try:
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"載入訓練模組失敗: {e}")
        return None

# 嘗試載入訓練模組
training_module = load_training_module()

if training_module:
    GaitDetectorLSTM = training_module.GaitDetectorLSTM
    GaitDetectorCNN = training_module.GaitDetectorCNN
    build_feature_columns = training_module.build_feature_columns
    extract_features = training_module.extract_features
    Config = training_module.Config
else:
    # 提供基本的備用定義
    import torch.nn as nn
    
    class Config:
        WINDOW_SIZE = 30
        HIDDEN_SIZE = 128
        NUM_LAYERS = 2
        DROPOUT = 0.3
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        KEYPOINTS = ['Ankle', 'Knee', 'Hip', 'MidHip']
        SIDES = ['L', 'R']
    
    def build_feature_columns():
        cols = []
        for side in Config.SIDES:
            for kp in Config.KEYPOINTS:
                cols.append(f"{side}{kp}_x")
                cols.append(f"{side}{kp}_y")
        return cols
    
    def extract_features(df, feature_cols):
        positions = df[feature_cols].values
        velocities = np.diff(positions, axis=0, prepend=positions[0:1])
        accelerations = np.diff(velocities, axis=0, prepend=velocities[0:1])
        features = np.hstack([positions, velocities, accelerations])
        return features
    
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
            lstm_out, _ = self.lstm(x)
            last_out = lstm_out[:, -1, :]
            out = self.fc(last_out)
            return out

# =========================
# 配置區
# =========================
class PredictorConfig:
    """預測器配置（根據你的實際路徑修改）"""
    
    # === 必須設定的路徑 ===
    
    # 1. 已訓練好的模型檔案
    MODEL_PATH = r"C:\mydata\sf\comp\model\0129_test.pth"
    
    # 2. 要預測的原始 CSV（可以是訓練時沒見過的新資料）
    CSV_PATH = r"C:\mydata\sf\open\output_csv\0128\A\0128_A_1.csv"
    
    # 3. 對應的影片（可選，用於視覺化驗證）
    VIDEO_PATH = r"C:\mydata\sf\open\walking_video\0128\0128_A_1.mp4"
    
    # 4. 輸出資料夾（模型自動檢測的週期會儲存在這裡）
    OUTPUT_DIR = r"C:\mydata\sf\open\output_csv\0128cut\A\auto"
    
    # === 預測參數 ===
    THRESHOLD = 0.3        # 端點機率閾值（越高越嚴格）- 降低到 0.3 更容易檢測
    MIN_DISTANCE = 15      # 端點之間最小距離（frames）
    MIN_CYCLE_LENGTH = 15  # 週期最小長度（frames）

# =========================
# 預測器類別
# =========================
class GaitPredictor:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = None
        self.scaler = None
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """載入訓練好的模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型檔案: {model_path}")
        
        # 計算輸入維度
        feature_cols = build_feature_columns()
        input_size = len(feature_cols) * 3  # 位置 + 速度 + 加速度
        
        # 創建模型結構
        self.model = GaitDetectorLSTM(
            input_size=input_size,
            hidden_size=Config.HIDDEN_SIZE,
            num_layers=Config.NUM_LAYERS,
            dropout=Config.DROPOUT
        ).to(self.device)
        
        # 載入權重
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 處理不同的儲存格式
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ 模型已載入 (Epoch {checkpoint.get('epoch', '?')})")
            print(f"  驗證損失: {checkpoint.get('val_loss', '?'):.4f}")
            print(f"  驗證準確率: {checkpoint.get('val_acc', '?'):.3f}")
        else:
            self.model.load_state_dict(checkpoint)
            print(f"✓ 模型已載入: {model_path}")
        
        self.model.eval()
    
    def predict(self, df):
        """
        預測每個 frame 是否為端點
        
        返回：
        - probabilities: 每個 frame 的端點機率 (0~1)
        """
        # 提取特徵
        feature_cols = build_feature_columns()
        
        # 檢查欄位是否存在
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            print(f"警告: CSV 缺少以下欄位，將設為 0: {missing_cols}")
            for col in missing_cols:
                df[col] = 0.0
        
        features = extract_features(df, feature_cols)
        
        # 標準化
        from sklearn.preprocessing import StandardScaler
        if self.scaler is None:
            self.scaler = StandardScaler()
            features = self.scaler.fit_transform(features)
        else:
            features = self.scaler.transform(features)
        
        # 滑動視窗預測
        window_size = Config.WINDOW_SIZE
        n_frames = len(features)
        probabilities = np.zeros(n_frames)
        
        print(f"開始預測 {n_frames} 幀...")
        
        with torch.no_grad():
            for i in range(n_frames - window_size):
                x = features[i:i+window_size]
                x = torch.FloatTensor(x).unsqueeze(0).to(self.device)
                
                prob = self.model(x).item()
                probabilities[i + window_size//2] = prob
                
                # 進度顯示
                if (i + 1) % 100 == 0:
                    print(f"  已處理: {i+1}/{n_frames-window_size}")
        
        print("✓ 預測完成")
        return probabilities
    
    def extract_endpoints(self, probabilities, threshold=0.5, min_distance=20):
        """
        從機率曲線中提取端點
        
        參數：
        - threshold: 機率閾值
        - min_distance: 端點之間最小距離
        """
        # 找出高於閾值的峰值
        peaks, properties = find_peaks(
            probabilities,
            height=threshold,
            distance=min_distance
        )
        
        return peaks, properties
    
    def extract_cycles(self, endpoints, min_cycle_length=20):
        """
        從端點提取週期
        """
        if len(endpoints) < 2:
            print("警告: 端點數量不足，無法形成週期")
            return []
        
        cycles = []
        for i in range(len(endpoints) - 1):
            start = endpoints[i]
            end = endpoints[i + 1]
            
            if end - start >= min_cycle_length:
                cycles.append({
                    'start': int(start),
                    'end': int(end),
                    'length': int(end - start)
                })
        
        return cycles
    
    def visualize_prediction(self, df, probabilities, endpoints, cycles, save_path=None):
        """視覺化預測結果"""
        fig, axes = plt.subplots(4, 1, figsize=(20, 12), sharex=True)
        
        n = len(df)
        frames = np.arange(n)
        
        # 1. 左腳踝 Y 座標
        if 'LAnkle_y' in df.columns:
            axes[0].plot(frames, df['LAnkle_y'], 'b-', linewidth=1, label='L Ankle Y')
            for ep in endpoints:
                axes[0].axvline(ep, color='red', linestyle='--', alpha=0.5)
            axes[0].set_ylabel('L Ankle Y')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[0].set_title('Left Ankle Y Position')
        
        # 2. 右腳踝 Y 座標
        if 'RAnkle_y' in df.columns:
            axes[1].plot(frames, df['RAnkle_y'], 'g-', linewidth=1, label='R Ankle Y')
            for ep in endpoints:
                axes[1].axvline(ep, color='red', linestyle='--', alpha=0.5)
            axes[1].set_ylabel('R Ankle Y')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].set_title('Right Ankle Y Position')
        
        # 3. 端點機率
        axes[2].plot(frames[:len(probabilities)], probabilities, 'purple', linewidth=2, label='Endpoint Probability')
        axes[2].axhline(PredictorConfig.THRESHOLD, color='orange', linestyle='--', label=f'Threshold ({PredictorConfig.THRESHOLD})')
        for ep in endpoints:
            axes[2].axvline(ep, color='red', linestyle='--', alpha=0.5)
        axes[2].scatter(endpoints, probabilities[endpoints], color='red', s=100, zorder=5, label='Detected Endpoints')
        axes[2].set_ylabel('Probability')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_title(f'Endpoint Detection (Found {len(endpoints)} endpoints)')
        
        # 4. 檢測到的週期
        if 'LAnkle_y' in df.columns:
            axes[3].plot(frames, df['LAnkle_y'], 'k-', linewidth=1, alpha=0.3)
        
        for i, cycle in enumerate(cycles):
            color = plt.cm.tab10(i % 10)
            axes[3].axvspan(cycle['start'], cycle['end'], color=color, alpha=0.3, label=f"Cycle {i+1} ({cycle['length']}f)")
            axes[3].axvline(cycle['start'], color=color, linestyle='-', linewidth=2)
            axes[3].axvline(cycle['end'], color=color, linestyle='-', linewidth=2)
        
        axes[3].set_ylabel('Detected Cycles')
        axes[3].set_xlabel('Frame Index')
        axes[3].set_title(f'Detected Cycles (Total: {len(cycles)})')
        if len(cycles) > 0 and len(cycles) <= 10:
            axes[3].legend(loc='upper right', ncol=2)
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            # 確保輸出資料夾存在
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 視覺化圖表已儲存: {save_path}")
        
        return fig

    def save_cycles(self, df, cycles, output_dir):
        """將檢測到的週期儲存為 CSV 檔案"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"✓ 創建輸出資料夾: {output_dir}")
        
        saved_files = []
        
        for i, cycle in enumerate(cycles):
            # 提取週期資料
            cycle_df = df.iloc[cycle['start']:cycle['end']].copy()
            cycle_df.reset_index(drop=True, inplace=True)
            
            # 生成檔名
            filename = f"auto_{i+1}_{cycle['start']}_{cycle['end']}.csv"
            filepath = os.path.join(output_dir, filename)
            
            # 儲存
            cycle_df.to_csv(filepath, index=False)
            saved_files.append(filepath)
        
        print(f"✓ 已儲存 {len(saved_files)} 個週期檔案到: {output_dir}")
        return saved_files

# =========================
# 主要函數
# =========================
def predict_and_extract(csv_path, model_path, 
                        threshold=0.5, 
                        min_distance=20,
                        min_cycle_length=20,
                        output_dir=None,
                        visualize=True):
    """
    完整的預測和週期提取流程
    
    參數：
    - csv_path: 要預測的 CSV 檔案路徑
    - model_path: 訓練好的模型路徑
    - threshold: 端點檢測閾值
    - min_distance: 端點之間最小距離
    - min_cycle_length: 週期最小長度
    - output_dir: 輸出資料夾（None 則不儲存）
    - visualize: 是否顯示視覺化
    
    返回：
    - cycles: 檢測到的週期列表
    - df: 原始資料
    """
    print("=" * 60)
    print("深度學習步態週期自動檢測")
    print("=" * 60)
    print(f"\n處理檔案: {csv_path}")
    
    # 載入資料
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到 CSV 檔案: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"✓ 資料已載入: {len(df)} frames")
    
    # 創建預測器
    predictor = GaitPredictor(model_path, device=Config.DEVICE)
    
    # 預測
    print(f"\n{'=' * 60}")
    print("開始預測...")
    print(f"{'=' * 60}")
    probabilities = predictor.predict(df)
    
    # 提取端點
    print(f"\n{'=' * 60}")
    print("提取端點...")
    print(f"{'=' * 60}")
    endpoints, properties = predictor.extract_endpoints(
        probabilities,
        threshold=threshold,
        min_distance=min_distance
    )
    print(f"✓ 找到 {len(endpoints)} 個端點")
    
    # 如果沒有找到端點，嘗試降低閾值
    if len(endpoints) == 0:
        print(f"\n⚠️  沒有找到端點，嘗試降低閾值重新檢測...")
        
        # 顯示機率分佈統計
        prob_max = probabilities.max()
        prob_mean = probabilities.mean()
        prob_std = probabilities.std()
        print(f"  機率統計: 最大={prob_max:.3f}, 平均={prob_mean:.3f}, 標準差={prob_std:.3f}")
        
        # 嘗試多個閾值
        for new_threshold in [0.3, 0.2, 0.1, 0.05]:
            print(f"  嘗試閾值 {new_threshold}...")
            endpoints, properties = predictor.extract_endpoints(
                probabilities,
                threshold=new_threshold,
                min_distance=min_distance
            )
            print(f"    → 找到 {len(endpoints)} 個端點")
            
            if len(endpoints) >= 2:
                print(f"  ✓ 使用閾值 {new_threshold} 找到足夠的端點")
                threshold = new_threshold
                break
    
    if len(endpoints) > 0:
        print(f"  端點位置: {endpoints[:10]}{'...' if len(endpoints) > 10 else ''}")
    
    # 提取週期
    print(f"\n{'=' * 60}")
    print("提取週期...")
    print(f"{'=' * 60}")
    cycles = predictor.extract_cycles(endpoints, min_cycle_length)
    print(f"✓ 提取 {len(cycles)} 個週期")
    
    if len(cycles) > 0:
        print("\n週期資訊:")
        for i, cycle in enumerate(cycles, 1):
            print(f"  週期 {i}: frame {cycle['start']:4d} ~ {cycle['end']:4d} (長度: {cycle['length']:3d})")
    
    # 儲存週期
    if output_dir and len(cycles) > 0:
        print(f"\n{'=' * 60}")
        print("儲存週期檔案...")
        print(f"{'=' * 60}")
        predictor.save_cycles(df, cycles, output_dir)
    
    # 視覺化
    if visualize:
        print(f"\n{'=' * 60}")
        print("生成視覺化圖表...")
        print(f"{'=' * 60}")
        
        save_path = None
        if output_dir:
            save_path = os.path.join(output_dir, "prediction_visualization.png")
        
        fig = predictor.visualize_prediction(df, probabilities, endpoints, cycles, save_path)
        plt.show()
    
    print(f"\n{'=' * 60}")
    print("完成!")
    print(f"{'=' * 60}\n")
    
    return cycles, df, probabilities, endpoints

# =========================
# 主程式
# =========================
def main():
    """主程式"""
    
    print("\n" + "=" * 60)
    print("深度學習步態週期自動檢測系統")
    print("=" * 60)
    
    # 檢查模型是否存在
    if not os.path.exists(PredictorConfig.MODEL_PATH):
        print(f"\n❌ 錯誤: 找不到模型檔案")
        print(f"   路徑: {PredictorConfig.MODEL_PATH}")
        print(f"\n請先執行 gait_dl_detector_fixed.py 訓練模型")
        print(f"或檢查 MODEL_PATH 設定是否正確\n")
        return
    
    # 檢查 CSV 是否存在
    if not os.path.exists(PredictorConfig.CSV_PATH):
        print(f"\n❌ 錯誤: 找不到 CSV 檔案")
        print(f"   路徑: {PredictorConfig.CSV_PATH}")
        print(f"\n請檢查 CSV_PATH 設定是否正確\n")
        return
    
    print(f"\n配置:")
    print(f"  模型: {PredictorConfig.MODEL_PATH}")
    print(f"  CSV:  {PredictorConfig.CSV_PATH}")
    print(f"  輸出: {PredictorConfig.OUTPUT_DIR}")
    print(f"\n參數:")
    print(f"  閾值: {PredictorConfig.THRESHOLD}")
    print(f"  最小端點距離: {PredictorConfig.MIN_DISTANCE}")
    print(f"  最小週期長度: {PredictorConfig.MIN_CYCLE_LENGTH}")
    
    # 執行預測
    try:
        cycles, df, probabilities, endpoints = predict_and_extract(
            csv_path=PredictorConfig.CSV_PATH,
            model_path=PredictorConfig.MODEL_PATH,
            threshold=PredictorConfig.THRESHOLD,
            min_distance=PredictorConfig.MIN_DISTANCE,
            min_cycle_length=PredictorConfig.MIN_CYCLE_LENGTH,
            output_dir=PredictorConfig.OUTPUT_DIR,
            visualize=True
        )
        
        # 檢查結果並給予建議
        if len(cycles) == 0:
            print("\n⚠️  沒有檢測到週期！")
            print("\n可能的原因和建議:")
            print("  1. 模型訓練資料不足或品質不佳")
            print("     → 增加訓練資料量，確保標註準確")
            print("  2. 測試資料與訓練資料差異太大")
            print("     → 使用更多樣化的訓練資料")
            print("  3. 參數設定不合適")
            print(f"     → 目前閾值: {PredictorConfig.THRESHOLD}")
            print(f"     → 目前最小距離: {PredictorConfig.MIN_DISTANCE}")
            print(f"     → 目前最小週期長度: {PredictorConfig.MIN_CYCLE_LENGTH}")
            
            # 顯示機率統計
            prob_stats = {
                'max': probabilities.max(),
                'mean': probabilities.mean(),
                'std': probabilities.std(),
                'median': np.median(probabilities)
            }
            print(f"\n  機率分佈統計:")
            print(f"     最大值: {prob_stats['max']:.4f}")
            print(f"     平均值: {prob_stats['mean']:.4f}")
            print(f"     中位數: {prob_stats['median']:.4f}")
            print(f"     標準差: {prob_stats['std']:.4f}")
            
            if prob_stats['max'] < 0.3:
                print(f"\n  ⚠️  所有預測機率都很低 (max={prob_stats['max']:.3f})")
                print(f"     這表示模型對這個資料不確定，建議:")
                print(f"     - 重新訓練模型，使用更多訓練資料")
                print(f"     - 檢查測試資料的關鍵點品質")
        else:
            print(f"\n✓ 成功檢測到 {len(cycles)} 個週期")
            print(f"  平均週期長度: {np.mean([c['length'] for c in cycles]):.1f} frames")
            print(f"  週期長度範圍: {min(c['length'] for c in cycles)} ~ {max(c['length'] for c in cycles)} frames")
        
    except Exception as e:
        print(f"\n❌ 執行錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()