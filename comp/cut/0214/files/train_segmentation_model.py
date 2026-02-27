"""
完整版序列標註訓練程式
支援從片段資料夾讀取真實標註
pip install tensorflow pandas numpy matplotlib scikit-learn
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    HAS_TF = True
    print(f"✓ TensorFlow {tf.__version__}")
except:
    HAS_TF = False
    print("⚠️  TensorFlow未安裝")

# =========================
# 配置
# =========================
output_dir = r"C:\mydata\sf\result_p\0214"
class Config:
    # ===== 修改這裡的路徑 =====
    ORIGINAL_CSV = r"C:\mydata\sf\open\output_csv\0128\A\0128_A_3.csv"
    SEGMENTS_DIR = r"C:\mydata\sf\open\output_csv\0128cut\A\A_3"
    OUTPUT_DIR = r"C:\mydata\sf\comp\model\0214test"
    # ========================
    
    # 模型參數
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    DROPOUT = 0.3
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    
    # 特徵
    USE_ANKLE_ONLY = True
    USE_VELOCITY = True
    USE_ACCELERATION = True
    
    # 標籤
    LABEL_MAP = {'O': 0, 'B': 1, 'I': 2}
    LABEL_NAMES = ['Outside', 'Begin', 'Inside']

config = Config()

# =========================
# 核心函數: 從片段資料夾生成標籤
# =========================
def load_segments_and_generate_labels(original_csv_path, segments_dir):
    """
    關鍵函數: 讀取切好的片段CSV,生成BIO標籤
    
    支援兩種匹配方式:
    1. 如果片段CSV有original_frame欄位,直接使用
    2. 否則通過特徵匹配找到對應的原始幀位置
    """
    
    print("\n" + "="*60)
    print("載入資料並生成標籤...")
    print("="*60)
    
    # 讀取原始CSV
    if not os.path.exists(original_csv_path):
        print(f"⚠️  檔案不存在: {original_csv_path}")
        print("使用測試資料...")
        original_csv_path = "/mnt/user-data/uploads/0128_A_3.csv"
    
    df_original = pd.read_csv(original_csv_path)
    n_frames = len(df_original)
    print(f"✓ 原始CSV: {n_frames} 幀")
    
    # 初始化標籤 (全部為O)
    labels = np.zeros(n_frames, dtype=int)
    
    # 檢查片段資料夾
    if not os.path.exists(segments_dir):
        print(f"⚠️  片段資料夾不存在: {segments_dir}")
        print("使用自動生成的示範標籤...")
        return df_original, generate_demo_labels(df_original)
    
    # 讀取所有片段
    segment_files = glob.glob(os.path.join(segments_dir, "*.csv"))
    
    if len(segment_files) == 0:
        print(f"⚠️  資料夾中沒有CSV檔案: {segments_dir}")
        return df_original, generate_demo_labels(df_original)
    
    print(f"✓ 找到 {len(segment_files)} 個片段CSV")
    
    segments_info = []
    matched_count = 0
    
    for seg_file in sorted(segment_files):
        try:
            df_seg = pd.read_csv(seg_file)
            seg_name = Path(seg_file).stem
            
            # 方法1: 使用original_frame欄位
            if 'original_frame' in df_seg.columns:
                start_frame = int(df_seg['original_frame'].iloc[0])
                end_frame = int(df_seg['original_frame'].iloc[-1]) + 1
                method = "original_frame"
            
            # 方法2: 特徵匹配
            else:
                start_frame, end_frame = match_segment_to_original(
                    df_seg, df_original
                )
                method = "feature_match"
            
            # 驗證範圍
            if start_frame < 0 or end_frame > n_frames or start_frame >= end_frame:
                print(f"  ⚠️  跳過 {seg_name}: 無效範圍 [{start_frame}, {end_frame}]")
                continue
            
            # 標註
            labels[start_frame] = 1  # B
            labels[start_frame+1:end_frame] = 2  # I
            
            segments_info.append({
                'name': seg_name,
                'start': start_frame,
                'end': end_frame,
                'length': end_frame - start_frame,
                'method': method
            })
            matched_count += 1
            
        except Exception as e:
            print(f"  ⚠️  讀取失敗 {Path(seg_file).name}: {e}")
    
    print(f"\n✓ 成功匹配 {matched_count}/{len(segment_files)} 個片段")
    print("\n片段資訊:")
    for seg in segments_info[:10]:  # 顯示前10個
        print(f"  {seg['name']:25s} | [{seg['start']:4d}, {seg['end']:4d}] "
              f"長度={seg['length']:3d} | {seg['method']}")
    
    if len(segments_info) > 10:
        print(f"  ... 還有 {len(segments_info)-10} 個片段")
    
    # 統計標籤分布
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\n標籤分布:")
    for label_id, count in zip(unique, counts):
        label_name = config.LABEL_NAMES[label_id]
        print(f"  {label_name:8s} ({label_id}): {count:5d} 幀 ({count/n_frames*100:5.1f}%)")
    
    return df_original, labels


def match_segment_to_original(df_seg, df_original):
    """
    通過特徵匹配找到片段在原始CSV中的位置
    
    策略:
    1. 使用多個穩定關節點(鼻子、頸部、臀部中點)
    2. 計算歐氏距離
    3. 找最小距離位置
    """
    
    # 選擇穩定的特徵進行匹配
    match_features = ['Nose_x', 'Nose_y', 'Neck_x', 'Neck_y', 'MidHip_x', 'MidHip_y']
    
    # 確保所有特徵都存在
    available_features = [f for f in match_features if f in df_seg.columns and f in df_original.columns]
    
    if len(available_features) == 0:
        # 降級使用腳踝
        available_features = ['LAnkle_x', 'LAnkle_y', 'RAnkle_x', 'RAnkle_y']
    
    # 提取第一幀特徵
    first_row = df_seg[available_features].iloc[0].values
    
    # 在原始CSV中搜尋
    original_features = df_original[available_features].values
    
    # 計算距離
    distances = np.sqrt(np.sum((original_features - first_row)**2, axis=1))
    
    # 找最小距離
    start_frame = np.argmin(distances)
    end_frame = start_frame + len(df_seg)
    
    # 驗證匹配品質
    min_distance = distances[start_frame]
    if min_distance > 10:  # 閾值可調整
        print(f"    ⚠️  匹配距離較大: {min_distance:.2f}")
    
    return start_frame, end_frame


def generate_demo_labels(df):
    """備用: 使用規則生成示範標籤"""
    print("  使用規則生成示範標籤...")
    
    labels = [0] * len(df)  # 全部Outside
    
    # 簡單規則找切割點
    FLAT_WINDOW = 5
    FLAT_SLOPE = 0.1
    MIN_FLAT_LEN = 5
    RISE_TOTAL = 3.0
    
    for side in ['L', 'R']:
        col_x = f"{side}Ankle_x"
        if col_x not in df.columns:
            continue
            
        x = df[col_x].values
        n = len(x)
        
        is_flat = np.zeros(n, dtype=bool)
        for i in range(n - FLAT_WINDOW):
            slope = (x[i + FLAT_WINDOW] - x[i]) / FLAT_WINDOW
            if abs(slope) < FLAT_SLOPE:
                is_flat[i:i + FLAT_WINDOW] = True
        
        flat_segments = []
        start = None
        for i, v in enumerate(is_flat):
            if v and start is None:
                start = i
            elif not v and start is not None:
                if i - start >= MIN_FLAT_LEN:
                    flat_segments.append((start, i))
                start = None
        
        for i in range(len(flat_segments) - 1):
            fs, fe = flat_segments[i]
            ns, _ = flat_segments[i + 1]
            
            if abs(x[ns] - x[fe]) > RISE_TOTAL:
                if fs < len(labels) and ns <= len(labels):
                    labels[fs] = 1
                    for j in range(fs + 1, ns):
                        labels[j] = 2
    
    return labels


# =========================
# 特徵提取
# =========================
def extract_features(df):
    """提取12維特徵向量"""
    
    features_list = []
    
    # 腳踝座標
    ankle_cols = ['LAnkle_x', 'LAnkle_y', 'RAnkle_x', 'RAnkle_y']
    ankle_data = df[ankle_cols].values
    features_list.append(ankle_data)
    
    # 速度
    velocity = np.zeros_like(ankle_data)
    velocity[1:] = ankle_data[1:] - ankle_data[:-1]
    velocity[0] = velocity[1]
    features_list.append(velocity)
    
    # 加速度
    acceleration = np.zeros_like(velocity)
    acceleration[1:] = velocity[1:] - velocity[:-1]
    acceleration[0] = acceleration[1]
    features_list.append(acceleration)
    
    features = np.concatenate(features_list, axis=1)
    
    return features


# =========================
# 建立模型
# =========================
def build_lstm_model(input_shape, num_classes):
    """建立雙向LSTM模型"""
    
    if not HAS_TF:
        return None
    
    model = models.Sequential([
        layers.Bidirectional(
            layers.LSTM(config.HIDDEN_SIZE, return_sequences=True),
            input_shape=input_shape
        ),
        layers.Dropout(config.DROPOUT),
        
        layers.Bidirectional(
            layers.LSTM(config.HIDDEN_SIZE // 2, return_sequences=True)
        ),
        layers.Dropout(config.DROPOUT),
        
        layers.TimeDistributed(
            layers.Dense(num_classes, activation='softmax')
        )
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# =========================
# 視覺化
# =========================
def visualize_results(df, features, true_labels, pred_labels, history=None):
    """生成完整視覺化"""
    
    max_frames = min(500, len(true_labels))
    
    if history is not None:
        fig = plt.figure(figsize=(16, 14))
        gs = fig.add_gridspec(5, 2, hspace=0.3, wspace=0.3)
    else:
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 1, hspace=0.3)
    
    # 1. 軌跡
    ax1 = fig.add_subplot(gs[0, :] if history else gs[0])
    ax1.plot(features[:max_frames, 1], 'b-', label='Left Ankle Y', alpha=0.7, lw=1.5)
    ax1.plot(features[:max_frames, 3], 'r-', label='Right Ankle Y', alpha=0.7, lw=1.5)
    ax1.set_ylabel('Y Position', fontsize=12)
    ax1.set_title('Ankle Trajectories', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 真實標籤
    ax2 = fig.add_subplot(gs[1, :] if history else gs[1])
    colors = ['gray', 'green', 'blue']
    for i, label_name in enumerate(config.LABEL_NAMES):
        mask = true_labels[:max_frames] == i
        if np.any(mask):
            frames = np.where(mask)[0]
            ax2.scatter(frames, [1]*len(frames), c=colors[i], 
                       label=label_name, s=20, alpha=0.7)
    ax2.set_ylabel('True Labels', fontsize=12)
    ax2.set_title('Ground Truth Annotations', fontsize=14, fontweight='bold')
    ax2.set_yticks([])
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.set_xlim([0, max_frames])
    
    # 3. 預測標籤
    ax3 = fig.add_subplot(gs[2, :] if history else gs[2])
    for i, label_name in enumerate(config.LABEL_NAMES):
        mask = pred_labels[:max_frames] == i
        if np.any(mask):
            frames = np.where(mask)[0]
            ax3.scatter(frames, [1]*len(frames), c=colors[i], 
                       label=label_name, s=20, alpha=0.7)
    ax3.set_xlabel('Frame Index', fontsize=12)
    ax3.set_ylabel('Predictions', fontsize=12)
    ax3.set_title('Model Predictions', fontsize=14, fontweight='bold')
    ax3.set_yticks([])
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.set_xlim([0, max_frames])
    
    if history:
        # 訓練曲線
        ax4 = fig.add_subplot(gs[3, 0])
        ax4.plot(history.history['loss'], 'b-', lw=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4.set_title('Training Loss', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        ax5 = fig.add_subplot(gs[3, 1])
        ax5.plot(history.history['accuracy'], 'g-', lw=2)
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Accuracy')
        ax5.set_title('Training Accuracy', fontweight='bold')
        ax5.set_ylim([0, 1])
        ax5.grid(True, alpha=0.3)
        
        # 混淆矩陣
        ax6 = fig.add_subplot(gs[4, :])
        cm = confusion_matrix(true_labels, pred_labels)
        im = ax6.imshow(cm, cmap='Blues')
        ax6.set_xticks(range(3))
        ax6.set_yticks(range(3))
        ax6.set_xticklabels(config.LABEL_NAMES)
        ax6.set_yticklabels(config.LABEL_NAMES)
        ax6.set_xlabel('Predicted')
        ax6.set_ylabel('True')
        ax6.set_title('Confusion Matrix', fontweight='bold')
        
        for i in range(3):
            for j in range(3):
                text = ax6.text(j, i, f'{cm[i, j]}',
                              ha="center", va="center",
                              color="white" if cm[i, j] > cm.max()/2 else "black")
        plt.colorbar(im, ax=ax6)
    
    return fig


# =========================
# 主程式
# =========================
def main():
    print("\n" + "="*60)
    print("序列標註模型訓練")
    print("="*60)
    
    # 1. 載入資料並生成標籤
    df, labels = load_segments_and_generate_labels(
        config.ORIGINAL_CSV,
        config.SEGMENTS_DIR
    )
    
    # 2. 提取特徵
    print("\n提取特徵...")
    features = extract_features(df)
    print(f"✓ 特徵形狀: {features.shape}")
    
    # 3. 標準化
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    # 4. 準備訓練資料
    X = features.reshape(1, -1, features.shape[1])
    y = np.array(labels).reshape(1, -1)
    
    # 5. 建立並訓練模型
    history = None
    
    if HAS_TF:
        print("\n建立LSTM模型...")
        model = build_lstm_model(
            input_shape=(X.shape[1], X.shape[2]),
            num_classes=3
        )
        model.summary()
        
        print("\n開始訓練...")
        history = model.fit(X, y, epochs=config.NUM_EPOCHS, verbose=1)
        
        # 預測
        predictions = model.predict(X)
        pred_labels = np.argmax(predictions[0], axis=-1)
        
    else:
        print("\n⚠️  TensorFlow未安裝,跳過訓練")
        pred_labels = labels  # 使用真實標籤代替
    
    # 6. 評估
    print("\n" + "="*60)
    print("評估結果")
    print("="*60)
    accuracy = np.mean(pred_labels == labels)
    print(f"準確率: {accuracy:.4f}")
    print("\n分類報告:")
    print(classification_report(labels, pred_labels, target_names=config.LABEL_NAMES))
    
    # 7. 視覺化
    print("\n生成視覺化...")
    fig = visualize_results(df, features, labels, pred_labels, history)
    
    # 8. 儲存
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig_path = os.path.join(output_dir, 'training_results.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"✓ 結果已儲存: {fig_path}")
    
    if HAS_TF and model:
        model_path = os.path.join(output_dir, 'segmentation_model.h5')
        model.save(model_path)
        print(f"✓ 模型已儲存: {model_path}")
    
    print("\n" + "="*60)
    print("完成!")
    print("="*60)

if __name__ == "__main__":
    main()
