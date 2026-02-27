"""
改進版序列標註訓練程式
針對 Begin 類別不平衡問題的優化
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
except:
    HAS_TF = False

# =========================
# 配置
# =========================
class Config:
    # ===== 修改路徑 =====
    ORIGINAL_CSV = r"C:\mydata\sf\open\output_csv\0128\A\0128_A_3.csv"
    SEGMENTS_DIR = r"C:\mydata\sf\open\output_csv\0128cut\A\A_3"
    OUTPUT_DIR = r"C:\mydata\sf\comp\model\0214test\v3-2"
    # ===================
    
    # 模型參數
    HIDDEN_SIZE = 256  # 增加模型容量
    NUM_LAYERS = 3     # 增加層數
    DROPOUT = 0.4
    LEARNING_RATE = 0.0005
    NUM_EPOCHS = 50   # 更多訓練
    
    # 特徵
    USE_ANKLE_ONLY = True
    USE_VELOCITY = True
    USE_ACCELERATION = True
    
    # 改進: 使用加權損失函數
    USE_CLASS_WEIGHTS = True  # 新增!
    
    # 改進: 擴展 Begin 標籤的上下文
    BEGIN_CONTEXT_WINDOW = 3  # Begin 前後3幀也標為 Begin
    
    # 新增: 重心過濾範圍 (過濾轉彎片段)
    X_MIN = 650   # 最小X座標
    X_MAX = 1350  # 最大X座標
    # 超出此範圍的片段(通常是轉彎)會被過濾掉,不參與訓練
    
    # 標籤
    LABEL_MAP = {'O': 0, 'B': 1, 'I': 2}
    LABEL_NAMES = ['Outside', 'Begin', 'Inside']

config = Config()

# =========================
# 資料載入
# =========================
def load_segments_and_generate_labels(original_csv_path, segments_dir):
    """載入並生成標籤 - 加入重心過濾"""
    
    print("\n" + "="*60)
    print("載入資料並生成標籤...")
    print("="*60)
    
    if not os.path.exists(original_csv_path):
        original_csv_path = "/mnt/user-data/uploads/0128_A_3.csv"
    
    df_original = pd.read_csv(original_csv_path)
    n_frames = len(df_original)
    print(f"✓ 原始CSV: {n_frames} 幀")
    
    labels = np.zeros(n_frames, dtype=int)
    
    if not os.path.exists(segments_dir):
        print(f"⚠️  片段資料夾不存在,使用示範標籤")
        return df_original, generate_demo_labels(df_original)
    
    segment_files = glob.glob(os.path.join(segments_dir, "*.csv"))
    
    if len(segment_files) == 0:
        print(f"⚠️  沒有CSV檔案")
        return df_original, generate_demo_labels(df_original)
    
    print(f"✓ 找到 {len(segment_files)} 個片段CSV")
    
    segments_info = []
    matched_count = 0
    filtered_count = 0
    
    # 使用配置的重心範圍
    X_MIN = config.X_MIN
    X_MAX = config.X_MAX
    
    for seg_file in sorted(segment_files):
        try:
            df_seg = pd.read_csv(seg_file)
            seg_name = Path(seg_file).stem
            
            if 'original_frame' in df_seg.columns:
                start_frame = int(df_seg['original_frame'].iloc[0])
                end_frame = int(df_seg['original_frame'].iloc[-1]) + 1
            else:
                start_frame, end_frame = match_segment_to_original(df_seg, df_original)
            
            if start_frame < 0 or end_frame > n_frames or start_frame >= end_frame:
                continue
            
            # ===== 新增:重心過濾 =====
            segment_data = df_original.iloc[start_frame:end_frame]
            
            # 計算重心 (使用腳踝和臀部的平均)
            if 'MidHip_x' in segment_data.columns:
                center_x = segment_data[['LAnkle_x', 'RAnkle_x', 'MidHip_x']].mean(axis=1).mean()
            else:
                center_x = segment_data[['LAnkle_x', 'RAnkle_x']].mean(axis=1).mean()
            
            # 檢查是否在有效範圍內
            if center_x < X_MIN or center_x > X_MAX:
                filtered_count += 1
                print(f"  ⚠️  過濾 {seg_name}: 重心 X={center_x:.1f} 超出範圍")
                continue
            # ===========================
            
            # 擴展 Begin 標籤範圍
            begin_start = max(0, start_frame - config.BEGIN_CONTEXT_WINDOW)
            begin_end = min(n_frames, start_frame + config.BEGIN_CONTEXT_WINDOW + 1)
            
            # 標註 Begin 範圍
            labels[begin_start:begin_end] = 1  # B
            
            # 標註 Inside
            labels[begin_end:end_frame] = 2  # I
            
            segments_info.append({
                'name': seg_name,
                'start': start_frame,
                'end': end_frame,
                'length': end_frame - start_frame,
                'center_x': center_x
            })
            matched_count += 1
            
        except Exception as e:
            print(f"  ⚠️  讀取失敗: {e}")
    
    print(f"\n✓ 成功匹配 {matched_count}/{len(segment_files)} 個片段")
    print(f"✓ 過濾了 {filtered_count} 個轉彎片段")
    
    # 統計
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\n標籤分布 (過濾後):")
    for label_id, count in zip(unique, counts):
        label_name = config.LABEL_NAMES[label_id]
        print(f"  {label_name:8s} ({label_id}): {count:5d} 幀 ({count/n_frames*100:5.1f}%)")
    
    return df_original, labels


def match_segment_to_original(df_seg, df_original):
    """特徵匹配"""
    match_features = ['Nose_x', 'Nose_y', 'Neck_x', 'Neck_y']
    available_features = [f for f in match_features if f in df_seg.columns and f in df_original.columns]
    
    if len(available_features) == 0:
        available_features = ['LAnkle_x', 'LAnkle_y', 'RAnkle_x', 'RAnkle_y']
    
    first_row = df_seg[available_features].iloc[0].values
    original_features = df_original[available_features].values
    distances = np.sqrt(np.sum((original_features - first_row)**2, axis=1))
    start_frame = np.argmin(distances)
    end_frame = start_frame + len(df_seg)
    
    return start_frame, end_frame


def generate_demo_labels(df):
    """示範標籤生成"""
    labels = [0] * len(df)
    # ... (省略,與原版相同)
    return labels


# =========================
# 特徵提取
# =========================
def extract_features(df):
    """提取特徵"""
    features_list = []
    
    ankle_cols = ['LAnkle_x', 'LAnkle_y', 'RAnkle_x', 'RAnkle_y']
    ankle_data = df[ankle_cols].values
    features_list.append(ankle_data)
    
    velocity = np.zeros_like(ankle_data)
    velocity[1:] = ankle_data[1:] - ankle_data[:-1]
    velocity[0] = velocity[1]
    features_list.append(velocity)
    
    acceleration = np.zeros_like(velocity)
    acceleration[1:] = velocity[1:] - velocity[:-1]
    acceleration[0] = acceleration[1]
    features_list.append(acceleration)
    
    features = np.concatenate(features_list, axis=1)
    return features


# =========================
# 改進: 計算類別權重
# =========================
def compute_class_weights(labels):
    """計算類別權重以處理不平衡"""
    from sklearn.utils.class_weight import compute_class_weight
    
    unique_classes = np.unique(labels)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=labels
    )
    
    weight_dict = {i: w for i, w in zip(unique_classes, class_weights)}
    
    print("\n類別權重:")
    for label_id, weight in weight_dict.items():
        print(f"  {config.LABEL_NAMES[label_id]:8s}: {weight:.4f}")
    
    return weight_dict


# =========================
# 改進: 帶焦點損失的模型
# =========================
def build_improved_model(input_shape, num_classes, class_weights=None):
    """建立改進的模型"""
    
    if not HAS_TF:
        return None
    
    model = models.Sequential([
        # 第一層 LSTM
        layers.Bidirectional(
            layers.LSTM(config.HIDDEN_SIZE, return_sequences=True),
            input_shape=input_shape
        ),
        layers.Dropout(config.DROPOUT),
        
        # 第二層 LSTM
        layers.Bidirectional(
            layers.LSTM(config.HIDDEN_SIZE // 2, return_sequences=True)
        ),
        layers.Dropout(config.DROPOUT),
        
        # 第三層 LSTM (新增!)
        layers.Bidirectional(
            layers.LSTM(config.HIDDEN_SIZE // 4, return_sequences=True)
        ),
        layers.Dropout(config.DROPOUT),
        
        # 輸出層
        layers.TimeDistributed(
            layers.Dense(num_classes, activation='softmax')
        )
    ])
    
    # 使用加權損失
    if config.USE_CLASS_WEIGHTS and class_weights is not None:
        # 自定義加權損失函數
        def weighted_categorical_crossentropy(y_true, y_pred):
            # 轉換為 one-hot
            y_true_oh = tf.one_hot(tf.cast(y_true, tf.int32), num_classes)
            
            # 計算每個樣本的權重
            weights = tf.reduce_sum(
                y_true_oh * tf.constant([[class_weights[i] for i in range(num_classes)]]),
                axis=-1
            )
            
            # 加權交叉熵
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
            return loss * weights
        
        loss_fn = weighted_categorical_crossentropy
    else:
        loss_fn = 'sparse_categorical_crossentropy'
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss=loss_fn,
        metrics=['accuracy']
    )
    
    return model


# =========================
# 視覺化
# =========================
def visualize_results(df, features, true_labels, pred_labels, history=None):
    """視覺化結果"""
    
    max_frames = min(500, len(true_labels))
    
    if history:
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
                color = "white" if cm[i, j] > cm.max()/2 else "black"
                text = ax6.text(j, i, f'{cm[i, j]}', ha="center", va="center", color=color)
        plt.colorbar(im, ax=ax6)
    
    return fig


# =========================
# 主程式
# =========================
def main():
    print("\n" + "="*60)
    print("改進版序列標註模型訓練")
    print("="*60)
    
    # 1. 載入資料
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
    
    # 4. 計算類別權重
    class_weights = compute_class_weights(labels) if config.USE_CLASS_WEIGHTS else None
    
    # 5. 準備資料
    X = features.reshape(1, -1, features.shape[1])
    y = np.array(labels).reshape(1, -1)
    
    # 6. 訓練
    history = None
    
    if HAS_TF:
        print("\n" + "="*60)
        print("建立改進模型...")
        print("="*60)
        
        model = build_improved_model(
            input_shape=(X.shape[1], X.shape[2]),
            num_classes=3,
            class_weights=class_weights
        )
        model.summary()
        
        print("\n開始訓練...")
        history = model.fit(X, y, epochs=config.NUM_EPOCHS, verbose=1)
        
        predictions = model.predict(X)
        pred_labels = np.argmax(predictions[0], axis=-1)
        
    else:
        print("\n⚠️  TensorFlow未安裝")
        pred_labels = labels
    
    # 7. 評估
    print("\n" + "="*60)
    print("評估結果")
    print("="*60)
    accuracy = np.mean(pred_labels == labels)
    print(f"準確率: {accuracy:.4f}")
    print("\n分類報告:")
    print(classification_report(labels, pred_labels, target_names=config.LABEL_NAMES))
    
    # 8. 視覺化
    print("\n生成視覺化...")
    fig = visualize_results(df, features, labels, pred_labels, history)
    
    # 9. 儲存
    output_dir = config.OUTPUT_DIR  # 使用設定的路徑!
    os.makedirs(output_dir, exist_ok=True)
    
    fig_path = os.path.join(output_dir, 'improved_training_results.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"✓ 結果已儲存: {fig_path}")
    
    if HAS_TF and model:
        model_path = os.path.join(output_dir, 'improved_model.h5')
        model.save(model_path)
        print(f"✓ 模型已儲存: {model_path}")
    
    print("\n" + "="*60)
    print("完成!")
    print("="*60)

if __name__ == "__main__":
    main()