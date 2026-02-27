"""
測試程式 - 匹配 final_model (含 Hip_x 特徵)
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TF = True
except:
    HAS_TF = False
    print("❌ TensorFlow未安裝!")
    exit()

# =========================
# 配置
# =========================
class Config:
    # ===== 測試資料路徑 =====
    TEST_CSV = r"C:\mydata\sf\open\output_csv\0128\A\0128_A_2.csv"  # 測試資料的完整CSV
    TEST_SEGMENTS_DIR = r"C:\mydata\sf\open\output_csv\0128cut\A\A_2"  # 測試資料的片段(Ground Truth)
    
    # ===== 訓練好的模型路徑 =====
    MODEL_PATH = r"C:\mydata\sf\comp\model\0214test\v3-3\final_model.h5"  # 或 segmentation_model.h5
    
    # ===== 輸出路徑 =====
    OUTPUT_DIR = r"C:\mydata\sf\result_p\0214\v3-3"
    # =====================
    
    # 特徵配置 (必須與訓練時一致!)
    USE_HIP_X = True  # 關鍵!
    USE_VELOCITY = True
    USE_ACCELERATION = True
    
    BEGIN_CONTEXT_WINDOW = 3
    
    LABEL_MAP = {'O': 0, 'B': 1, 'I': 2}
    LABEL_NAMES = ['Outside', 'Begin', 'Inside']

config = Config()

# =========================
# 載入Ground Truth標籤
# =========================
def load_test_labels(csv_path, segments_dir):
    """載入測試資料的Ground Truth標籤"""
    
    print("\n" + "="*60)
    print("載入測試資料...")
    print("="*60)
    
    if not os.path.exists(csv_path):
        print(f"❌ 測試CSV不存在: {csv_path}")
        return None, None
    
    df = pd.read_csv(csv_path)
    n_frames = len(df)
    print(f"✓ 測試CSV: {n_frames} 幀")
    
    labels = np.zeros(n_frames, dtype=int)
    
    if not os.path.exists(segments_dir):
        print(f"⚠️  沒有Ground Truth片段資料夾")
        return df, None
    
    segment_files = glob.glob(os.path.join(segments_dir, "*.csv"))
    
    if len(segment_files) == 0:
        print(f"⚠️  片段資料夾為空")
        return df, None
    
    print(f"✓ 找到 {len(segment_files)} 個Ground Truth片段")
    
    matched_count = 0
    for seg_file in sorted(segment_files):
        try:
            df_seg = pd.read_csv(seg_file)
            
            if 'original_frame' in df_seg.columns:
                start_frame = int(df_seg['original_frame'].iloc[0])
                end_frame = int(df_seg['original_frame'].iloc[-1]) + 1
            else:
                start_frame, end_frame = match_segment(df_seg, df)
            
            if start_frame < 0 or end_frame > n_frames or start_frame >= end_frame:
                continue
            
            begin_start = max(0, start_frame - config.BEGIN_CONTEXT_WINDOW)
            begin_end = min(n_frames, start_frame + config.BEGIN_CONTEXT_WINDOW + 1)
            
            labels[begin_start:begin_end] = 1
            labels[begin_end:end_frame] = 2
            
            matched_count += 1
            
        except Exception as e:
            continue
    
    print(f"✓ 成功匹配 {matched_count}/{len(segment_files)} 個片段")
    
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nGround Truth標籤分布:")
    for label_id, count in zip(unique, counts):
        label_name = config.LABEL_NAMES[label_id]
        print(f"  {label_name:8s} ({label_id}): {count:5d} 幀 ({count/n_frames*100:5.1f}%)")
    
    return df, labels


def match_segment(df_seg, df_original):
    """特徵匹配找片段位置"""
    match_features = ['Nose_x', 'Nose_y', 'Neck_x', 'Neck_y']
    available = [f for f in match_features if f in df_seg.columns and f in df_original.columns]
    
    if len(available) == 0:
        available = ['LAnkle_x', 'LAnkle_y', 'RAnkle_x', 'RAnkle_y']
    
    first_row = df_seg[available].iloc[0].values
    original_features = df_original[available].values
    distances = np.sqrt(np.sum((original_features - first_row)**2, axis=1))
    
    start_frame = np.argmin(distances)
    end_frame = start_frame + len(df_seg)
    
    return start_frame, end_frame


# =========================
# 關鍵: 與訓練時相同的特徵提取 (含 Hip_x)
# =========================
def extract_features(df):
    """
    提取特徵 - 必須與訓練時一致!
    包含 Hip_x
    """
    print("\n提取特徵...")
    
    features_list = []
    
    # 1. 腳踝座標
    ankle_cols = ['LAnkle_x', 'LAnkle_y', 'RAnkle_x', 'RAnkle_y']
    ankle_data = df[ankle_cols].values
    features_list.append(ankle_data)
    
    # 2. 關鍵: Hip_x
    if config.USE_HIP_X:
        if 'MidHip_x' in df.columns:
            hip_x = df['MidHip_x'].values.reshape(-1, 1)
            features_list.append(hip_x)
            print("  ✓ 加入 Hip_x 特徵")
        else:
            print("  ⚠️  警告: 測試資料沒有 MidHip_x,用平均值代替")
            hip_x = np.mean(df[['LAnkle_x', 'RAnkle_x']].values, axis=1).reshape(-1, 1)
            features_list.append(hip_x)
    
    # 3. 速度
    if config.USE_VELOCITY:
        velocity = np.zeros_like(ankle_data)
        velocity[1:] = ankle_data[1:] - ankle_data[:-1]
        velocity[0] = velocity[1]
        features_list.append(velocity)
        
        # Hip_x 速度
        if config.USE_HIP_X:
            hip_vel = np.zeros((len(hip_x), 1))
            hip_vel[1:] = hip_x[1:] - hip_x[:-1]
            hip_vel[0] = hip_vel[1]
            features_list.append(hip_vel)
    
    # 4. 加速度
    if config.USE_ACCELERATION:
        acceleration = np.zeros_like(velocity)
        acceleration[1:] = velocity[1:] - velocity[:-1]
        acceleration[0] = acceleration[1]
        features_list.append(acceleration)
    
    features = np.concatenate(features_list, axis=1)
    
    print(f"✓ 特徵形狀: {features.shape}")
    expected_dims = 4 + 1 + 5 + 4  # ankle(4) + hip(1) + vel(5) + acc(4) = 14
    if features.shape[1] != expected_dims:
        print(f"  ⚠️  警告: 特徵維度 {features.shape[1]}, 預期 {expected_dims}")
    
    return features


# =========================
# 模型預測
# =========================
def predict_with_model(model, features):
    """使用模型預測(支援任意長度)"""
    
    print("\n進行預測...")
    
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    n_frames = len(features_normalized)
    expected_length = model.input_shape[1]
    
    print(f"測試資料長度: {n_frames} 幀")
    print(f"模型期望輸入長度: {expected_length} 幀")
    
    if n_frames == expected_length:
        X = features_normalized.reshape(1, -1, features_normalized.shape[1])
        predictions = model.predict(X, verbose=0)
        pred_labels = np.argmax(predictions[0], axis=-1)
        predictions_prob = predictions[0]
    
    elif n_frames < expected_length:
        print(f"資料較短,進行padding...")
        padded = np.zeros((expected_length, features_normalized.shape[1]))
        padded[:n_frames] = features_normalized
        
        X = padded.reshape(1, -1, padded.shape[1])
        predictions = model.predict(X, verbose=0)
        
        pred_labels = np.argmax(predictions[0][:n_frames], axis=-1)
        predictions_prob = predictions[0][:n_frames]
    
    else:
        print(f"資料較長,使用滑動窗口預測...")
        
        window_size = expected_length
        stride = window_size // 2
        
        all_predictions = np.zeros((n_frames, 3))
        prediction_counts = np.zeros(n_frames)
        
        for start in range(0, n_frames, stride):
            end = min(start + window_size, n_frames)
            
            if end - start < window_size:
                start = max(0, n_frames - window_size)
                end = n_frames
            
            window_data = np.zeros((window_size, features_normalized.shape[1]))
            actual_len = end - start
            window_data[:actual_len] = features_normalized[start:end]
            
            X = window_data.reshape(1, -1, window_data.shape[1])
            preds = model.predict(X, verbose=0)
            
            all_predictions[start:end] += preds[0][:actual_len]
            prediction_counts[start:end] += 1
            
            if start % (stride * 4) == 0:
                progress = min(100, int(end / n_frames * 100))
                print(f"  進度: {progress}%", end='\r')
        
        print(f"  進度: 100%")
        
        all_predictions = all_predictions / prediction_counts[:, np.newaxis]
        
        pred_labels = np.argmax(all_predictions, axis=-1)
        predictions_prob = all_predictions
    
    unique, counts = np.unique(pred_labels, return_counts=True)
    print(f"\n預測標籤分布:")
    for label_id, count in zip(unique, counts):
        label_name = config.LABEL_NAMES[label_id]
        print(f"  {label_name:8s} ({label_id}): {count:5d} 幀 ({count/len(pred_labels)*100:5.1f}%)")
    
    return pred_labels, predictions_prob


# =========================
# 評估與視覺化
# =========================
def evaluate_and_visualize(df, features, true_labels, pred_labels, predictions_prob):
    """評估並生成視覺化報告"""
    
    print("\n" + "="*60)
    print("測試結果評估")
    print("="*60)
    
    has_ground_truth = true_labels is not None
    
    if has_ground_truth:
        accuracy = accuracy_score(true_labels, pred_labels)
        print(f"\n整體準確率: {accuracy:.4f}")
        
        print("\n分類報告:")
        print(classification_report(true_labels, pred_labels, 
                                   target_names=config.LABEL_NAMES, 
                                   zero_division=0))
        
        cm = confusion_matrix(true_labels, pred_labels)
        print("\n混淆矩陣:")
        print("真實\\預測  ", "  ".join([f"{name:8s}" for name in config.LABEL_NAMES]))
        for i, name in enumerate(config.LABEL_NAMES):
            print(f"{name:8s}  " + "  ".join([f"{cm[i][j]:8d}" for j in range(3)]))
    
    # 視覺化 (簡化版,只顯示關鍵圖表)
    print("\n生成視覺化報告...")
    
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(5, 2, hspace=0.35, wspace=0.3)
    
    max_frames = min(500, len(pred_labels))
    
    # 1. 軌跡 + Hip_x
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(features[:max_frames, 1], 'b-', label='Left Ankle Y', alpha=0.7)
    ax1.plot(features[:max_frames, 3], 'r-', label='Right Ankle Y', alpha=0.7)
    if features.shape[1] >= 5:
        ax1_twin = ax1.twinx()
        ax1_twin.plot(features[:max_frames, 4], 'g--', label='Hip X', alpha=0.5)
        ax1_twin.set_ylabel('Hip X', color='g')
        ax1_twin.legend(loc='upper left')
    ax1.set_ylabel('Ankle Y')
    ax1.set_title('Trajectories (with Hip_x)', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    if has_ground_truth:
        # 2. Ground Truth
        ax2 = fig.add_subplot(gs[1, :])
        colors = ['gray', 'green', 'blue']
        for i, label_name in enumerate(config.LABEL_NAMES):
            mask = true_labels[:max_frames] == i
            if np.any(mask):
                ax2.scatter(np.where(mask)[0], [1]*mask.sum(), c=colors[i], 
                           label=label_name, s=20, alpha=0.7)
        ax2.set_ylabel('Ground Truth')
        ax2.set_title('Ground Truth Labels', fontweight='bold')
        ax2.set_yticks([])
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.set_xlim([0, max_frames])
    
    # 3. 預測
    ax3 = fig.add_subplot(gs[2, :])
    for i, label_name in enumerate(config.LABEL_NAMES):
        mask = pred_labels[:max_frames] == i
        if np.any(mask):
            ax3.scatter(np.where(mask)[0], [1]*mask.sum(), c=colors[i], 
                       label=label_name, s=20, alpha=0.7)
    ax3.set_xlabel('Frame Index')
    ax3.set_ylabel('Predictions')
    ax3.set_title('Model Predictions', fontweight='bold')
    ax3.set_yticks([])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.set_xlim([0, max_frames])
    
    if has_ground_truth:
        # 4. 信心度
        ax4 = fig.add_subplot(gs[3, :])
        for i, label_name in enumerate(config.LABEL_NAMES):
            ax4.plot(predictions_prob[:max_frames, i], 
                    label=label_name, alpha=0.7)
        ax4.set_xlabel('Frame Index')
        ax4.set_ylabel('Confidence')
        ax4.set_title('Prediction Confidence', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim([0, max_frames])
        ax4.set_ylim([0, 1])
        
        # 5. 混淆矩陣
        ax5 = fig.add_subplot(gs[4, :])
        im = ax5.imshow(cm, cmap='Blues')
        ax5.set_xticks(range(3))
        ax5.set_yticks(range(3))
        ax5.set_xticklabels(config.LABEL_NAMES)
        ax5.set_yticklabels(config.LABEL_NAMES)
        ax5.set_xlabel('Predicted')
        ax5.set_ylabel('True')
        ax5.set_title('Confusion Matrix', fontweight='bold')
        
        for i in range(3):
            for j in range(3):
                color = "white" if cm[i, j] > cm.max()/2 else "black"
                ax5.text(j, i, f'{cm[i, j]}', ha="center", va="center", 
                        color=color, fontsize=14)
        plt.colorbar(im, ax=ax5)
    
    return fig, cm if has_ground_truth else None


# =========================
# 主程式
# =========================
def main():
    print("\n" + "="*60)
    print("模型測試程式 (含 Hip_x 特徵)")
    print("="*60)
    
    # 1. 載入模型
    print(f"\n載入模型: {config.MODEL_PATH}")
    if not os.path.exists(config.MODEL_PATH):
        print(f"❌ 模型不存在")
        return
    
    model = keras.models.load_model(config.MODEL_PATH, compile=False)
    print("✓ 模型載入成功")
    print(f"  模型期望輸入: {model.input_shape}")
    
    # 2. 載入測試資料
    df, true_labels = load_test_labels(config.TEST_CSV, config.TEST_SEGMENTS_DIR)
    
    if df is None:
        print("❌ 測試資料載入失敗")
        return
    
    # 3. 提取特徵 (含 Hip_x!)
    features = extract_features(df)
    
    # 4. 預測
    pred_labels, predictions_prob = predict_with_model(model, features)
    
    # 5. 評估與視覺化
    fig, cm = evaluate_and_visualize(df, features, true_labels, pred_labels, predictions_prob)
    
    # 6. 儲存
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    fig_path = os.path.join(config.OUTPUT_DIR, 'test_results_with_hip.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 視覺化報告已儲存: {fig_path}")
    
    # 儲存預測結果
    result_df = df.copy()
    result_df['predicted_label'] = pred_labels
    result_df['predicted_label_name'] = [config.LABEL_NAMES[l] for l in pred_labels]
    
    output_path = os.path.join(config.OUTPUT_DIR, 'predictions.csv')
    result_df.to_csv(output_path, index=False)
    print(f"✓ 預測結果已儲存: {output_path}")
    
    if cm is not None:
        cm_path = os.path.join(config.OUTPUT_DIR, 'confusion_matrix.csv')
        cm_df = pd.DataFrame(cm, 
                            index=config.LABEL_NAMES, 
                            columns=config.LABEL_NAMES)
        cm_df.to_csv(cm_path)
        print(f"✓ 混淆矩陣已儲存: {cm_path}")
    
    print("\n" + "="*60)
    print("測試完成!")
    print("="*60)

if __name__ == "__main__":
    main()
