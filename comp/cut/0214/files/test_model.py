"""
模型測試程式
用訓練好的模型測試新資料
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
    
    # 模型參數(要與訓練時一致)
    BEGIN_CONTEXT_WINDOW = 3  # 如果用改進版,設為3; 原版設為0
    
    # 標籤
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
    
    # 初始化標籤為Outside
    labels = np.zeros(n_frames, dtype=int)
    
    # 檢查是否有片段資料夾(Ground Truth)
    if not os.path.exists(segments_dir):
        print(f"⚠️  沒有Ground Truth片段資料夾")
        print(f"   將無法計算準確率,只能輸出預測結果")
        return df, None
    
    segment_files = glob.glob(os.path.join(segments_dir, "*.csv"))
    
    if len(segment_files) == 0:
        print(f"⚠️  片段資料夾為空")
        return df, None
    
    print(f"✓ 找到 {len(segment_files)} 個Ground Truth片段")
    
    # 讀取片段生成標籤
    matched_count = 0
    for seg_file in sorted(segment_files):
        try:
            df_seg = pd.read_csv(seg_file)
            
            # 方法1: 使用original_frame
            if 'original_frame' in df_seg.columns:
                start_frame = int(df_seg['original_frame'].iloc[0])
                end_frame = int(df_seg['original_frame'].iloc[-1]) + 1
            else:
                # 方法2: 特徵匹配
                start_frame, end_frame = match_segment(df_seg, df)
            
            if start_frame < 0 or end_frame > n_frames or start_frame >= end_frame:
                continue
            
            # 標註(與訓練時相同)
            begin_start = max(0, start_frame - config.BEGIN_CONTEXT_WINDOW)
            begin_end = min(n_frames, start_frame + config.BEGIN_CONTEXT_WINDOW + 1)
            
            labels[begin_start:begin_end] = 1  # Begin
            labels[begin_end:end_frame] = 2     # Inside
            
            matched_count += 1
            
        except Exception as e:
            continue
    
    print(f"✓ 成功匹配 {matched_count}/{len(segment_files)} 個片段")
    
    # 統計
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
# 特徵提取(與訓練時相同)
# =========================
def extract_features(df):
    """提取特徵"""
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
# 模型預測
# =========================
def predict_with_model(model, features):
    """使用模型預測(支援任意長度)"""
    
    print("\n進行預測...")
    
    # 標準化
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    n_frames = len(features_normalized)
    
    # 方法1: 如果資料比訓練時短,直接padding
    # 方法2: 如果資料比訓練時長,用滑動窗口
    # 方法3: 重新訓練一個接受任意長度的模型
    
    # 這裡用方法:逐幀預測(雖然慢,但最穩定)
    print(f"測試資料長度: {n_frames} 幀")
    print(f"使用逐幀預測模式...")
    
    # 檢查模型期望的輸入長度
    expected_length = model.input_shape[1]
    print(f"模型期望輸入長度: {expected_length} 幀")
    
    if n_frames == expected_length:
        # 長度相同,直接預測
        X = features_normalized.reshape(1, -1, features_normalized.shape[1])
        predictions = model.predict(X, verbose=0)
        pred_labels = np.argmax(predictions[0], axis=-1)
        predictions_prob = predictions[0]
    
    elif n_frames < expected_length:
        # 資料較短,補0
        print(f"資料較短,進行padding...")
        padded = np.zeros((expected_length, features_normalized.shape[1]))
        padded[:n_frames] = features_normalized
        
        X = padded.reshape(1, -1, padded.shape[1])
        predictions = model.predict(X, verbose=0)
        
        # 只取有效部分
        pred_labels = np.argmax(predictions[0][:n_frames], axis=-1)
        predictions_prob = predictions[0][:n_frames]
    
    else:
        # 資料較長,使用滑動窗口
        print(f"資料較長,使用滑動窗口預測...")
        
        window_size = expected_length
        stride = window_size // 2  # 50% 重疊
        
        all_predictions = np.zeros((n_frames, 3))
        prediction_counts = np.zeros(n_frames)
        
        # 滑動窗口
        for start in range(0, n_frames, stride):
            end = min(start + window_size, n_frames)
            
            # 如果最後一段不夠長,從後面取
            if end - start < window_size:
                start = max(0, n_frames - window_size)
                end = n_frames
            
            # 取窗口資料
            window_data = np.zeros((window_size, features_normalized.shape[1]))
            actual_len = end - start
            window_data[:actual_len] = features_normalized[start:end]
            
            # 預測
            X = window_data.reshape(1, -1, window_data.shape[1])
            preds = model.predict(X, verbose=0)
            
            # 累積預測結果
            all_predictions[start:end] += preds[0][:actual_len]
            prediction_counts[start:end] += 1
            
            # 顯示進度
            if start % (stride * 4) == 0:
                progress = min(100, int(end / n_frames * 100))
                print(f"  進度: {progress}%", end='\r')
        
        print(f"  進度: 100%")
        
        # 平均多次預測結果
        all_predictions = all_predictions / prediction_counts[:, np.newaxis]
        
        pred_labels = np.argmax(all_predictions, axis=-1)
        predictions_prob = all_predictions
    
    # 統計預測分布
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
        # 計算準確率
        accuracy = accuracy_score(true_labels, pred_labels)
        print(f"\n整體準確率: {accuracy:.4f}")
        
        # 分類報告
        print("\n分類報告:")
        print(classification_report(true_labels, pred_labels, 
                                   target_names=config.LABEL_NAMES, 
                                   zero_division=0))
        
        # 混淆矩陣
        cm = confusion_matrix(true_labels, pred_labels)
        print("\n混淆矩陣:")
        print("           預測")
        print("         ", "  ".join([f"{name:8s}" for name in config.LABEL_NAMES]))
        for i, name in enumerate(config.LABEL_NAMES):
            print(f"{name:8s}  " + "  ".join([f"{cm[i][j]:8d}" for j in range(3)]))
    
    # 生成視覺化
    print("\n生成視覺化報告...")
    
    if has_ground_truth:
        fig = plt.figure(figsize=(18, 16))
        gs = fig.add_gridspec(6, 2, hspace=0.35, wspace=0.3)
    else:
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    max_frames = min(500, len(pred_labels))
    
    # 1. 腳踝軌跡
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(features[:max_frames, 1], 'b-', label='Left Ankle Y', alpha=0.7, lw=1.5)
    ax1.plot(features[:max_frames, 3], 'r-', label='Right Ankle Y', alpha=0.7, lw=1.5)
    ax1.set_ylabel('Y Position', fontsize=12)
    ax1.set_title('Ankle Trajectories (Test Data)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, max_frames])
    
    if has_ground_truth:
        # 2. Ground Truth
        ax2 = fig.add_subplot(gs[1, :])
        colors = ['gray', 'green', 'blue']
        for i, label_name in enumerate(config.LABEL_NAMES):
            mask = true_labels[:max_frames] == i
            if np.any(mask):
                frames = np.where(mask)[0]
                ax2.scatter(frames, [1]*len(frames), c=colors[i], 
                           label=label_name, s=20, alpha=0.7)
        ax2.set_ylabel('Ground Truth', fontsize=12)
        ax2.set_title('Ground Truth Labels', fontsize=14, fontweight='bold')
        ax2.set_yticks([])
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.set_xlim([0, max_frames])
    
    # 3. 預測結果
    ax3 = fig.add_subplot(gs[2 if has_ground_truth else 1, :])
    colors = ['gray', 'green', 'blue']
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
    
    if has_ground_truth:
        # 4. 預測信心度(前500幀)
        ax4 = fig.add_subplot(gs[3, :])
        for i, label_name in enumerate(config.LABEL_NAMES):
            ax4.plot(predictions_prob[:max_frames, i], 
                    label=label_name, alpha=0.7, lw=1.5)
        ax4.set_xlabel('Frame Index', fontsize=12)
        ax4.set_ylabel('Confidence', fontsize=12)
        ax4.set_title('Prediction Confidence', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim([0, max_frames])
        ax4.set_ylim([0, 1])
        
        # 5. 混淆矩陣
        ax5 = fig.add_subplot(gs[4, :])
        im = ax5.imshow(cm, cmap='Blues', aspect='auto')
        ax5.set_xticks(range(3))
        ax5.set_yticks(range(3))
        ax5.set_xticklabels(config.LABEL_NAMES)
        ax5.set_yticklabels(config.LABEL_NAMES)
        ax5.set_xlabel('Predicted', fontsize=12)
        ax5.set_ylabel('True', fontsize=12)
        ax5.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        for i in range(3):
            for j in range(3):
                color = "white" if cm[i, j] > cm.max()/2 else "black"
                text = ax5.text(j, i, f'{cm[i, j]}', 
                              ha="center", va="center", color=color, fontsize=14)
        plt.colorbar(im, ax=ax5)
        
        # 6. 每類別的Recall和Precision
        ax6 = fig.add_subplot(gs[5, 0])
        from sklearn.metrics import precision_score, recall_score
        
        precisions = []
        recalls = []
        for i in range(3):
            prec = precision_score(true_labels, pred_labels, labels=[i], average=None, zero_division=0)
            rec = recall_score(true_labels, pred_labels, labels=[i], average=None, zero_division=0)
            precisions.append(prec[0] if len(prec) > 0 else 0)
            recalls.append(rec[0] if len(rec) > 0 else 0)
        
        x = np.arange(3)
        width = 0.35
        ax6.bar(x - width/2, precisions, width, label='Precision', alpha=0.8)
        ax6.bar(x + width/2, recalls, width, label='Recall', alpha=0.8)
        ax6.set_xlabel('Class', fontsize=12)
        ax6.set_ylabel('Score', fontsize=12)
        ax6.set_title('Precision & Recall by Class', fontsize=13, fontweight='bold')
        ax6.set_xticks(x)
        ax6.set_xticklabels(config.LABEL_NAMES)
        ax6.legend()
        ax6.set_ylim([0, 1])
        ax6.grid(True, alpha=0.3, axis='y')
        
        # 7. 錯誤分析
        ax7 = fig.add_subplot(gs[5, 1])
        errors = (true_labels != pred_labels).astype(int)
        error_rate_window = 50
        error_rates = []
        for i in range(0, len(errors) - error_rate_window, error_rate_window):
            error_rates.append(np.mean(errors[i:i+error_rate_window]))
        
        ax7.plot(range(0, len(errors) - error_rate_window, error_rate_window), 
                error_rates, 'r-', lw=2)
        ax7.set_xlabel('Frame Index', fontsize=12)
        ax7.set_ylabel('Error Rate', fontsize=12)
        ax7.set_title(f'Error Rate (window={error_rate_window})', fontsize=13, fontweight='bold')
        ax7.grid(True, alpha=0.3)
        ax7.set_ylim([0, 1])
    
    return fig, cm if has_ground_truth else None


# =========================
# 儲存預測結果
# =========================
def save_predictions(df, pred_labels, output_dir):
    """儲存預測結果為CSV"""
    
    result_df = df.copy()
    result_df['predicted_label'] = pred_labels
    result_df['predicted_label_name'] = [config.LABEL_NAMES[l] for l in pred_labels]
    
    output_path = os.path.join(output_dir, 'predictions.csv')
    result_df.to_csv(output_path, index=False)
    print(f"✓ 預測結果已儲存: {output_path}")
    
    return output_path


# =========================
# 主程式
# =========================
def main():
    print("\n" + "="*60)
    print("模型測試程式")
    print("="*60)
    
    # 1. 載入模型
    print(f"\n載入模型: {config.MODEL_PATH}")
    if not os.path.exists(config.MODEL_PATH):
        print(f"❌ 模型不存在: {config.MODEL_PATH}")
        return
    
    model = keras.models.load_model(config.MODEL_PATH, compile=False)
    print("✓ 模型載入成功")
    
    # 2. 載入測試資料
    df, true_labels = load_test_labels(config.TEST_CSV, config.TEST_SEGMENTS_DIR)
    
    if df is None:
        print("❌ 測試資料載入失敗")
        return
    
    # 3. 提取特徵
    print("\n提取特徵...")
    features = extract_features(df)
    print(f"✓ 特徵形狀: {features.shape}")
    
    # 4. 預測
    pred_labels, predictions_prob = predict_with_model(model, features)
    
    # 5. 評估與視覺化
    fig, cm = evaluate_and_visualize(df, features, true_labels, pred_labels, predictions_prob)
    
    # 6. 儲存結果
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # 儲存視覺化
    fig_path = os.path.join(config.OUTPUT_DIR, 'test_results.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 視覺化報告已儲存: {fig_path}")
    
    # 儲存預測CSV
    save_predictions(df, pred_labels, config.OUTPUT_DIR)
    
    # 儲存混淆矩陣(如果有)
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
    print(f"\n結果儲存在: {config.OUTPUT_DIR}")
    print("包含:")
    print("  - test_results.png (完整視覺化報告)")
    print("  - predictions.csv (每一幀的預測結果)")
    if cm is not None:
        print("  - confusion_matrix.csv (混淆矩陣)")

if __name__ == "__main__":
    main()