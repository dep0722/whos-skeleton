"""
DTW 測試腳本 - GPU 加速版
使用 PyTorch 向量化 DTW，將測試樣本與所有訓練樣本批次比對
加速原理：把 14 個維度的 DTW cost matrix 同時在 GPU 上計算，
          取代原本 Python 層的雙層 for 迴圈

pip install torch numpy pandas scikit-learn matplotlib seaborn
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_recall_fscore_support
)



# =====================================================
# 特徵萃取（與訓練腳本完全相同）
# =====================================================
JOINTS     = ["RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle"]
T_RESAMPLE = 40


def compute_knee_angle(hip, knee, ankle):
    v1 = hip - knee
    v2 = ankle - knee
    dot  = np.sum(v1 * v2, axis=1)
    norm = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    norm = np.maximum(norm, 1e-8)
    return np.arccos(np.clip(dot / norm, -1.0, 1.0))


def resample(series, length=T_RESAMPLE):
    if len(series) == length:
        return series.copy()
    t_orig = np.linspace(0, 1, len(series))
    t_new  = np.linspace(0, 1, length)
    return np.interp(t_new, t_orig, series)


def zscore(series):
    std = np.std(series)
    return (series - np.mean(series)) / std if std > 0 else series - np.mean(series)


def load_features(file_path):
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"  [讀檔失敗] {file_path}: {e}")
        return None

    node_xy = {}
    for joint in JOINTS:
        xc, yc = f"{joint}_x", f"{joint}_y"
        if xc not in df.columns or yc not in df.columns:
            print(f"  [缺欄位] {xc}/{yc}：{os.path.basename(file_path)}")
            return None

        x = df[xc].astype(float).values
        y = df[yc].astype(float).values
        mask = (x > 0) & (y > 0)
        if mask.sum() < 5:
            return None

        x[~mask] = x[mask].mean()
        y[~mask] = y[mask].mean()
        node_xy[joint] = np.stack([resample(x), resample(y)], axis=1)

    right_angle = compute_knee_angle(node_xy["RHip"], node_xy["RKnee"], node_xy["RAnkle"])
    left_angle  = compute_knee_angle(node_xy["LHip"], node_xy["LKnee"], node_xy["LAnkle"])

    xy_feat    = np.concatenate([node_xy[j] for j in JOINTS], axis=1)
    angle_feat = np.stack([right_angle, left_angle], axis=1)
    features   = np.concatenate([xy_feat, angle_feat], axis=1).astype(np.float32)
    features   = np.apply_along_axis(zscore, 0, features)
    return features


def collect_csv_recursive(directory):
    csv_files = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith(".csv"):
                csv_files.append(os.path.join(root, f))
    return sorted(csv_files)


# =====================================================
# GPU 向量化 DTW
# =====================================================
def batch_dtw_gpu(query, train_bank, weights, device):
    """
    一次計算單一 query 對所有訓練樣本的加權 DTW 距離

    query:      (T, D)         - 單筆測試特徵
    train_bank: (N, T, D)      - 所有訓練樣本
    weights:    (D,)           - 各維度權重
    回傳:       (N,) distances

    做法：
      1. 先對所有序列取一階導數（derivative DTW）
      2. 對每個維度 d，同時計算 query[d] vs train_bank[:, :, d] 的 DTW
         → 利用 GPU 平行計算 N 條序列
      3. 加權求和得到最終距離
    """
    T, D = query.shape
    N    = train_bank.shape[0]

    # 移至 GPU，取一階導數
    # query_d:  (T, D)
    # train_d:  (N, T, D)
    q = torch.tensor(query, dtype=torch.float32, device=device)
    t = torch.tensor(train_bank, dtype=torch.float32, device=device)
    w = torch.tensor(weights, dtype=torch.float32, device=device)

    # 一階導數（沿時間軸 dim=0 / dim=1）
    q_d = torch.gradient(q, dim=0)[0]        # (T, D)
    t_d = torch.gradient(t, dim=1)[0]        # (N, T, D)

    # 對每個維度做向量化 DTW
    total_dist = torch.zeros(N, device=device)

    for d in range(D):
        qs = q_d[:, d]          # (T,)
        ts = t_d[:, :, d]       # (N, T)

        # cost matrix: (N, T, T)  — ts[n,i] vs qs[j]
        # cost[n, i, j] = |ts[n,i] - qs[j]|
        cost = torch.abs(ts.unsqueeze(2) - qs.unsqueeze(0).unsqueeze(0))
        # cost shape: (N, T, T)

        # DTW DP，沿對角線推進
        # 用 cumsum 近似：逐列更新 DTW matrix
        dtw_mat = torch.full((N, T, T), float('inf'), device=device)
        dtw_mat[:, 0, 0] = cost[:, 0, 0]

        for i in range(1, T):
            dtw_mat[:, i, 0] = dtw_mat[:, i-1, 0] + cost[:, i, 0]
        for j in range(1, T):
            dtw_mat[:, 0, j] = dtw_mat[:, 0, j-1] + cost[:, 0, j]
        for i in range(1, T):
            for j in range(1, T):
                prev = torch.minimum(
                    torch.minimum(dtw_mat[:, i-1, j], dtw_mat[:, i, j-1]),
                    dtw_mat[:, i-1, j-1]
                )
                dtw_mat[:, i, j] = cost[:, i, j] + prev

        # normalized distance
        dist_d = dtw_mat[:, T-1, T-1] / (2 * T)
        total_dist += dist_d * w[d]

    return total_dist.cpu().numpy()


# =====================================================
# 繪圖函式（與 gtcn_test_6node.py 風格一致）
# =====================================================
def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_metrics(y_true, y_pred, class_names, save_path):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(len(class_names)))
    )
    x     = np.arange(len(class_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label='Precision', color='#3498db')
    ax.bar(x,         recall,    width, label='Recall',    color='#2ecc71')
    ax.bar(x + width, f1,        width, label='F1-Score',  color='#e74c3c')
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Classification Metrics by Class', fontsize=16, fontweight='bold')
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
    counts = [(np.array(y_true) == i).sum() for i in range(len(class_names))]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(class_names)), counts, color='#9b59b6', edgecolor='black')
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., h,
                 f'{int(h)}', ha='center', va='bottom', fontsize=10)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title('Sample Distribution', fontsize=16, fontweight='bold')
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# =====================================================
# 主程式
# =====================================================
if __name__ == "__main__":
    # ===== 配置 =====
    MODEL_PATH    = r"C:\mydata\sf\comp\model\DTW\dtw_6node_model.pkl"
    TEST_DATA_DIR = r"C:\mydata\sf\open\output_csv\0219_test"
    OUTPUT_DIR    = r"C:\mydata\sf\comp\DTW_6node_10min"
    DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'
    # ================

    print(f"使用裝置：{DEVICE}")
    if DEVICE == 'cuda':
        print(f"GPU：{torch.cuda.get_device_name(0)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 載入模型
    print(f"\n載入模型：{MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)

    train_features  = model_data["train_features"]   # list of (T, 14)
    train_labels    = model_data["train_labels"]
    class_names     = model_data["class_names"]
    class2idx       = model_data["class2idx"]
    feature_weights = model_data["feature_weights"]  # (14,)

    print(f"Classes: {class_names}")
    print(f"訓練樣本：{len(train_features)} 筆")

    # 將訓練樣本堆疊成 (N, T, D) tensor，一次上傳 GPU
    train_bank = np.stack(train_features, axis=0)  # (N, T, 14)
    train_labels_arr = np.array(train_labels)

    # 載入測試資料
    print(f"\n載入測試資料：{TEST_DATA_DIR}")
    test_features = []
    test_labels   = []
    test_files    = []

    for cls in class_names:
        cls_path = os.path.join(TEST_DATA_DIR, cls)
        if not os.path.isdir(cls_path):
            print(f"  [警告] 測試資料夾不存在：{cls_path}")
            continue

        csv_files = collect_csv_recursive(cls_path)
        print(f"  [{cls}] 找到 {len(csv_files)} 個 CSV")

        for fp in csv_files:
            feat = load_features(fp)
            if feat is not None:
                test_features.append(feat)
                test_labels.append(class2idx[cls])
                test_files.append(fp)

    print(f"\n測試樣本：{len(test_features)} 筆")

    if len(test_features) == 0:
        print("錯誤：沒有成功載入任何測試樣本")
        exit(1)

    # GPU 1-NN DTW 推論
    print(f"\n開始推論（GPU 1-NN DTW，device={DEVICE}）...")
    y_pred = []

    for i, feat_q in enumerate(test_features):
        # batch_dtw_gpu 一次算出 query 對所有訓練樣本的距離 (N,)
        dists = batch_dtw_gpu(feat_q, train_bank, feature_weights, DEVICE)

        best_idx   = int(np.argmin(dists))
        pred_label = int(train_labels_arr[best_idx])
        best_dist  = float(dists[best_idx])
        y_pred.append(pred_label)

        fname    = os.path.basename(test_files[i])
        true_cls = class_names[test_labels[i]]
        pred_cls = class_names[pred_label]
        mark     = "✔" if pred_label == test_labels[i] else "✘"
        print(f"  [{i+1:3d}/{len(test_features)}] {mark} {fname:<30} "
              f"真實:{true_cls:<12} 預測:{pred_cls:<12} dist={best_dist:.4f}")

    # 整體準確率
    accuracy = accuracy_score(test_labels, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f}")

    # 分類報告
    print("\nClassification Report:")
    report = classification_report(test_labels, y_pred, target_names=class_names, digits=4)
    print(report)

    # 繪製圖表
    print("\nGenerating plots...")
    plot_confusion_matrix(test_labels, y_pred, class_names,
                          os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plot_metrics(test_labels, y_pred, class_names,
                 os.path.join(OUTPUT_DIR, "metrics_by_class.png"))
    plot_support(test_labels, class_names,
                 os.path.join(OUTPUT_DIR, "sample_distribution.png"))

    # 儲存文字結果
    results_txt = os.path.join(OUTPUT_DIR, "results.txt")
    with open(results_txt, "w", encoding="utf-8") as f:
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"Saved: {results_txt}")

    print("\nDone!")