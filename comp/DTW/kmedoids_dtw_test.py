"""
K-Medoids + DTW 測試腳本
對齊 exp_cosine_pyramid_test.py 的輸出風格

載入訓練腳本產生的 .pkl，對每筆測試樣本做
1-NN DTW（只比對 medoid 模板，速度遠快於比對全部訓練樣本）
輸出：混淆矩陣、分類指標圖、results.txt

pip install numpy pandas scikit-learn matplotlib seaborn scipy
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
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
    old_x = np.linspace(0, 1, len(series))
    new_x = np.linspace(0, 1, length)
    return interp1d(old_x, series, kind='linear')(new_x)


def zscore(series):
    std = np.std(series)
    return (series - np.mean(series)) / std if std > 0 else series - np.mean(series)


def load_features(file_path):
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"  [讀檔失敗] {os.path.basename(file_path)}: {e}")
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
# 純 numpy DTW
# =====================================================
def _dtw_1d(s1, s2):
    n, m = len(s1), len(s2)
    dtw_mat = np.full((n, m), np.inf)
    dtw_mat[0, 0] = abs(s1[0] - s2[0])
    for i in range(1, n):
        dtw_mat[i, 0] = dtw_mat[i-1, 0] + abs(s1[i] - s2[0])
    for j in range(1, m):
        dtw_mat[0, j] = dtw_mat[0, j-1] + abs(s1[0] - s2[j])
    for i in range(1, n):
        for j in range(1, m):
            dtw_mat[i, j] = abs(s1[i] - s2[j]) + min(
                dtw_mat[i-1, j], dtw_mat[i, j-1], dtw_mat[i-1, j-1]
            )
    return dtw_mat[n-1, m-1] / (n + m)


def dtw_distance(feat_a, feat_b, weights):
    distances = []
    for dim in range(feat_a.shape[1]):
        ds1 = np.gradient(feat_a[:, dim])
        ds2 = np.gradient(feat_b[:, dim])
        distances.append(_dtw_1d(ds1, ds2))
    return float(np.dot(distances, weights))


def predict_one(feat_query, medoid_feats, medoid_labels, class_names, weights):
    """
    1-NN DTW：找最近的 medoid，回傳類別和各類別最小距離
    同時支援軟分類（取各類 medoid 中最小距離做類別分數）
    """
    # 各類別最小距離
    n_classes = len(class_names)
    class_min_dist = np.full(n_classes, np.inf)

    for feat_m, label_m in zip(medoid_feats, medoid_labels):
        d = dtw_distance(feat_query, feat_m, weights)
        if d < class_min_dist[label_m]:
            class_min_dist[label_m] = d

    pred_label = int(np.argmin(class_min_dist))
    return pred_label, class_min_dist


# =====================================================
# 繪圖函式（對齊 exp_cosine_pyramid_test.py）
# =====================================================
def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix (K-Medoids + DTW)', fontsize=16, fontweight='bold')
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
    ax.set_title('Classification Metrics (K-Medoids + DTW)', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_distance_heatmap(y_true, all_dists, class_names, save_path):
    """
    繪製每筆測試樣本對各類別的 DTW 距離熱圖
    y_true: list of int
    all_dists: (N_test, N_classes) array
    """
    # 依真實類別排序
    sort_idx = np.argsort(y_true)
    sorted_dists = all_dists[sort_idx]
    sorted_labels = np.array(y_true)[sort_idx]

    plt.figure(figsize=(max(8, len(class_names) * 1.5), 8))
    sns.heatmap(sorted_dists, cmap='YlOrRd_r',
                xticklabels=class_names,
                yticklabels=False,
                cbar_kws={'label': 'DTW Distance (lower = closer)'})

    # 標示真實類別邊界
    boundaries = [0]
    for c in range(len(class_names)):
        boundaries.append(int((sorted_labels == c).sum()) + boundaries[-1])
    for b in boundaries[1:-1]:
        plt.axhline(b, color='blue', linewidth=1.5)

    plt.title('DTW Distance to Each Class Prototype\n(sorted by true label, blue lines = class boundaries)',
              fontsize=13, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('Test Samples (sorted by true label)', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# =====================================================
# 主程式
# =====================================================
if __name__ == "__main__":
    # ===== 配置 =====
    MODEL_PATH    = r"C:\mydata\sf\comp\model\DTW\kmedoids_dtw_model.pkl"
    TEST_DATA_DIR = r"C:\mydata\sf\open\output_csv\0219_test"
    OUTPUT_DIR    = r"C:\mydata\sf\result_p\dtw_kmedoids"
    # ================

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 載入模型
    print(f"載入模型：{MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)

    medoid_feats    = model_data["medoid_feats"]
    medoid_labels   = model_data["medoid_labels"]
    class_names     = model_data["class_names"]
    class2idx       = model_data["class2idx"]
    feature_weights = model_data["feature_weights"]
    k               = model_data["k"]

    print(f"Classes: {class_names}")
    print(f"K={k}，總模板數：{len(medoid_feats)}")

    label_arr = np.array(medoid_labels)
    for cls in class_names:
        count = (label_arr == class2idx[cls]).sum()
        print(f"  {cls}: {count} 個 medoid")

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

    # 推論
    print(f"\n開始推論（K-Medoids 1-NN DTW，模板數={len(medoid_feats)}）...")
    y_pred    = []
    all_dists = []

    for i, feat_q in enumerate(test_features):
        pred_label, class_dists = predict_one(
            feat_q, medoid_feats, medoid_labels, class_names, feature_weights
        )
        y_pred.append(pred_label)
        all_dists.append(class_dists)

        fname    = os.path.basename(test_files[i])
        true_cls = class_names[test_labels[i]]
        pred_cls = class_names[pred_label]
        mark     = "✔" if pred_label == test_labels[i] else "✘"
        dist_str = " | ".join([f"{class_names[c]}:{class_dists[c]:.3f}"
                                for c in range(len(class_names))])
        print(f"  [{i+1:3d}/{len(test_features)}] {mark} {fname:<30} "
              f"真實:{true_cls:<12} 預測:{pred_cls:<12} [{dist_str}]")

    all_dists = np.array(all_dists)  # (N_test, N_classes)

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
                          os.path.join(OUTPUT_DIR, "confusion_matrix_kmedoids.png"))
    plot_metrics(test_labels, y_pred, class_names,
                 os.path.join(OUTPUT_DIR, "metrics_kmedoids.png"))
    plot_distance_heatmap(test_labels, all_dists, class_names,
                          os.path.join(OUTPUT_DIR, "distance_heatmap_kmedoids.png"))

    # 儲存文字結果
    results_txt = os.path.join(OUTPUT_DIR, "results_kmedoids.txt")
    with open(results_txt, "w", encoding="utf-8") as f:
        f.write(f"Model: K-Medoids + DTW (K={k})\n")
        f.write(f"總模板數：{len(medoid_feats)}\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"Saved: {results_txt}")

    print("\nDone!")
