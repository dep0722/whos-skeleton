"""
K-Medoids + DTW 訓練腳本（GPU 加速版）
對齊 exp_cosine_pyramid_train.py 的輸入格式

GPU 加速原理：
  距離矩陣計算是訓練最耗時的步驟（N×N 對）
  使用 PyTorch 向量化：一次算出一個樣本對所有其他樣本的 DTW 距離
  DTW DP 在 GPU 上批次執行，速度比純 numpy 快 10~50 倍

pip install torch numpy pandas scikit-learn matplotlib scipy
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# =====================================================
# 常數（與 TCNN 一致）
# =====================================================
JOINTS     = ["RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle"]
T_RESAMPLE = 40

FEATURE_WEIGHTS = np.array([1.0] * 12 + [2.0, 2.0], dtype=np.float32)
FEATURE_WEIGHTS /= FEATURE_WEIGHTS.sum()


# =====================================================
# 特徵萃取（CPU，只做一次）
# =====================================================
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
    """CSV → (T, 14) 特徵矩陣，失敗回傳 None"""
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
# GPU 向量化 DTW 距離矩陣
# =====================================================
def compute_distance_matrix_gpu(feats, weights, device):
    """
    計算類別內所有樣本兩兩 DTW 距離矩陣（GPU 加速）

    feats:   list of N 個 (T, D) numpy array
    weights: (D,) numpy array
    回傳:    (N, N) numpy float32 距離矩陣

    加速原理：
      每次取一個 query，一次計算它對所有 N 個樣本的距離
      14 個維度序列化，但每個維度的 DTW DP 是對 N 個樣本同時向量化執行
    """
    N = len(feats)
    T = feats[0].shape[0]
    D = feats[0].shape[1]

    w = torch.tensor(weights, dtype=torch.float32, device=device)  # (D,)

    # 預先把所有特徵上傳 GPU 並取導數
    # bank_d: (N, T, D) 導數
    bank = torch.tensor(np.stack(feats, axis=0), dtype=torch.float32, device=device)  # (N,T,D)
    bank_d = torch.gradient(bank, dim=1)[0]  # (N, T, D)

    dist_mat = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        q      = bank[i]    # (T, D)
        q_d    = bank_d[i]  # (T, D)

        total_dist = torch.zeros(N, device=device)

        for d in range(D):
            qs = q_d[:, d]         # (T,)
            ts = bank_d[:, :, d]   # (N, T)

            # cost[n, i, j] = |ts[n,i] - qs[j]|  → (N, T, T)
            cost = torch.abs(ts.unsqueeze(2) - qs.unsqueeze(0).unsqueeze(0))

            # DTW DP：N 個樣本平行計算
            dtw_g = torch.full((N, T, T), float('inf'), device=device)
            dtw_g[:, 0, 0] = cost[:, 0, 0]
            for ii in range(1, T):
                dtw_g[:, ii, 0] = dtw_g[:, ii-1, 0] + cost[:, ii, 0]
            for jj in range(1, T):
                dtw_g[:, 0, jj] = dtw_g[:, 0, jj-1] + cost[:, 0, jj]
            for ii in range(1, T):
                for jj in range(1, T):
                    prev = torch.minimum(
                        torch.minimum(dtw_g[:, ii-1, jj], dtw_g[:, ii, jj-1]),
                        dtw_g[:, ii-1, jj-1]
                    )
                    dtw_g[:, ii, jj] = cost[:, ii, jj] + prev

            total_dist += (dtw_g[:, T-1, T-1] / (2 * T)) * w[d]

        dist_mat[i] = total_dist.cpu().numpy()

        if (i + 1) % 10 == 0 or i == N - 1:
            print(f"    距離矩陣進度：{i+1}/{N}", end='\r')

    print()

    # 對稱化（消除浮點誤差造成的微小不對稱）
    dist_mat = (dist_mat + dist_mat.T) / 2
    np.fill_diagonal(dist_mat, 0)
    return dist_mat


# =====================================================
# K-Medoids PAM
# =====================================================
def k_medoids(dist_mat, k, max_iter=100, random_state=42):
    """
    K-Medoids（PAM），使用預先計算的距離矩陣
    回傳: medoid 索引 list（長度 k）
    """
    np.random.seed(random_state)
    N = dist_mat.shape[0]
    medoids = list(np.random.choice(N, k, replace=False))

    for iteration in range(max_iter):
        # 指派
        assignments = np.argmin(dist_mat[:, medoids], axis=1).tolist()

        # 更新
        new_medoids = []
        changed = False
        for k_idx in range(k):
            members = [i for i, a in enumerate(assignments) if a == k_idx]
            if len(members) == 0:
                new_medoids.append(medoids[k_idx])
                continue

            sub = dist_mat[np.ix_(members, members)]
            best_local = int(np.argmin(sub.sum(axis=1)))
            best_global = members[best_local]
            new_medoids.append(best_global)
            if best_global != medoids[k_idx]:
                changed = True

        medoids = new_medoids
        if not changed:
            print(f"    K-Medoids 收斂於第 {iteration+1} 次迭代")
            break

    return medoids


# =====================================================
# 繪圖：medoid 膝角曲線
# =====================================================
def plot_medoid_curves(medoid_feats, medoid_labels, class_names, save_dir):
    n_classes = len(class_names)
    fig, axes = plt.subplots(n_classes, 1, figsize=(10, 3 * n_classes))
    if n_classes == 1:
        axes = [axes]

    class_medoids = {c: [] for c in range(n_classes)}
    for feat, label in zip(medoid_feats, medoid_labels):
        class_medoids[label].append(feat)

    for cls_idx, cls_name in enumerate(class_names):
        ax = axes[cls_idx]
        for k_idx, feat in enumerate(class_medoids[cls_idx]):
            ax.plot(feat[:, 12], label=f'Medoid {k_idx+1} - Right Knee', linestyle='-')
            ax.plot(feat[:, 13], label=f'Medoid {k_idx+1} - Left Knee',  linestyle='--')
        ax.set_title(f'{cls_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Knee Angle (z-scored)')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle('Medoid Knee Angle Curves by Class', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'medoid_knee_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# =====================================================
# 主程式
# =====================================================
if __name__ == "__main__":
    # ===== 配置 =====
    DATA_DIR   = r"C:\mydata\sf\open\output_csv\0219_train"
    SAVE_PATH  = r"C:\mydata\sf\comp\model\DTW\kmedoids_dtw_model.pkl"
    SAVE_DIR   = r"C:\mydata\sf\comp\model\DTW"
    K          = 3      # 每個類別保留幾個 medoid（建議 1~5）
    DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
    # ================

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"使用裝置：{DEVICE}")
    if DEVICE == 'cuda':
        print(f"GPU：{torch.cuda.get_device_name(0)}")

    class_names = sorted([
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ])
    class2idx = {c: i for i, c in enumerate(class_names)}
    print(f"Classes: {class_names}")
    print(f"K (每類 medoid 數)：{K}\n")

    medoid_feats  = []
    medoid_labels = []

    for cls in class_names:
        cls_idx   = class2idx[cls]
        cls_path  = os.path.join(DATA_DIR, cls)
        csv_files = collect_csv_recursive(cls_path)
        print(f"[{cls}] 找到 {len(csv_files)} 個 CSV")

        feats = []
        for fp in csv_files:
            feat = load_features(fp)
            if feat is not None:
                feats.append(feat)

        print(f"  成功載入：{len(feats)} 筆")

        if len(feats) == 0:
            print(f"  [警告] {cls} 無有效樣本，跳過\n")
            continue

        k_actual = min(K, len(feats))

        if len(feats) <= k_actual:
            print(f"  樣本數 {len(feats)} 筆 ≤ K，直接全部使用為模板")
            for f in feats:
                medoid_feats.append(f)
                medoid_labels.append(cls_idx)
        else:
            print(f"  計算 {len(feats)}×{len(feats)} GPU 距離矩陣...")
            dist_mat = compute_distance_matrix_gpu(feats, FEATURE_WEIGHTS, DEVICE)

            print(f"  執行 K-Medoids（k={k_actual}）...")
            medoid_indices = k_medoids(dist_mat, k=k_actual)

            for rank, idx in enumerate(medoid_indices):
                avg_dist = float(dist_mat[idx].mean())
                print(f"  Medoid {rank+1}：樣本索引={idx}，類別內平均距離={avg_dist:.4f}")
                medoid_feats.append(feats[idx])
                medoid_labels.append(cls_idx)

        print()

    print(f"總模板數：{len(medoid_feats)}")

    plot_medoid_curves(medoid_feats, medoid_labels, class_names, SAVE_DIR)

    model_data = {
        "medoid_feats"    : medoid_feats,
        "medoid_labels"   : medoid_labels,
        "class_names"     : class_names,
        "class2idx"       : class2idx,
        "feature_weights" : FEATURE_WEIGHTS,
        "t_resample"      : T_RESAMPLE,
        "joints"          : JOINTS,
        "k"               : K,
    }

    with open(SAVE_PATH, "wb") as f:
        pickle.dump(model_data, f)

    print(f"\n模型已儲存：{SAVE_PATH}")

    label_arr = np.array(medoid_labels)
    print("\n模板分布：")
    for cls in class_names:
        count = (label_arr == class2idx[cls]).sum()
        print(f"  {cls}: {count} 個 medoid")