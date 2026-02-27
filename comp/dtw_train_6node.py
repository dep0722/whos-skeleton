"""
DTW 訓練腳本（模板建立）- 對齊 GTCN 輸入格式
使用 6 個下半身關節：RHip, RKnee, RAnkle, LHip, LKnee, LAnkle
特徵：6 關節 xy + 左右膝角（共 14 維）
輸出：.pkl 模型檔（儲存所有訓練樣本特徵供 1-NN DTW 測試使用）

資料結構：
  DATA_DIR/
    類別A/
      xxx.csv
      子資料夾/xxx.csv   ← 支援多層子資料夾
    類別B/
      ...

pip install numpy pandas scikit-learn dtw-python
"""
import os
import pickle
import numpy as np
import pandas as pd
from dtw import dtw

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# =====================================================
# 常數（與 GTCN 一致）
# =====================================================
JOINTS      = ["RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle"]
T_RESAMPLE  = 40

# 各維度權重：前 12 維為關節 xy，後 2 維為膝角（加倍權重）
FEATURE_WEIGHTS = np.array([1.0] * 12 + [2.0, 2.0], dtype=np.float32)
FEATURE_WEIGHTS /= FEATURE_WEIGHTS.sum()


# =====================================================
# 特徵萃取（與 GTCN compute_knee_angle 完全一致）
# =====================================================
def compute_knee_angle(hip, knee, ankle):
    """hip/knee/ankle: (T, 2)  →  回傳 (T,)"""
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
    """
    讀取單一 CSV → (T, 14) 特徵矩陣
    欄位：[RHip_x, RHip_y, RKnee_x, ..., LAnkle_y, RKnee_angle, LKnee_angle]
    失敗回傳 None
    """
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
            print(f"  [有效幀不足] {os.path.basename(file_path)}")
            return None

        x[~mask] = x[mask].mean()
        y[~mask] = y[mask].mean()

        node_xy[joint] = np.stack([resample(x), resample(y)], axis=1)  # (T, 2)

    right_angle = compute_knee_angle(node_xy["RHip"], node_xy["RKnee"], node_xy["RAnkle"])
    left_angle  = compute_knee_angle(node_xy["LHip"], node_xy["LKnee"], node_xy["LAnkle"])

    xy_feat    = np.concatenate([node_xy[j] for j in JOINTS], axis=1)       # (T, 12)
    angle_feat = np.stack([right_angle, left_angle], axis=1)                 # (T, 2)
    features   = np.concatenate([xy_feat, angle_feat], axis=1).astype(np.float32)  # (T, 14)

    # 對每個維度做 Z-score
    features = np.apply_along_axis(zscore, 0, features)
    return features


def collect_csv_recursive(directory):
    csv_files = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith(".csv"):
                csv_files.append(os.path.join(root, f))
    return sorted(csv_files)


# =====================================================
# 主程式
# =====================================================
if __name__ == "__main__":
    # ===== 配置 =====
    DATA_DIR  = r"C:\mydata\sf\open\output_csv\0219_train"
    SAVE_PATH = r"C:\mydata\sf\comp\model\DTW\dtw_6node_model.pkl"
    # ================

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    class_names = sorted([
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ])
    class2idx = {c: i for i, c in enumerate(class_names)}
    print(f"Classes: {class_names}")

    # 收集所有訓練樣本特徵
    train_features = []   # list of (T, 14)
    train_labels   = []   # list of int

    total_loaded = 0
    total_failed = 0

    for cls in class_names:
        cls_path  = os.path.join(DATA_DIR, cls)
        csv_files = collect_csv_recursive(cls_path)
        print(f"\n  [{cls}] 找到 {len(csv_files)} 個 CSV")

        for fp in csv_files:
            feat = load_features(fp)
            if feat is not None:
                train_features.append(feat)
                train_labels.append(class2idx[cls])
                total_loaded += 1
            else:
                total_failed += 1

    print(f"\n載入完成：{total_loaded} 筆成功 / {total_failed} 筆失敗")

    if total_loaded == 0:
        print("錯誤：沒有成功載入任何樣本，請確認 CSV 欄位格式")
        exit(1)

    # 計算類別分布
    label_arr = np.array(train_labels)
    print("\n類別分布：")
    for cls in class_names:
        idx   = class2idx[cls]
        count = (label_arr == idx).sum()
        print(f"  {cls}: {count} 筆")

    # 儲存模型（所有訓練樣本 + 元資料）
    model_data = {
        "train_features"  : train_features,   # list of (T, 14) np.float32
        "train_labels"    : train_labels,      # list of int
        "class_names"     : class_names,
        "class2idx"       : class2idx,
        "feature_weights" : FEATURE_WEIGHTS,
        "t_resample"      : T_RESAMPLE,
        "joints"          : JOINTS,
    }

    with open(SAVE_PATH, "wb") as f:
        pickle.dump(model_data, f)

    print(f"\n模型已儲存：{SAVE_PATH}")
    print(f"訓練樣本總數：{total_loaded}")
