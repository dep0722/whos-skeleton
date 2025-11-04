# RFCheck.py
# 即時用 RandomForest 做骨架分類（支援 weighted / normal pkl）
# 2025-11-02

import os
import cv2
import json
import joblib
import numpy as np
import pandas as pd
import mediapipe as mp

# ====== 使用者設定（已由你確認） ======
MODEL_PATH = r"C:/mydata/sf/conda/1025_test/test_model/RFmodel_1104_normal.pkl"
RF_CSV_PATH = r"C:/mydata/sf/conda/1025_test/test_csv/pose_1104.csv"
# 如果有單獨的 weight json 放在別處，程式會嘗試與 pkl 同資料夾找同名 json

# ====== 固定的 feature 欄位順序（你確認過的 14 欄） ======
FEATURE_COLUMNS = [
    "LEFT_SHOULDER_LEFT_ELBOW",
    "LEFT_ELBOW_LEFT_WRIST",
    "RIGHT_SHOULDER_RIGHT_ELBOW",
    "RIGHT_ELBOW_RIGHT_WRIST",
    "LEFT_SHOULDER_RIGHT_SHOULDER",
    "LEFT_HIP_RIGHT_HIP",
    "LEFT_SHOULDER_LEFT_HIP",
    "RIGHT_SHOULDER_RIGHT_HIP",
    "LEFT_HIP_LEFT_KNEE",
    "LEFT_KNEE_LEFT_ANKLE",
    "RIGHT_HIP_RIGHT_KNEE",
    "RIGHT_KNEE_RIGHT_ANKLE",
]

# ====== Mediapipe & edges mapping (same as訓練) ======
LM = {'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12,
    'LEFT_ELBOW': 13, 'RIGHT_ELBOW': 14,
    'LEFT_WRIST': 15, 'RIGHT_WRIST': 16,
    'LEFT_HIP': 23, 'RIGHT_HIP': 24,
    'LEFT_KNEE': 25, 'RIGHT_KNEE': 26,
    'LEFT_ANKLE': 27, 'RIGHT_ANKLE': 28,
    'LEFT_HEEL': 29, 'RIGHT_HEEL': 30,
}
EDGES = [
    ('LEFT_SHOULDER', 'LEFT_ELBOW'),
    ('LEFT_ELBOW', 'LEFT_WRIST'),
    ('RIGHT_SHOULDER', 'RIGHT_ELBOW'),
    ('RIGHT_ELBOW', 'RIGHT_WRIST'),
    ('LEFT_SHOULDER', 'RIGHT_SHOULDER'),
    ('LEFT_HIP', 'RIGHT_HIP'),
    ('LEFT_SHOULDER', 'LEFT_HIP'),
    ('RIGHT_SHOULDER', 'RIGHT_HIP'),
    ('LEFT_HIP', 'LEFT_KNEE'),
    ('LEFT_KNEE', 'LEFT_ANKLE'),
    ('RIGHT_HIP', 'RIGHT_KNEE'),
    ('RIGHT_KNEE', 'RIGHT_ANKLE')
]

# ====== 載入 model/scaler/weights（支援多種 pkl 結構） ======
print("Loading model:", MODEL_PATH)
obj = joblib.load(MODEL_PATH)

# initialize holders
rf_model = None
scaler_mean = None
scaler_scale = None
feature_columns = None
weights = None

# 判斷 pkl 物件型別
if isinstance(obj, dict):
    # new-style: dict 包含 keys
    rf_model = obj.get("model") or obj.get("rf") or obj.get("estimator")
    feature_columns = obj.get("feature_columns") or obj.get("columns") or FEATURE_COLUMNS
    # scaler 可能是 sklearn object or dict with mean_/scale_
    scaler_obj = obj.get("scaler")
    if scaler_obj is not None:
        if hasattr(scaler_obj, "mean_") and hasattr(scaler_obj, "scale_"):
            scaler_mean = np.array(scaler_obj.mean_)
            scaler_scale = np.array(scaler_obj.scale_)
        elif isinstance(scaler_obj, dict) and "mean_" in scaler_obj and "scale_" in scaler_obj:
            scaler_mean = np.array(scaler_obj["mean_"])
            scaler_scale = np.array(scaler_obj["scale_"])
    # weights may be list/np.array or dict mapping feature->weight
    w = obj.get("weights")
    if w is not None:
        if isinstance(w, dict):
            # convert to array following feature_columns order
            feature_columns = list(feature_columns)  # ensure list
            weights = np.array([w.get(col, 1.0) for col in feature_columns], dtype=float)
        else:
            weights = np.array(w, dtype=float)
else:
    # old-style: obj is estimator
    rf_model = obj

# fallback ensure rf_model exists
if rf_model is None:
    raise RuntimeError("Cannot find RF model in pkl. Please save model as estimator or dict{'model':...}.")

# feature columns fallback to fixed list if not provided
if feature_columns is None:
    feature_columns = FEATURE_COLUMNS
else:
    # standardize to list and ensure ordering matches FEATURE_COLUMNS if possible
    feature_columns = list(feature_columns)
    # if provided list differs in content, prefer fixed FEATURE_COLUMNS but warn
    if set(feature_columns) != set(FEATURE_COLUMNS):
        print("Warning: feature_columns from pkl differ from expected. Forcing expected FEATURE_COLUMNS order.")
        feature_columns = FEATURE_COLUMNS

print("Using feature columns (len={}):".format(len(feature_columns)))
print(feature_columns)

# detect weighted mode by filename
basename = os.path.basename(MODEL_PATH).lower()
weighted_intent = "_weighted" in basename or "_weithed" in basename  # accept both spellings

# if no weights loaded from pkl, try find same-base json in same folder
if weights is None and weighted_intent:
    alt_json = os.path.splitext(MODEL_PATH)[0] + ".json"
    if os.path.exists(alt_json):
        try:
            with open(alt_json, "r", encoding="utf-8") as f:
                wdict = json.load(f)
            # wdict could be {"feat":{"variance":..,"weight":..},...} or {"feat": weight}
            if isinstance(wdict, dict):
                # try parse mapping
                if all(isinstance(v, dict) for v in wdict.values()):
                    weights = np.array([wdict.get(col, {}).get("weight", 1.0) for col in feature_columns], dtype=float)
                else:
                    weights = np.array([wdict.get(col, 1.0) for col in feature_columns], dtype=float)
            else:
                weights = None
        except Exception as e:
            print("Failed to load alt json weights:", e)
    else:
        print("Weighted intent from filename but no weights found in pkl; no external json at:", alt_json)

use_weights = (weights is not None) and weighted_intent
print("Weighted intent by filename:", weighted_intent, "| weights available:", weights is not None, "| using weights:", use_weights)

# ====== 若沒有 scaler 資訊就用 CSV 計算 mean/std （並做 safe_scale） ======
if (scaler_mean is None) or (scaler_scale is None):
    if not os.path.exists(RF_CSV_PATH):
        raise FileNotFoundError(f"Scaler not in pkl and CSV not found: {RF_CSV_PATH}")
    print("Computing scaler mean/std from CSV:", RF_CSV_PATH)
    df_csv = pd.read_csv(RF_CSV_PATH)
    X_csv = df_csv[feature_columns]
    scaler_mean = X_csv.mean().values
    scaler_scale = X_csv.std().values
# safe scale: avoid zeros
safe_scale = np.where(scaler_scale == 0, 1.0, scaler_scale)

# if weights is array, ensure length matches feature_columns
if weights is not None:
    if len(weights) != len(feature_columns):
        print("Warning: weights length != feature columns length. Ignoring weights.")
        weights = None
    else:
        weights = np.array(weights, dtype=float)

# ====== Mediapipe 初始化 ======
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2,
                    enable_segmentation=False, min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def extract_features_from_frame(frame):
    h, w, _ = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if not results.pose_landmarks:
        return None
    lm = results.pose_landmarks.landmark
    # compute hip width in pixel space
    left_hip = np.array([lm[LM['LEFT_HIP']].x * w, lm[LM['LEFT_HIP']].y * h])
    right_hip = np.array([lm[LM['RIGHT_HIP']].x * w, lm[LM['RIGHT_HIP']].y * h])
    hip_width = np.linalg.norm(left_hip - right_hip)
    if hip_width == 0 or np.isnan(hip_width):
        return None
    feats = []
    for (p1, p2), colname in zip(EDGES, feature_columns):
        x1, y1 = lm[LM[p1]].x * w, lm[LM[p1]].y * h
        x2, y2 = lm[LM[p2]].x * w, lm[LM[p2]].y * h
        d = np.linalg.norm([x2 - x1, y2 - y1]) / hip_width
        feats.append(d)
    return np.array(feats, dtype=float).reshape(1, -1)

# ====== 開攝影機 ======
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera (0)")

print("Start camera. Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        feats = extract_features_from_frame(frame)
        label_text = "NoPose"
        prob_text = ""
        if feats is not None:
            # standardize using scaler mean/scale computed earlier
            feats_scaled = (feats - scaler_mean) / safe_scale  # shape (1, n_features)

            # apply weights if requested and available
            if use_weights:
                feats_scaled = feats_scaled * np.sqrt(weights)

            # Build DataFrame in right column order (RF expects columns)
            df_in = pd.DataFrame(feats_scaled, columns=feature_columns)

            # predict
            try:
                pred = rf_model.predict(df_in)[0]
                label_text = str(pred)
                # optionally show probability if supported
                if hasattr(rf_model, "predict_proba"):
                    probs = rf_model.predict_proba(df_in)[0]
                    top_idx = np.argmax(probs)
                    top_prob = probs[top_idx]
                    prob_text = f" p={top_prob:.2f}"
            except Exception as e:
                label_text = "Err"
                prob_text = f" ({e})"
        # overlay
        weight_status = "Weighted" if use_weights else "Normal"
        display_text = f"{label_text} [{weight_status}]{prob_text}"
        # draw landmarks anyway for UX
        try:
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        except Exception:
            pass
        cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        # show frame
        #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        cv2.imshow("RF Real-time Classification", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    print("Exiting.")

