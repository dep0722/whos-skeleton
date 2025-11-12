import os
import sys
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor

# ==========================
# 🔧 自動尋找 yolov7-pose 主目錄
# ==========================
FILE = os.path.abspath(__file__)
ROOT = os.path.dirname(FILE)

temp = ROOT
while temp != os.path.dirname(temp):
    if 'yolov7-pose' in os.listdir(temp):
        ROOT = os.path.join(temp, 'yolov7-pose')
        break
    temp = os.path.dirname(temp)

if 'yolov7-pose' not in ROOT:
    raise FileNotFoundError("❌ 找不到 yolov7-pose 目錄，請確認路徑。")

sys.path.append(ROOT)

# 匯入 YOLOv7-Pose 模組
from models.experimental import attempt_load
from models.yolo import Model
from utils.general import non_max_suppression, scale_coords

# 🔒 加入安全反序列化清單 (PyTorch 2.6+ 必須)
torch.serialization.add_safe_globals([Model, nn.modules.container.Sequential])

# ==========================
# 基本設定
# ==========================
BASE_DIR = r"C:/mydata/sf/conda/1025_test/p1105test"
WEIGHTS = os.path.join(ROOT, "yolov7-w6-pose.pt")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# 關節點定義 (17 點)
LM = {
    0:'nose', 1:'left_eye', 2:'right_eye', 3:'left_ear', 4:'right_ear',
    5:'left_shoulder', 6:'right_shoulder',
    7:'left_elbow', 8:'right_elbow',
    9:'left_wrist', 10:'right_wrist',
    11:'left_hip', 12:'right_hip',
    13:'left_knee', 14:'right_knee',
    15:'left_ankle', 16:'right_ankle'
}

# 要計算距離的骨架連線
EDGES = [
    ('LEFT_SHOULDER', 'LEFT_ELBOW'),
    ('LEFT_ELBOW', 'LEFT_WRIST'),
    ('RIGHT_SHOULDER', 'RIGHT_ELBOW'),
    ('RIGHT_ELBOW', 'RIGHT_WRIST'),
    ('LEFT_SHOULDER', 'RIGHT_SHOULDER'),
    ('LEFT_HIP', 'RIGHT_HIP'),
    ('LEFT_HIP', 'LEFT_KNEE'),
    ('LEFT_KNEE', 'LEFT_ANKLE'),
    ('RIGHT_HIP', 'RIGHT_KNEE'),
    ('RIGHT_KNEE', 'RIGHT_ANKLE'),
]

# ==========================
# 初始化模型（只載入一次）
# ==========================
print("🧠 載入 YOLOv7-Pose 模型中...")
model = attempt_load(WEIGHTS, map_location=DEVICE)
model.eval()
print("✅ 模型載入完成。")

# ==========================
# 單張圖片處理
# ==========================
def process_image(args):
    img_path, label = args
    image0 = cv2.imread(img_path)
    if image0 is None:
        print(f"❌ 無法讀取: {img_path}")
        return None

    img = cv2.resize(image0, (640, 640))
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(DEVICE)

    with torch.no_grad():
        pred = model(img_tensor)[0]
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, kpt_label=True)

    if len(pred) == 0 or pred[0] is None or len(pred[0]) == 0:
        print(f"⚠️ 未偵測到人體: {os.path.basename(img_path)}")
        return None

    det = pred[0][0]  # 取第一個人
    kpts = det[6:].view(-1, 3).cpu().numpy().astype(float)

    # 回原圖大小
    h0, w0 = image0.shape[:2]
    h, w = img_tensor.shape[2:]
    kpts[:, 0] *= (w0 / w)
    kpts[:, 1] *= (h0 / h)

    # dict: {名稱: (x, y, conf)}
    kpt_dict = {LM[i].upper(): (kpts[i, 0], kpts[i, 1], kpts[i, 2]) for i in LM}

    # 臀部寬度基準
    if 'LEFT_HIP' in kpt_dict and 'RIGHT_HIP' in kpt_dict:
        lh = np.array(kpt_dict['LEFT_HIP'][:2])
        rh = np.array(kpt_dict['RIGHT_HIP'][:2])
        hip_width = np.linalg.norm(lh - rh)
    else:
        hip_width = 1.0

    if hip_width < 1e-6:
        return None

    # 計算骨架比例距離
    feature_row = {}
    for (p1, p2) in EDGES:
        if p1 in kpt_dict and p2 in kpt_dict:
            a = np.array(kpt_dict[p1][:2])
            b = np.array(kpt_dict[p2][:2])
            dist = np.linalg.norm(a - b) / hip_width
            feature_row[f"{p1}_{p2}"] = dist
        else:
            feature_row[f"{p1}_{p2}"] = np.nan

    feature_row["label"] = label
    feature_row["filename"] = os.path.basename(img_path)
    return feature_row

# ==========================
# 批次處理
# ==========================
def read_images_and_extract_features():
    subfolders = [f for f in os.listdir(BASE_DIR)
                  if os.path.isdir(os.path.join(BASE_DIR, f))]
    subfolders.sort()

    all_images = []
    for subfolder in subfolders:
        folder_path = os.path.join(BASE_DIR, subfolder)
        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                all_images.append((os.path.join(folder_path, filename), subfolder))

    print(f"📸 總共 {len(all_images)} 張圖片，開始平行處理...")

    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for r in executor.map(process_image, all_images):
            if r is not None:
                results.append(r)

    df = pd.DataFrame(results)
    print(f"✅ 完成 {len(df)} 張圖片的特徵提取。")
    return df

# ==========================
# 主程式
# ==========================
if __name__ == "__main__":
    df = read_images_and_extract_features()
    output_path = r"C:/mydata/sf/conda/yolov7-pose/yolocode/output_CSV/yolo_pose_1106test.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"💾 已輸出 CSV: {output_path}")
