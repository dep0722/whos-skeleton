"""
mediapipe_pose_single_image_to_table.py

功能：
- 讀入單張影像
- 使用 MediaPipe Pose 偵測姿勢（landmarks）
- 將 landmarks（normalized x,y,z, visibility）轉為像素座標
- 建成 pandas.DataFrame 並印出、存成 CSV
- 在影像上畫出 landmark 與連線並儲存顯示

需求：
pip install mediapipe opencv-python pandas
"""

import cv2
import mediapipe as mp
import pandas as pd
import os

# -------- 參數：把這裡改成你的影像檔路徑 ----------
image_path = r"C:\mydata\sf\ppt\S__75546628.jpg"   # <-- 改成你的圖片檔名或完整路徑
output_csv = r"C:/mydata/sf/conda/1025_test/1228_1.csv"
output_annotated = r"C:\mydata\sf\conda\1025_test\test_photo\1228_1.jpg"
# --------------------------------------------------

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 可選參數：
POSE_STATIC_IMAGE_MODE = True   # 單張影像設為 True
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5   # 不影響 static image 太多

# landmark 名稱（MediaPipe Pose 33 個）
LANDMARK_NAMES = [
    "NOSE",
    "LEFT_EYE_INNER",
    "LEFT_EYE",
    "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER",
    "RIGHT_EYE",
    "RIGHT_EYE_OUTER",
    "LEFT_EAR",
    "RIGHT_EAR",
    "MOUTH_LEFT",
    "MOUTH_RIGHT",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_PINKY",
    "RIGHT_PINKY",
    "LEFT_INDEX",
    "RIGHT_INDEX",
    "LEFT_THUMB",
    "RIGHT_THUMB",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
    "LEFT_HEEL",
    "RIGHT_HEEL",
    "LEFT_FOOT_INDEX",
    "RIGHT_FOOT_INDEX",
]

# ---- 讀影像 ----
if not os.path.exists(image_path):
    raise FileNotFoundError(f"找不到影像檔：{image_path}")

img_bgr = cv2.imread(image_path)
if img_bgr is None:
    raise ValueError("讀取影像失敗，請檢查檔案格式與路徑。")
height, width = img_bgr.shape[:2]

# 將 BGR 轉成 RGB（MediaPipe 要 RGB）
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# ---- 呼叫 MediaPipe Pose ----
with mp_pose.Pose(
    static_image_mode=POSE_STATIC_IMAGE_MODE,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE
) as pose:
    results = pose.process(img_rgb)

# 若沒偵測到則給提示
if not results.pose_landmarks:
    print("沒有偵測到姿勢（pose_landmarks 為空）。")
else:
    # 擷取 landmarks（normalized coords）
    lm = results.pose_landmarks.landmark
    rows = []
    for i, lm_i in enumerate(lm):
        name = LANDMARK_NAMES[i] if i < len(LANDMARK_NAMES) else f"LANDMARK_{i}"
        x_norm = lm_i.x
        y_norm = lm_i.y
        z = lm_i.z
        visibility = lm_i.visibility

        # 轉成像素座標（若超出邊界也會得到相對值）
        x_px = int(round(x_norm * width))
        y_px = int(round(y_norm * height))

        rows.append({
            "index": i,
            "landmark": name,
            "x_norm": float(x_norm),
            "y_norm": float(y_norm),
            "z": float(z),
            "visibility": float(visibility),
            "x_px": x_px,
            "y_px": y_px,
        })

    # 建成 DataFrame
    df = pd.DataFrame(rows, columns=["index","landmark","x_norm","y_norm","z","visibility","x_px","y_px"])
    # 印出 DataFrame（完整列印）
    pd.set_option("display.max_rows", None)
    pd.set_option("display.precision", 6)
    print(df)

    # 存成 CSV
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n已儲存 CSV：{output_csv}")

    # ---- 在影像上畫出 landmark 與連線 ----
    annotated_image = img_bgr.copy()
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
    )

    # 可選：在每個 landmark 旁標上 index 與 name（如果畫太雜可註解掉）
    for r in rows:
        cv2.putText(
            annotated_image,
            f"{r['index']}:{r['landmark']}",
            (r['x_px'] + 3, r['y_px'] - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

    cv2.imwrite(output_annotated, annotated_image)
    print(f"已儲存帶標註影像：{output_annotated}")

    # 顯示影像（按任意鍵關閉視窗）
    cv2.imshow("Annotated Pose", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
