import cv2
import json
import os

# ====== 路徑設定 ======
video_path = r"C:/Users/User/Pictures/Camera Roll/1210_none.mp4"
json_dir = r"C:/mydata/sf/openpose/output_json/steps/1210_1"
output_dir = r"C:/mydata/sf/openpose/output_videos/1210_1"
os.makedirs(output_dir, exist_ok=True)

# ====== 骨架連線設定（BODY_25 範例） ======
BODY_25_PAIRS = [
    (1, 8), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14),
    (1, 0), (0, 15), (15, 17), (0, 16), (16, 18), (14, 19),
    (19, 20), (14, 21), (11, 22)
]

# ====== 讀取原影片 ======
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()

# ====== 遍歷每一段 JSON ======
json_files = sorted([f for f in os.listdir(json_dir) if f.endswith(".json")])

for json_file in json_files:
    path = os.path.join(json_dir, json_file)
    with open(path, "r") as f:
        seg_frames = json.load(f)

    # 設定影片寫入
    out_file = os.path.join(output_dir, json_file.replace(".json", ".mp4"))
    out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    for frame_data in seg_frames:
        idx = frame_data["frame_index"]
        frame = frames[idx].copy()
        kp = frame_data["keypoints"]

        # 畫骨架
        for i, j in BODY_25_PAIRS:
            x1, y1, c1 = kp[i*3:i*3+3]
            x2, y2, c2 = kp[j*3:j*3+3]
            if c1 > 0 and c2 > 0:  # 只畫可見點
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        # 畫關節點
        for i in range(len(kp)//3):
            x, y, c = kp[i*3:i*3+3]
            if c > 0:
                cv2.circle(frame, (int(x), int(y)), 4, (0,0,255), -1)

        out.write(frame)

    out.release()
    print(f"影片已輸出: {out_file}")

print("全部完成！")
