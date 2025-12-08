import cv2
import mediapipe as mp
import pandas as pd

# 定義節點
LM = {
    'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12,
    'LEFT_ELBOW': 13, 'RIGHT_ELBOW': 14,
    'LEFT_WRIST': 15, 'RIGHT_WRIST': 16,
    'LEFT_HIP': 23, 'RIGHT_HIP': 24,
    'LEFT_KNEE': 25, 'RIGHT_KNEE': 26,
    'LEFT_ANKLE': 27, 'RIGHT_ANKLE': 28,
    'LEFT_HEEL': 29, 'RIGHT_HEEL': 30,
}

# 初始化 MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 輸入影片檔案
video_path = "C:/Users/User/Pictures/Camera Roll/WIN_20251207_16_33_04_Pro.mp4"  # 改成你的影片檔案路徑
cap = cv2.VideoCapture(video_path)

# 儲存座標
all_landmarks = []
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    # BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 偵測姿勢
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        frame_data = {'frame': frame_idx}
        for name, idx in LM.items():
            lm = results.pose_landmarks.landmark[idx]
            frame_data[f'{name}_x'] = lm.x
            frame_data[f'{name}_y'] = lm.y
            frame_data[f'{name}_z'] = lm.z
        all_landmarks.append(frame_data)

cap.release()

# 儲存成 CSV
df = pd.DataFrame(all_landmarks)
df.to_csv('pose_1207.csv', index=False)
print("已儲存 pose_landmarks.csv")
