import cv2
import mediapipe as mp

# 初始化 MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 輸入影片檔案
video_path = "C:/Users/User/Pictures/Camera Roll/WIN_20251207_17_07_12_Pro.mp4"
cap = cv2.VideoCapture(video_path)

# 影片資訊
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 指定輸出影片名稱
output_path = "C:/mydata/sf/conda/1025_test/output_video/120703.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
        )

    # 寫入影片
    out.write(frame)

cap.release()
out.release()
pose.close()

print(f"已輸出骨架影片：{output_path}")
