import cv2
import mediapipe as mp

# === 自訂骨架連線 ===
PAIRS = [
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

# === 初始化 MediaPipe Pose ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
mp_pose_connections = mp_pose.PoseLandmark

# === 開啟攝影機 ===
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

if not cap.isOpened():
    print("❌ 無法開啟攝影機")
    exit()

print("✅ 攝影機已開啟，按下 Q 結束")

# === 讀取與顯示畫面 ===
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ 無法讀取畫面")
        break

    # 旋轉畫面（順時針 90°）
    #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # Mediapipe 要使用 RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    # 若偵測到人體關鍵點
    if results.pose_landmarks:
        h, w, _ = frame.shape
        landmarks = results.pose_landmarks.landmark

        # 繪製自訂骨架線條
        for pair in PAIRS:
            p1, p2 = pair
            if (hasattr(mp_pose_connections, p1) and hasattr(mp_pose_connections, p2)):
                idx1 = mp_pose_connections[p1].value
                idx2 = mp_pose_connections[p2].value

                x1, y1 = int(landmarks[idx1].x * w), int(landmarks[idx1].y * h)
                x2, y2 = int(landmarks[idx2].x * w), int(landmarks[idx2].y * h)

                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # 也可選擇畫出所有關節點
        for lm in landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

    # 縮放顯示畫面
    frame = cv2.resize(frame, None, fx=0.95, fy=0.95)
    cv2.imshow("Live Pose Detection", frame)

    # 按下 Q 鍵離開
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === 清理資源 ===
cap.release()
cv2.destroyAllWindows()
