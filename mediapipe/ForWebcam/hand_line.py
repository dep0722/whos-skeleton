import cv2
import mediapipe as mp
import numpy as np

# === Mediapipe Pose 初始化 ===
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("無法讀取攝影機畫面")
            break

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        h, w, _ = image.shape

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            RS = landmarks[12]  # RIGHT_SHOULDER
            RE = landmarks[14]  # RIGHT_ELBOW
            RW = landmarks[16]  # RIGHT_WRIST

            # 原始三維座標
            points_3d = {
                "Shoulder": (RS.x, RS.y, RS.z),
                "Elbow": (RE.x, RE.y, RE.z),
                "Wrist": (RW.x, RW.y, RW.z)
            }

            # 計算 3D 空間距離
            def distance_3d(p1, p2):
                x1, y1, z1 = p1
                x2, y2, z2 = p2
                return np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

            L1 = distance_3d(points_3d["Shoulder"], points_3d["Elbow"])
            L2 = distance_3d(points_3d["Elbow"], points_3d["Wrist"])
            total_length = L1 + L2

            # x, y 轉成像素座標，用來畫在螢幕上
            points_2d = {name: (int(x*w), int(y*h)) for name, (x, y, z) in points_3d.items()}

            for name, (cx, cy) in points_2d.items():
                cv2.circle(image, (cx, cy), 6, (0, 255, 0), -1)
                cv2.putText(image, name, (cx + 10, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 顯示三維距離
            cv2.putText(image, f"Shoulder-Elbow: {L1:.4f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(image, f"Elbow-Wrist: {L2:.4f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(image, f"Total Arm Length: {total_length:.4f}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 畫骨架線
            def draw_line(p1, p2):
                cv2.line(image, points_2d[p1], points_2d[p2], (0, 255, 255), 3)

            draw_line("Shoulder", "Elbow")
            draw_line("Elbow", "Wrist")

        cv2.imshow("Right Arm 3D Length", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()