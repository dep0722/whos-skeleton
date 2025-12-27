import cv2
import mediapipe as mp
import numpy as np

# === Mediapipe Pose 初始化 ===
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 開啟攝影機
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

        # 翻轉畫面並轉成 RGB
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        h, w, _ = image.shape

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # 右肩、右肘、右腕 index
            RS = landmarks[12]  # RIGHT_SHOULDER
            RE = landmarks[14]  # RIGHT_ELBOW
            RW = landmarks[16]  # RIGHT_WRIST

            points = {
                "Shoulder": (RS.x, RS.y, RS.z),
                "Elbow": (RE.x, RE.y, RE.z),
                "Wrist": (RW.x, RW.y, RW.z)
            }

            z_vals = []
            for name, (x, y, z) in points.items():
                cx, cy = int(x * w), int(y * h)
                z_vals.append(z)
                cv2.circle(image, (cx, cy), 6, (0, 255, 0), -1)
                cv2.putText(image, f"{name}: z={z:.3f}", (cx + 10, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            mean_z = np.mean(z_vals)
            std_z = np.std(z_vals)
            cv2.putText(image, f"Right Arm Z mean: {mean_z:.3f} (std {std_z:.3f})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # 畫出手臂骨架線
            def draw_line(p1, p2):
                x1, y1 = int(points[p1][0] * w), int(points[p1][1] * h)
                x2, y2 = int(points[p2][0] * w), int(points[p2][1] * h)
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 255), 3)

            draw_line("Shoulder", "Elbow")
            draw_line("Elbow", "Wrist")

        cv2.imshow("Right Arm Z Detection", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
