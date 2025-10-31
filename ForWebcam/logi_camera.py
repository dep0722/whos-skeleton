'''
開啟攝影機直接看畫面
'''
import cv2

# === 開啟攝影機 ===
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 攝影機索引值 (0/1)，依電腦而定
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # 設定MJPG格式


if not cap.isOpened():
    print("❌ 無法開啟攝影機")
    exit()

print("✅ 攝影機已開啟，按下 Q 結束")

# === 讀取與顯示畫面 ===
while True:
    ret, frame = cap.read()
    #順轉90:cv2.ROTATE_90_CLOCKWISE，逆90:cv2.ROTATE_90_COUNTERCLOCKWISE，180:cv2.ROTATE_180
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if not ret:
        print("⚠️ 無法讀取畫面")
        break

    frame = cv2.resize(frame, None, fx=0.95, fy=0.95)
    cv2.imshow("Live Camera", frame)

    # 按下 Q 鍵離開
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === 清理資源 ===
cap.release()
cv2.destroyAllWindows()
