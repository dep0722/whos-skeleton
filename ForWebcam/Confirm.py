'''
確認外部攝影機連接用
'''
import cv2

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # 用 DirectShow
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # 強制 MJPG 格式
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("❌ 無法開啟攝影機")
    exit()

print("✅ 攝影機啟動，按 Q 離開")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ 無法讀取畫面，請確認未被其他程式占用")
        break

    cv2.imshow("Logitech C270", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
