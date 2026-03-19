import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/train19/weights/best.pt")

cap = cv2.VideoCapture("rtmp://172.20.10.5:1935/live")
while True:
    ret, frame = cap.read()
    if not ret: break

    outputs = model(frame)
    las = outputs[0].plot()

    cv2.imshow("Stream", las)
    if cv2.waitKey(1) == 27: break
cap.release()
cv2.destroyAllWindows()
