from ultralytics import YOLO
import cv2
import threading
import time

# 🧠 Threaded camera class
class VideoStream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.cap.read()
            time.sleep(0.01)

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

# ✅ Load YOLO models
chips_model = YOLO('modeltrained/chips_model.pt')
# drinks_model = YOLO('modeltrained/drinks_model.pt')

# 📷 Start phone camera
ip_camera_url = 'http://192.168.1.109:8080/video'
stream = VideoStream(ip_camera_url)

print("✅ Running chips & drinks detection. Press 'q' to quit.")

conf_threshold = 0.5

while True:
    ret, frame = stream.read()
    if not ret:
        continue

    # Resize frame to 416x416 for faster detection
    resized = cv2.resize(frame, (416, 416))
    annotated_frame = frame.copy()

    # Detect with both models
    chips_results = chips_model(resized, show=False, verbose=False)[0]
    # drinks_results = drinks_model(resized, show=False, verbose=False)[0]

    # Scale factor to draw on full frame
    scale_x = frame.shape[1] / 416
    scale_y = frame.shape[0] / 416

    # 🟨 Draw chips_model boxes (yellow)
    for result in chips_results.boxes.data.tolist():
        x1, y1, x2, y2, score, cls_id = result
        if score < conf_threshold:
            continue
        label = f"{chips_model.names[int(cls_id)]} {score:.2f}"
        x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # 🟥 Draw drinks_model boxes (red)
    # for result in drinks_results.boxes.data.tolist():
    #     x1, y1, x2, y2, score, cls_id = result
    #     if score < conf_threshold:
    #         continue
    #     label = f"{drinks_model.names[int(cls_id)]} {score:.2f}"
    #     x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
    #     cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #     cv2.putText(annotated_frame, label, (x1, y1 - 10),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Show result
    cv2.imshow("YOLOv8 Detection (Chips & Drinks)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

stream.stop()
cv2.destroyAllWindows()
