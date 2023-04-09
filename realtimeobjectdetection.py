import cv2
import math
import numpy as np

#cap = cv2.VideoCapture(2)
cap = cv2.VideoCapture("images/satlite.mp4")
model = YOLO("/home/fox/AICoding/NNProject/best.pt")



classNames = ["airport", "baseball-diamond", "basketball-court", "beach", "bridge", "chaparral", "church",
              "circular-farmland", "cloud","commercial-area", "dense-residential", "desert", "forest", "freeway", "golf-course", "ground-track-field","harbor", "industrial-area", "intersection", "island", "lake", "meadow", "medium-residential", "mobile-home-park", "mountain", "overpass","palace", "tparking-lot",
              "railway-line", "railway-station", "rectangular-farmland", "river", "roundabout", "runway", "sea-ice",
              "snowberg", "sparse-residential", "stadium", "storage-tank", "tennis-court", "terrace", "thermal-power-plant","wetland"]


while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale = 0.7, thickness = 1)


    cv2.imshow("Webcam", img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()