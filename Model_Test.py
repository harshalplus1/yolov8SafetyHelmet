from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
import math

classes=[ "Safety","Not Safety"]
model=YOLO("best.pt")
mycolour=(0,0,255)
def video_detection(path=0):
    cap = cv2.VideoCapture(path)
    while True:
        success, img = cap.read()
        results = model(img, stream=True)
        for r in results:
            boxes=r.boxes
            for box in boxes:
                x1,y1,x2,y2=box.xyxy[0]
                x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
                w,h=x2-x1,y2-y1
                conf = math.ceil((box.conf[0] * 100)) / 100
                # cvzone.cornerRect(cap,(x1,y1,w,h))
                cls = int(box.cls[0])
                if conf > 0.5:
                    if classes[cls] == 'Not Safety':
                        myColor = (0, 0, 255)
                    elif classes[cls] == 'Safety':
                        myColor = (0, 255, 0)
                    else:
                        myColor = (255, 0, 0)

                cvzone.putTextRect(img, f'{classes[cls]} {conf}',
                                    (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
                                    colorT=(255, 255, 255), colorR=myColor, offset=5)
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)
        # cv2.imshow("Image",img)
        # if cv2.waitKey(1) & 0xFF==ord("s"):
        #     break
        yield img
cv2.destroyAllWindows()
