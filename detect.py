from ultralytics import YOLO
import cv2  as cv
model = YOLO("yolov8n.pt")

cap = cv.VideoCapture(r"C:\Users\DELL\Downloads\WhatsApp Video 2024-05-03 at 14.39.04_c05bb76c.mp4")
c=0
# results = model(frame,conf=0.7)
print(model.names)
while True:
    success, frame = cap.read()
    if success:     
        results = model(frame,conf=0.7,verbose=False,classes=[0, 2, 1, 5, 3, 9,  7 ])
        if len(results[0]) > 0:
            for result in results:
                if result.boxes:
                    box = result.boxes[0]
                    class_id = int(box.cls)
                    object_name = model.names[class_id]
                    print(object_name,class_id)
                    if object_name =="car":
                        c+=1
        annotated_frame = results[0].plot()
        
        print(c)

        cv.imshow("YOLOv8 Inference", annotated_frame)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv.destroyAllWindows()
