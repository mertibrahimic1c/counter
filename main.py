import cv2 
from ultralytics import YOLO 

model_yolo=YOLO("yolov8m.pt") 

cap=cv2.VideoCapture("people.mp4") 
cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Result", 1280, 720)
ids=set() 
count=0
while True:
    ret, frame=cap.read() 

    if not ret:
        print("Unable to play video") 
        break

    result=model_yolo.track(frame, persist=True) 
    arr_result=result[0].boxes 

    color = (0, 225, 0)
    for box in arr_result:
        class_id=int(box.cls[0]) 
        name=model_yolo.names[class_id] 
        
        if name=="person":
             confidence = float(box.conf[0])
             if confidence < 0.6:  
                continue
             track_id=int(box.id[0])
             if track_id not in ids:
                ids.add(track_id)
                count=count+1
             x1,y1,x2,y2=map(int,box.xyxy[0]) 
             cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

        

    cv2.putText(frame,f"People:{count}",(10,50),cv2.FONT_HERSHEY_SIMPLEX,1.5,color,3)
    cv2.imshow("Result",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

