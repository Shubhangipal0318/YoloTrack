import cv2
from ultralytics import YOLO

# Load pretrained YOLOv8 model (you can also try "yolov8n.pt" for faster inference)
model = YOLO("yolov8s.pt")

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLO inference
    results = model(frame, stream=True)
    
    # Draw bounding boxes
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # bounding box
            conf = float(box.conf[0])               # confidence
            cls = int(box.cls[0])                   # class id
            label = model.names[cls]
            
            # Draw rectangle + label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0, 255, 0), 2)
    
    # Show the result
    cv2.imshow("YOLOv8 Object Detection", frame)
    
    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

