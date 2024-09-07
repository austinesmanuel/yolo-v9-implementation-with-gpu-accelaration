#this code just implements yolov9 and doesnt apply segmentattion this is just the basic implementation

import cv2
from ultralytics import YOLO
from torchvision.ops import nms
import torch

# Load the pretrained YOLOv9 model
model_path = "yolov9e.pt"
model = YOLO(model_path)

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Initialize the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to the format expected by YOLO
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run YOLO model on the frame (on GPU if available, otherwise CPU)
    results = model.predict(frame_rgb, device=device)

    # Process the detection results
    for i, det in enumerate(results[0].boxes):
        # Transfer bounding boxes and scores to CPU for NMS
        boxes = det.xyxy.to('cpu')   # Bounding boxes
        scores = det.conf.to('cpu')  # Confidence scores

        # Perform NMS on CPU
        keep = nms(boxes, scores, iou_threshold=0.5)

        # Loop through the filtered detections
        for j in keep:
            x1, y1, x2, y2 = map(int, boxes[j].tolist())
            confidence = scores[j].item()
            class_id = int(det.cls[j].item())
            class_name = results[0].names[class_id]

            # Draw rectangle and label on the frame
            label = f"{class_name}: {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with detections
    cv2.imshow("YOLO Webcam Detection", frame)

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
