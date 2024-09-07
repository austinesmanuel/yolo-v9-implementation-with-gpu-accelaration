import cv2
from ultralytics import YOLO
from torchvision.ops import nms
import torch

# Load the pretrained YOLOv9 segmentation model
model_path = "yolov9c-seg.pt"
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

    # Process the detection results (boxes + segmentation masks)
    for i, det in enumerate(results[0].boxes):
        # Transfer bounding boxes and scores to CPU for NMS
        boxes = det.xyxy.to('cpu')   # Bounding boxes
        scores = det.conf.to('cpu')  # Confidence scores
        masks = results[0].masks.data.to('cpu')  # Segmentation masks data

        # Perform NMS on CPU
        keep = nms(boxes, scores, iou_threshold=0.5)

        # Loop through the filtered detections
        for j in keep:
            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, boxes[j].tolist())
            confidence = scores[j].item()
            class_id = int(det.cls[j].item())
            class_name = results[0].names[class_id]

            # Draw rectangle and label on the frame
            label = f"{class_name}: {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Get the segmentation mask and apply it to the frame
            mask = masks[j].numpy()  # Convert the mask tensor to numpy
            mask = (mask * 255).astype('uint8')  # Rescale mask to [0, 255]
            mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))  # Resize mask to match frame size

            # Apply a color to the mask
            colored_mask = cv2.applyColorMap(mask_resized, cv2.COLORMAP_JET)

            # Overlay the mask on the frame
            frame = cv2.addWeighted(frame, 1, colored_mask, 0.5, 0)

    # Display the frame with detections and segmentation masks
    cv2.imshow("YOLO Segmentation Preview", frame)

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
