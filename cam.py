import torch
import cv2

# Define the class names corresponding to the class IDs
class_names = ['with mask', 'no mask']

try:
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
    
    capture = cv2.VideoCapture(0)
    
    while True:
        ret, frame = capture.read()  # Read a frame from the video capture
    
        # Perform face detection on the frame
        results = model(frame)
    
        # Extract the detected face bounding boxes
        faces = results.xyxy[0]
    
        for face in faces:
            x1, y1, x2, y2, confidence, class_id = face.tolist()
            pt1 = (int(x1), int(y1))  # Convert the coordinates to integers
            pt2 = (int(x2), int(y2))  # Convert the coordinates to integers
            thickness = 2  # Thickness of the rectangle's edges
            
            # Determine the color based on the label
            if int(class_id) == 0:
                color = (0, 255, 0)  # Green color for 'with mask'
            else:
                color = (0, 0, 255)  # Red color for 'no mask'
    
            # Write the label on the box
            label = class_names[int(class_id)]
            label_position = (int(x1), int(y1) - 10)  # Position of the label text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            cv2.putText(frame, label, label_position, font, font_scale, color, thickness)
            cv2.rectangle(frame, pt1, pt2, color, thickness)
            
        cv2.imshow('Face Detection', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    
    capture.release()
    cv2.destroyAllWindows()

except Exception as e:
    print("An error occurred:", str(e))
