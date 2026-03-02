#made by qore

import cv2
import os
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO

def process_video():
    # 1. Selection UI
    root = tk.Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(
        title="Select Video File", 
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
    )
    
    if not video_path:
        print("No file selected. Exiting.")
        return

    # 2. Setup YOLOv8 (Nano version is fast and accurate)
    # This will download the weights automatically on first run
    model = YOLO('yolov8n.pt') 

    # 3. Setup Video Capture
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    
    output_path = "tracked_people_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing... Saving to: {output_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 4. Run Detection
        # classes=0 limits detection to 'person' only
        results = model(frame, classes=0, conf=0.3, verbose=False)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # The "Size" logic: 
                # Deep learning automatically scales the box to the person's size in the frame.
                # We can adjust thickness based on how far (small) they are.
                w = x2 - x1
                thickness = 1 if w < 50 else 2

                # Draw the ESP box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness)
                
                # Label
                cv2.putText(frame, "Tracked People ESP", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        out.write(frame)
        cv2.imshow('Tracking People', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Finished!")

if __name__ == "__main__":
    process_video()
