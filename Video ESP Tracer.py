import cv2
import os
import tkinter as tk
from tkinter import filedialog

def process_video():
    # 1. Ask user for a video file
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    video_path = filedialog.askopenfilename(title="Send ur mp4 file u fatfuck nigger", 
                                            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
    
    if not video_path:
        print("No file selected. Exiting.")
        return

    # 2. Setup Video Capture
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties for saving
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    
    # Define output path (same folder as script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "tracked_people_output.mp4")
    
    # Define Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize the HOG person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    print(f"Processing... Tracking people and saving to: {output_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 3. Detect people in the frame
        # winStride and padding help balance speed vs accuracy
        (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

        # 4. Draw bounding boxes around walking people
        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "ESP TEST", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write frame to output file
        out.write(frame)
        
        # (Optional) Show the processing in real-time
        cv2.imshow('Tracking People', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Finished! Video saved successfully.")

if __name__ == "__main__":
    process_video()