# from ultralytics import YOLO

# model = YOLO(r"/app/yolov8pose/weights/runs/detect/train3/weights/best.pt")
# # # Define path to video file
# source = r"/app/videos/market_person_search.mp4"
# #Run inference on the source
# results = model(source, stream=True, save=True, device=0, conf=0.25)

import cv2
from ultralytics import YOLO
import numpy as np
import torch

# Set the device to GPU if available
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Initialize the YOLO model on the specified device
model = YOLO(r"/app/yolov8pose/weights/runs/detect/train3/weights/best.pt").to(device)

# Open the source video
cap = cv2.VideoCapture(r"/app/videos/market_person_search.mp4")

# Define the codec and create a VideoWriter object for AVI format
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, cap.get(cv2.CAP_PROP_FPS), 
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

try:
    # Process the video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame from BGR (OpenCV format) to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run YOLO prediction with specified confidence level
        results = model(frame_rgb, conf=0.25)

        # Customize and plot results
        for r in results:
            im_array = r.plot(line_width=2, font_size=4)
            # Convert array from RGB (PIL format) to BGR (OpenCV format)
            frame_processed = cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR)

        # Write the processed frame to the output video
        out.write(frame_processed)

finally:
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
