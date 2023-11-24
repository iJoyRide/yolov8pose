from ultralytics import YOLO 
#Load a pretrained YOLOv8n model
model = YOLO(r"\\192.168.77.100\Model_Center\Yolo\YoloV8CY\yolov8pose\weights\runs\pose\train6\weights\best.pt")
# Define path to video file
source = r"\\192.168.77.100\Model_Center\Yolo\YoloV8CY\videos\crossing.mp4"
#Run inference on the source
results = model(source, save=True, device=0, conf=0.1)#, classes = [0,7])  
# generator of Results objects