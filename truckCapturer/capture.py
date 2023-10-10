import cv2
import time
from detect import main

model = ('yolov8n.onnx')

cap = cv2.VideoCapture('../pics/sample_video_1.mov')

stepper = 1  # For naming output files
sleep = 0
starttime = time.time()
frames = 0
while cap.isOpened():
    ret, frame = cap.read()
    frames += 1
    if sleep == 0:
        truck = main(model, frame)
    else:
        sleep -= 1
        truck = None

    if truck is not None:
        cv2.imwrite(f"{stepper}.jpg", truck)
        stepper += 1
        sleep = 30

    fps = frames / (time.time() - starttime)
    print(f"{fps} Frames/Sec")