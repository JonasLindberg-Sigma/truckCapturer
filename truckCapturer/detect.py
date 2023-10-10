import argparse

import cv2.dnn
import numpy as np

from ultralytics.yolo.utils import yaml_load
from ultralytics.yolo.utils.checks import check_yaml

CLASSES = yaml_load(check_yaml('coco128.yaml'))['names']
_DEBUG = True
needed_classes = ['truck']

colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f'{CLASSES[class_id]} ({confidence:.2f})'
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def main(onnx_model, input_image):
    model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(onnx_model)
    if isinstance(input_image, str):
        original_image: np.ndarray = cv2.imread(input_image)
    else:
        original_image: np.ndarray = input_image
    [height, width, _] = original_image.shape
    area = height * width
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image
    scale = length / 640

    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
    model.setInput(blob)
    outputs = model.forward()

    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []

    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        if maxScore >= 0.25:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2], outputs[0][i][3]]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

    detections = []
    frame = None
    for i in range(len(result_boxes)):
        index = result_boxes[i]
        if CLASSES[class_ids[index]] in needed_classes:
            box = boxes[index]
            detection = {
                'class_id': class_ids[index],
                'class_name': CLASSES[class_ids[index]],
                'confidence': scores[index],
                'box': box,
                'scale': scale}
            detections.append(detection)
            draw_bounding_box(original_image, class_ids[index], scores[index], round(box[0] * scale), round(box[1] * scale),
                              round((box[0] + box[2]) * scale), round((box[1] + box[3]) * scale))
            bbox_size = round((box[0] + box[2]) * scale) * round((box[1] + box[3]) * scale)
            bbox_area_ratio = bbox_size / area
            if bbox_area_ratio > 0.7:
                cv2.imshow("Frame", original_image)
                x1 = round(box[0] * scale)
                x2 = round(box[0] * scale) + round(box[2] * scale)
                y1 = round(box[1] * scale)
                y2 = round(box[1] * scale) + round(box[3] * scale)
                frame = original_image[y1:y2,x1:x2]
                # Press Q to exit
                #if cv2.waitKey(25) & 0xFF == ord('q'):
                #    break

    return frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolov8n.onnx', help='Input your onnx model.')
    parser.add_argument('--img', default=str('../pics/test.png'), help='Path to input image.')
    args = parser.parse_args()
    main(args.model, args.img)