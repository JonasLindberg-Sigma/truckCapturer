import cv2
from requests import post

from anprmodule.predict import run

def get_registration(img):
    reg = run(img)
    if reg:
        post("", reg)

def draw_boxes(img, bbox, identities=None, categories=None, names=None, offset=(0, 0)):
    _SCREEN_SZ = img.shape[0] * img.shape[1]
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        area = (x2 - x1) * (y2 - y1)
        ratio = area / _SCREEN_SZ
        id = int(identities[i]) if identities is not None else 0
        label = str(id)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 253), 2)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255, 144, 30), -1)
        cv2.putText(img, str(round(ratio,2)), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 1)

    return img