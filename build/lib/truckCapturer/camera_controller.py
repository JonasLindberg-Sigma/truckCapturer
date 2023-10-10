import hydra
import torch
from random import randint
from sort import *
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator
tracker = None
detected_trucks = []

from lib.config_loader import get_config
from lib.utils import draw_boxes, grab_truck

def init_tracker():
    global tracker

    sort_max_age = 5
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    tracker = Sort(max_age=sort_max_age, min_hits=sort_min_hits, iou_threshold=sort_iou_thresh)


rand_color_list = []


def random_color_list():
    global rand_color_list
    rand_color_list = []
    for i in range(0, 5005):
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        rand_color = (r, g, b)
        rand_color_list.append(rand_color)
    # ......................................


class DetectionPredictor(BasePredictor):

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):

        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
        # tracker
        self.data_path = p

        save_path = str(self.save_dir / p.name)  # im.jpg
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        self.all_outputs.append(det)
        if len(det) == 0:
            return log_string
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "

        # #..................USE TRACK FUNCTION....................
        dets_to_sort = np.empty((0, 6))

        for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
            dets_to_sort = np.vstack((dets_to_sort,
                                      np.array([x1, y1, x2, y2, conf, detclass])))

        tracked_dets = tracker.update(dets_to_sort)

        if len(tracked_dets) > 0:
            bbox_xyxy = tracked_dets[:, :4]
            identities = tracked_dets[:, 8]
            categories = tracked_dets[:, 4]
            #draw_boxes(im0, bbox_xyxy, identities, categories, self.model.names)
            grab_truck(bbox_xyxy, im0, detected_trucks, identities)

        return log_string


def run(src, model=None):
    with hydra.initialize(version_base=None, config_path=None, job_name='truckCapture'):
        cfg = get_config()
        def _run(src, cfg, model=None):
            init_tracker()
            random_color_list()

            cfg.model = cfg.model or "yolov8n.pt"
            cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
            cfg.source = src
            cfg.show = True
            cfg.device = "mps"  # Only apllicable for MacO
            predictor = DetectionPredictor(cfg)
            predictor()
        _run(src, cfg)


if __name__ == "__main__":
    src = '../pics/sample_video_2.mov'
    run(src=src)