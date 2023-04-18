import os
import sys
import cv2
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
torch.set_grad_enabled(False)
import torchvision.transforms as T

from ultralytics import YOLO

from utils import load_config


# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

COCO_CAR_ID    = 3  # COCO class id for car.
COCO_TRUCK_ID  = 8  # COCO class id for truck.
COCO_PERSON_ID = 1  # COCO class id for person.
VALID_IDS_SUBS = [COCO_CAR_ID-1, COCO_TRUCK_ID-1]


def plot_results(pil_img, prob, boxes, output_path, classes=None):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for i, (p, (xmin, ymin, xmax, ymax), c) in enumerate(zip(prob, boxes, colors)):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax() if classes is None else classes[i]
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}' if classes is None else f'{CLASSES[cl]}: {p:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.savefig(output_path)


def run_inference_yolo(cfg):

    model = YOLO(cfg["weights"])

    for seq in cfg["sequences"]:

        for cam in cfg["cameras"][seq]:

            video_path = os.path.join(cfg["dataset_dir"], seq, cam, 'vdo.avi')

            res_dir = os.path.join(cfg["path_results"], seq, cam)
            os.makedirs(res_dir, exist_ok=True)

            res_path = os.path.join(res_dir, 'detections.txt')
            if os.path.exists(res_path):
                os.remove(res_path)

            cv2_vid = cv2.VideoCapture(video_path)
            num_frames = min( int(cv2_vid.get(cv2.CAP_PROP_FRAME_COUNT)), cfg["num_frames"] )

            f = open(res_path, 'a')
            for frame_id in tqdm(range(num_frames)):
                _, frame_orig = cv2_vid.read()
                frame_orig = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB)
                frame_pil = T.ToPILImage()(frame_orig)

                results = model.predict(source=frame_pil, save=False)  # save plotted images

                confs, bboxes, classes = [], [], []
                for result in results:
                    for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                        cls = int(cls.item())
                        if cls in VALID_IDS_SUBS:
                            box = box.cpu().numpy()
                            confs.append(conf)
                            bboxes.append(box)
                            classes.append(cls+1)
                            det = str(frame_id+1)+',-1,'+str(box[0])+','+str(box[1])+','+str(box[2]-box[0])+','+str(box[3]-box[1])+','+str(conf.item())+',-1,-1,-1\n'
                            f.write(det)

                if cfg["store_results"]:
                    output_path = os.path.join(res_dir, 'det_frame_' + str(frame_id) + '.png')
                    plot_results(frame_pil, confs, bboxes, output_path, classes=classes)

            f.close()
            print("Results for sequence {} camera {} saved in {}".format(seq, cam, res_path))

    return res_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/inference_yolo.yaml")
    args = parser.parse_args(sys.argv[1:])

    cfg = load_config(args.config)

    run_inference_yolo(cfg)
