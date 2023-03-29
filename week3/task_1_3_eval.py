import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

import torch
torch.set_grad_enabled(False)
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T

from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import datautils as rapapa
from detectron2.checkpoint import DetectionCheckpointer

from utils import (
    load_annotations,
    load_predictions,
    generate_test_subset,
    group_annotations_by_frame,
)

COCO_CAR_ID = 1  # COCO class id for car.
COCO_TRUCK_ID = 1  # COCO class id for truck.
COCO_PERSON_ID = 0  # COCO class id for person.
VALID_IDS_SUBS = [0, 1]
VALID_IDS_ORIG = [COCO_CAR_ID, COCO_TRUCK_ID]

COCO_ID_TO_NAME = {
    COCO_CAR_ID-1: "car",
    COCO_TRUCK_ID-1: "car",
    COCO_PERSON_ID-1: "car",
}

OURS_TO_COCO = {
    1: 0,
    0: 1,
    2: 2
}

def run_inference_detectron(args, split = 'random', json_val = './datafolds/val_first.json'):
    data = json.load(open(json_val, 'r'))
    idxs_val = [x['id'] for x in data]    
    print(idxs_val[0])

    MODELS = {
        'retina': 'retinanet_R_50_FPN_3x',
        'faster': 'faster_rcnn_R_50_FPN_3x',
    }

    model_path = 'COCO-Detection/' + MODELS[args.model] + '.yaml'

    # Run a pre-trained detectron2 model
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(model_path))
    if args.model == 'retina':
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.RETINANET.NMS_THRESH_TEST = 0.4
    elif args.model == 'faster':
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.4
    cfg.OUTPUT_DIR = os.path.join(args.path_results, args.model)
    cfg.MODEL.WEIGHTS = './results_guarrada/faster/model_final.pth' #model_zoo.get_checkpoint_url(model_path)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    predictor = DefaultPredictor(cfg)
    #DetectionCheckpointer(predictor).load()  # load a file, usually from cfg.MODEL.WEIGHTS

    res_path = os.path.join(cfg.OUTPUT_DIR, 'detections.txt')
    if os.path.exists(res_path):
        os.remove(res_path)

    cv2_vid = cv2.VideoCapture(args.path_video)
    num_frames = int(cv2_vid.get(cv2.CAP_PROP_FRAME_COUNT))

    # keep track of time (s/img) to run inferece
    timestamps = []
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    if args.format.lower() == "aicity":
        f = open(res_path, 'a')
    elif args.format.lower() == "kitti":
        labels_dir = os.path.join(cfg.OUTPUT_DIR, 'label_2')
        os.makedirs(labels_dir, exist_ok=True)
    print(num_frames)
    
    W = int(cv2_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cv2_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    outvid = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), cv2_vid.get(cv2.CAP_PROP_FPS), (W, H))
    for frame_id in tqdm(range(num_frames)):
        if not (frame_id+1) in idxs_val: continue
        _, frame = cv2_vid.read()


        model_preds = predictor(frame)
        instances = model_preds["instances"]

        if args.format.lower() == "aicity":
            filter = torch.logical_or(instances.pred_classes == VALID_IDS_SUBS[0], instances.pred_classes == VALID_IDS_SUBS[1])
        elif args.format.lower() == "kitti":
            filter = torch.logical_or(
                torch.logical_or(instances.pred_classes == VALID_IDS_SUBS[0], instances.pred_classes == VALID_IDS_SUBS[1]),
                instances.pred_classes == COCO_PERSON_ID-1  # also detect people
            )
            filter = torch.logical_and(filter, instances.scores > .75)
            f = open(os.path.join(labels_dir, f'{frame_id:05d}.txt'), 'w')

        filtered_instances = instances[filter]  # a car or a (pickup) truck
        bboxes = filtered_instances.pred_boxes.to("cpu")
        confs = filtered_instances.scores.to("cpu")
        classes = [x.item() for x in filtered_instances.pred_classes.to("cpu")]
        



        ### HERE WE REGISTER THE DATASET ###
        #loader = rapapa.load_first_data if split == 'first' else rapapa.load_random_data
        #for d in [f"train_{split}", f"val_{split}"]:
            #DatasetCatalog.register(d, lambda d=d: loader(d))
            #MetadataCatalog.get(d).set(thing_classes=["car", "bike"])
        metadata = MetadataCatalog.get(f"train_{split}")

        for i, prediction in enumerate(classes):
            # TODO: also allow predicting trucks (because pick-up trucks are also cars, but in COCO they are considered trucks)
            box = bboxes[i].tensor.numpy()[0]

            if args.format.lower() == "aicity":
                # Store in AI City Format:
                # <frame> <id> <bb_left> <bb_top> <bb_width> <bb_height> <conf> <x> <y> <z>
                det = str(frame_id+1)+',-1,'+str(box[0])+','+str(box[1])+','+str(box[2]-box[0])+','+str(box[3]-box[1])+','+str(confs[i].item())+',-1,-1,-1\n'
            elif args.format.lower() == "kitti":
                # car 0.00 0 0.00 587.01 173.33 614.12 200.12 0.00 0.00 0.00 0.00 0.00 0.00 0.00
                # person 0.00 0 0.00 587.01 173.33 614.12 200.12 0.00 0.00 0.00 0.00 0.00 0.00 0.00
                category = 'car' if not classes[i].item() else 'pedestrian'
                det = f'{category} 0.00 0 0.00 {box[0]} {box[1]} {box[2]} {box[3]} 0.00 0.00 0.00 0.00 0.00 0.00 0.00\n'
            else:
                raise ValueError("Unknown format: {}".format(args.format))

            f.write(det)

        if args.store_results:
            output_path = os.path.join(cfg.OUTPUT_DIR, 'det_frame_' + str(frame_id) + '.png')
            v = Visualizer(frame[:, :, ::-1], metadata, scale=1)
            out = v.draw_instance_predictions(filtered_instances.to("cpu"))
            #cv2.imwrite(output_path, out.get_image()[:, :, ::-1])
            outvid.write(out.get_image()[:, :, ::-1])

        if args.format.lower() == "kitti":
            f.close()

    if args.format.lower() == "aicity":
        f.close()
    outvid.release()

    print('Inference time (s/img): ', np.mean(timestamps)/1000)

    return idxs_val


from metrics import voc_eval


def filter_by_conf(detections, conf_thr=0.5):
    filtered_detections = []
    for det in detections:
        if det.confidence >= conf_thr:
            filtered_detections.append(det)

    return filtered_detections


def task_1_3_evaluation(args, valid_idxs = []):
    gt_ = load_annotations(args.path_GT, grouped=True, use_parked=True)
    gt_ = [x for n, x in enumerate(gt_) if (n +1) in valid_idxs]
    
    gt = []
    for x in gt_: gt.extend(x)

    det = load_predictions(args.path_det, grouped=False)

    for min_conf in [0.5, 0.75, 0.9]:
        for min_iou in [0.5, 0.75, 0.9]:
            det_filt = filter_by_conf(det, conf_thr=min_conf)

            test_perc = 0.75
            test_gt = generate_test_subset(gt, N_frames=2141, test_p=test_perc)
            test_det = generate_test_subset(det_filt, N_frames=2141, test_p=test_perc)

            rec, prec, f1, ap, iou = voc_eval(
                group_annotations_by_frame(test_det),
                group_annotations_by_frame(test_gt),
                min_iou,
            )
            print(f'Min. Confidence: {min_conf}; Min. IoU: {min_iou}', )
            print('AP' + str(min_iou) + ': ', ap)
            # print(f'Recall: {rec}  |  Precision: {prec}  |  F1: {f1}  |  IoU: {iou}')
            print('-'*50)

if __name__ == '__main__': 
    # FIXME: sry i am just crazy atm
    class Args:
        ## IM A GUARRADA SO I DONT HAVE TO PARSE ANY ARGS ##
        model = 'faster'
        path_results = 'results_guarrada'
        path_video = '../data/AICity_S03_c010/vdo.avi'
        format = 'aicity'
        store_results = True
        path_GT = "../data/AICity_S03_c010/ai_challenge_s03_c010-full_annotation.xml"
        path_det = "results_guarrada/faster/detections.txt"


        def __init__(self) -> None:
            pass
    idx_val = run_inference_detectron(Args())
    task_1_3_evaluation(Args(), idx_val)
