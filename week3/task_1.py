import argparse
import os, cv2
import numpy as np
import torch

from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from tqdm import tqdm


MODELS = {
    'retina': 'retinanet_R_50_FPN_3x',
}

def detect_re(args):

    coco_car_id = 3

    model_path = 'COCO-Detection/' + MODELS[args.model] + '.yaml'
    print(model_path)

    # Run a pre-trained detectron2 model
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(model_path))
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.RETINANET.NMS_THRESH_TEST = 0.4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_path)
    cfg.OUTPUT_DIR = args.path_results
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    predictor = DefaultPredictor(cfg)

    res_path = os.path.join(cfg.OUTPUT_DIR, 'detections.txt')
    if os.path.exists(res_path):
        os.remove(res_path)

    cv2_vid = cv2.VideoCapture(args.path_video)
    num_frames = int(cv2_vid.get(cv2.CAP_PROP_FRAME_COUNT))
    # num_frames = 3

    # keep track of time to compute s/img
    timestamps = []
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for frame_id in tqdm(range(num_frames)):
        _, frame = cv2_vid.read()

        # record inference time
        begin.record()
        model_preds = predictor(frame)
        end.record()

        torch.cuda.synchronize()
        timestamps.append(begin.elapsed_time(end))

        bboxes = model_preds["instances"].pred_boxes.to("cpu")
        confs = model_preds["instances"].scores.to("cpu")
        classes = model_preds["instances"].pred_classes.to("cpu")

        for i, prediction in enumerate(classes):
            if prediction.item() == coco_car_id:
                # TODO: also allow predicting trucks (because pick-up trucks are also cars, but in COCO they are considered trucks)
                box = bboxes[i].tensor.numpy()[0]

                # Store in AI City Format:
                # <frame> <id> <bb_left> <bb_top> <bb_width> <bb_height> <conf> <x> <y> <z>
                det = str(frame_id+1)+',-1,'+str(box[0])+','+str(box[1])+','+str(box[2]-box[0])+','+str(box[3]-box[1])+','+str(confs[i].item())+',-1,-1,-1\n'

                with open(res_path, 'a') as f:
                    f.write(det)

        if args.store_results:
            output_path = os.path.join(cfg.OUTPUT_DIR, 'det_frame_' + str(frame_id) + '.png')
            v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
            out = v.draw_instance_predictions(model_preds["instances"].to("cpu"))
            cv2.imwrite(output_path, out.get_image()[:, :, ::-1])

    print('Inference time (s/img): ', np.mean(timestamps)/1000)

    return res_path
