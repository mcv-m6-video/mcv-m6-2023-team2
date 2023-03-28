import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T

from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog, DatasetCatalog

import datautils as rapapa

MODELS = {
    'retina': 'retinanet_R_50_FPN_3x',
    'faster': 'faster_rcnn_R_50_FPN_3x',
}

def run_finetune_detectron(args, split: str = 'random', kfolds = True, folds = 'xxo'):

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
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_path)
    cfg.OUTPUT_DIR = os.path.join(args.path_results, args.model)
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.CHECKPOINT_PERIOD = 10
    cfg.SOLVER.MAX_ITER = 1000
    
    ### HERE WE REGISTER THE DATASET ###
    if not kfolds:
        loader = rapapa.load_first_data if split == 'first' else rapapa.load_random_data
    else:
        if folds == 'xox':
            loader = rapapa.load_xox
        elif folds == 'xxo':
            loader = rapapa.load_xxo
        elif folds == 'oxx':
            loader = rapapa.load_oxx

    for d in [f"train_{split if not kfolds else folds}", f"val_{split if not kfolds else folds}"]:
        DatasetCatalog.register(d, lambda d=d: loader(d.split('_')[0] ))
        MetadataCatalog.get(d).set(thing_classes=["car", "bike"])
    metadata = MetadataCatalog.get(f"train_{split if not kfolds else folds}")

    cfg.DATASETS.TRAIN = (f"train_{split if not kfolds else folds}",)
    cfg.DATASETS.TEST = (f"val_{split if not kfolds else folds}",)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == '__main__': 
    # FIXME: sry i am just crazy atm
    class Args:
        ## IM A GUARRADA SO I DONT HAVE TO PARSE ANY ARGS ##
        model = 'faster'
        path_results = 'results_guarrada'

        def __init__(self) -> None:
            pass
    run_finetune_detectron(Args())



    

