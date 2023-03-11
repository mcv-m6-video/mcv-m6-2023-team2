import os
import math
import numpy as np
import matplotlib.pyplot as plt

from utils import (
    load_predictions,
    load_annotations,
    group_annotations_by_frame,
    load_optical_flow,
)
from noise import create_fake_track_predictions
from metrics import (
    voc_eval,
    OF_MSEN,
    OF_PEPN,
)
import global_config as cfg


def task1_1():
    # TODO: Reformat and make it prettier
    annotations = load_annotations(cfg.ANNOTATIONS_PATH)
    grouped_annotations = group_annotations_by_frame(annotations)
    annotations_with_noise = create_fake_track_predictions(
        annotations, 
        noise=0.1, 
        prob_generate=0.1, 
        prob_delete=0.1
    )
    grouped_annotations_with_noise = group_annotations_by_frame(annotations_with_noise)

    n_frames = len(grouped_annotations)

    print(f'Number of frames: {n_frames}')
    print(f'Number of annotations: {len(annotations)}')
    print(f'Number of annotations with noise: {len(annotations_with_noise)}')

    rec, prec, ap = voc_eval(grouped_annotations_with_noise, grouped_annotations)

    print(f'AP: {ap}')
    print(f'Precision: {prec}')
    print(f'Recall: {rec}')


def task1_2():
    annotations = load_annotations(cfg.ANNOTATIONS_PATH)
    grouped_annotations = group_annotations_by_frame(annotations)
    predictions = load_predictions(cfg.PREDICTIONS_PATH)
    grouped_predictions = group_annotations_by_frame(predictions)

    n_frames = len(grouped_annotations)

    print(f'Number of frames: {n_frames}')
    print(f'Number of annotations: {len(annotations)}')
    print(f'Number of predictions: {len(predictions)}')

    rec, prec, ap = voc_eval(grouped_predictions, grouped_annotations)

    print(f'AP: {ap}')
    print(f'Precision: {prec}')
    print(f'Recall: {rec}')


def task3_1_2(gt_path, estimated_path, frame):
    print("--> Tasks 3.1 and 3.2 - Quantitatively evaluating optical flow - KITTI Dataset")

    gt = load_optical_flow(os.path.join(gt_path, frame))
    estimated_flow = load_optical_flow(os.path.join(estimated_path, "LKflow_" + frame))

    msen, sen = OF_MSEN(gt, estimated_flow)
    pepn = OF_PEPN(gt, estimated_flow, sen)

    print(f"MSEN: {msen}\n PEPN: {pepn} %")

    return msen, pepn, sen


def task3_3(sen, frame):
    print("--> Task 3.3 - Visualize Error in Optical Flow")

    max_range = int(math.ceil(np.amax(sen)))
    plt.title('Distribution of the Mean Square Error in Non-Occluded Areas')
    plt.ylabel('Density')
    plt.xlabel('MSEN')
    plt.hist(x=sen, bins=30, range=(0.0, max_range))
    plt.savefig('MSEN_hist_' + frame)
    plt.clf()


def task3():
    gt_dir = "../data/GT_OF"
    preds_dir = "../data/LK_OF"
    frames = ["000045_10.png", "000157_10.png"]

    for frame in frames:
        _, _, sen = task3_1_2(gt_dir, preds_dir, frame)

        task3_3(sen, frame)
