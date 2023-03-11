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


def task3_1_2(gt, estimated_flow, frame: str):
    print("--> Tasks 3.1 and 3.2 - Quantitatively evaluating optical flow - KITTI Dataset")

    msen, sen = OF_MSEN(gt, estimated_flow, frame=frame, verbose=False)
    pepn = OF_PEPN(sen)

    print(f"MSEN: {msen}\nPEPN: {pepn} %")

    return msen, pepn, sen


# def task3_3(sen, frame):
def task3_3(GT, OF_pred, frame):
    print("--> Task 3.3 - Visualize Error in Optical Flow")

    error_dist = u_diff, v_diff = GT[:, :, 0] - OF_pred[:, :, 0], GT[:, :, 1] - OF_pred[:, :, 1]
    error_dist = np.sqrt(u_diff ** 2 + v_diff ** 2)

    max_range = int(math.ceil(np.amax(error_dist)))

    plt.figure(figsize=(12, 5))
    plt.hist(error_dist[GT[...,2] == 1].ravel(), bins=30, range=(0.0, max_range))
    plt.title('MSEN Distribution')
    plt.ylabel('Count')
    plt.xlabel('Mean Square Error in Non-Occluded Areas')
    plt.savefig(f'./results/MSEN_hist_{frame}.png')


def task3():
    gt_dir = "../data/GT_OF"
    preds_dir = "../data/LK_OF"
    frames = ["000045_10.png", "000157_10.png"]

    for frame in frames:
        print(f"--> Processing frame: {frame}...")
        gt = load_optical_flow(os.path.join(gt_dir, frame))
        estimated_flow = load_optical_flow(os.path.join(preds_dir, "LKflow_" + frame))

        _, _, sen = task3_1_2(gt, estimated_flow, frame=frame.split('.')[0])
        task3_3(gt, estimated_flow, frame.split('.')[0])
        print(f"--> Finished processing frame: {frame}!\n")
