import os
import matplotlib.pyplot as plt

from utils import (
    load_predictions,
    load_annotations,
    group_annotations_by_frame,
    create_fake_track_predictions,
    load_optical_flow,
)
from metrics import (
    voc_eval,
    MSEN,
    PEPN,
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

    msen, sen = MSEN(gt, estimated_flow)
    pepn = PEPN(gt, estimated_flow, sen)

    print(msen, pepn)

    return msen, pepn, sen


def task3_3(sen, frame):
    print("--> Task 3.3 - Visualize Error in Optical Flow")

    plt.hist(x=sen, bins=50)
    plt.savefig(frame)
    plt.clf()

    return


def task3():
    gt_path = "../data/GT_OF"
    estimated_path = "../data/LK_OF"
    frames = ["000045_10.png", "000157_10.png"]

    for frame in frames:
        msen, pepn, sen = task3_1_2(gt_path, estimated_path, frame)

        task3_3(sen, frame)
