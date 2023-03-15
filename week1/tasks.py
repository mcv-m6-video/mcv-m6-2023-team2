import os
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional

from utils import (
    load_predictions,
    load_annotations,
    group_annotations_by_frame,
    load_optical_flow,
    iou_over_time,
)
from noise import create_fake_track_predictions
from metrics import (
    voc_eval,
    OF_MSEN,
    OF_PEPN,
)
import global_config as cfg


# frame 500
def task1_1(
        frame: Optional[int] = 500,
        std_size: float = 0.1,
        std_position: float = 0.1,
        prob_delete: float = 0.3,
        prob_similar: float = 0.1,
        std_similar: float = 0.2,
        min_random: int = 1,
        max_random: int = 1,
        similar_statistic: Optional[str] = None
    ):
    annotations = load_annotations(cfg.ANNOTATIONS_PATH)
    grouped_annotations = group_annotations_by_frame(annotations)
    annotations_with_noise = create_fake_track_predictions(
        annotations,
        height=1080,
        width=1920,
        std_size=std_size,
        std_position=std_position,
        prob_delete=prob_delete,
        prob_similar=prob_similar,
        std_similar=std_similar,
        min_random=min_random,
        max_random=max_random,
        similar_statistic=similar_statistic,
    )
    grouped_annotations_with_noise = group_annotations_by_frame(annotations_with_noise)

    n_frames = len(grouped_annotations)

    print(f'Number of frames: {n_frames}')
    print(f'Selected frame: {frame}')

    if frame:
        grouped_annotations = [grouped_annotations[frame]]
        grouped_annotations_with_noise = [grouped_annotations_with_noise[frame]]

    print(f'Number of annotations: {len(annotations)}')
    print(f'Number of annotations with noise: {len(annotations_with_noise)}')

    _, _, ap, iou = voc_eval(grouped_annotations_with_noise, grouped_annotations)

    print(f'IoU: {iou}')
    print(f'AP: {ap}')

    if frame:
        cap = cv2.VideoCapture(cfg.VIDEO_PATH)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        _, frame_image = cap.read()

        for box in grouped_annotations[0]:
            cv2.rectangle(frame_image, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), (0, 255, 0), 2)

        for box in grouped_annotations_with_noise[0]:
            cv2.rectangle(frame_image, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), (0, 0, 255), 2)

        cv2.imwrite(f'frame_{frame}_iou_{iou:.4f}_ap_{ap:.4f}_std_size_{std_size}_std_position_{std_position}_prob_delete_{prob_delete}_prob_similar_{prob_similar}_std_similar_{std_similar}_min_random_{min_random}_max_random_{max_random}_similar_statistic_{similar_statistic}.png', frame_image)
        cap.release()

def task1_2():
    annotations = load_annotations(cfg.ANNOTATIONS_PATH)
    grouped_annotations = group_annotations_by_frame(annotations)
    predictions = load_predictions(cfg.PREDICTIONS_PATH)
    grouped_predictions = group_annotations_by_frame(predictions)

    n_frames = len(grouped_annotations)

    print(f'Number of frames: {n_frames}')
    print(f'Number of annotations: {len(annotations)}')
    print(f'Number of predictions: {len(predictions)}')

    _, _, ap, iou = voc_eval(grouped_predictions, grouped_annotations)

    print(f'IoU: {iou}')
    print(f'AP: {ap}')


def task2():
    annotations = load_annotations(cfg.ANNOTATIONS_PATH)
    predictions = load_predictions(cfg.PREDICTIONS_PATH)

    # annotations_with_noise = create_fake_track_predictions(
    #     annotations,
    #     height=1080,
    #     width=1920,
    #     std_size=0.1,
    #     std_position=0.1,
    #     prob_delete=0.3,
    #     prob_similar=0.1,
    #     std_similar=0.2,
    #     min_random=0,
    #     max_random=1,
    #     similar_statistic=None,
    # )

    miou = iou_over_time(
        video_path=cfg.VIDEO_PATH,
        annotations=annotations,
        predictions=predictions,
        # predictions=annotations_with_noise,
        show_video=False,
        # max_frames=5,
        save_plots=True,
    )
    print(f'Mean IoU: {miou}')


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

    plt.figure(figsize=(8, 5))
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
