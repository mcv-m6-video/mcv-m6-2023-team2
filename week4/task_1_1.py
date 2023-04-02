import os
import argparse

import cv2
from tqdm import tqdm
from typing import List

from of.optical_flow import BlockMatching
from utils import load_optical_flow
from of.of_utils import (
    visualize_optical_flow_error,
    plot_optical_flow_hsv,
    plot_optical_flow_quiver,
)
from metrics import (
    OF_MSEN,
    OF_PEPN,
)


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Road Traffic Monitoring Analysis for Video Surveillance. MCV-M6-Project, week 4, task 1.1. Team 2'
    )

    parser.add_argument('--path_gt_dir', type=str, default='./data/GT_OF', 
                        help='Path to the directory containing the ground truth optical flow')
    parser.add_argument('--path_frames_dir', type=str, default='./data/FRAMES_OF', 
                        help='Path to the directory containing the frames')
    parser.add_argument('--frame', type=str, default='000045',
                        help='Frame number to process')

    args = parser.parse_args()
    return args


def main(args: argparse.Namespace):
    # Load frames in "data/FRAMES_OF/XXXXXX_XX.png"
    frame_prev = cv2.imread(os.path.join(args.path_frames_dir, f"{args.frame}_10.png"), cv2.IMREAD_GRAYSCALE)
    frame_next = cv2.imread(os.path.join(args.path_frames_dir, f"{args.frame}_11.png"), cv2.IMREAD_GRAYSCALE)

    gt_flow = load_optical_flow(os.path.join(args.path_gt_dir, f"{args.frame}_10.png"))
    
    block_matching = BlockMatching()
    pred_flow = block_matching.estimate_optical_flow(frame_prev, frame_next)

    msen, sen = OF_MSEN(gt_flow, pred_flow, frame=args.frame, verbose=False)
    pepn = OF_PEPN(sen)

    print(f"MSEN: {msen}\nPEPN: {pepn} %")
    visualize_optical_flow_error(gt_flow, pred_flow, args.frame)
    plot_optical_flow_hsv(pred_flow[:,:,:2], pred_flow[:,:,2])
    plot_optical_flow_quiver(pred_flow, frame_prev)
    # plot_optical_flow_hsv(gt_flow[:,:,:2], gt_flow[:,:,2])
    # plot_optical_flow_quiver(gt_flow, frame_prev)


if __name__ == "__main__":
    args = __parse_args()
    main(args)
