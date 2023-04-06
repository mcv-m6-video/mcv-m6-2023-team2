import sys
import os
import argparse
import cv2
import logging
import time
import optuna
import matplotlib 
import matplotlib.pyplot as plt

from utils import load_optical_flow
from of.of_utils import (
    visualize_optical_flow_error,
    plot_optical_flow_hsv,
    plot_optical_flow_quiver,
    plot_optical_flow_surface,
)
from metrics import (
    OF_MSEN,
    OF_PEPN,
)

sys.path.append("MaskFlownet")
from predict_new_data import flow_maskflownet


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Road Traffic Monitoring Analysis for Video Surveillance. MCV-M6-Project, week 4, task 1.2.2 Team 2'
    )

    parser.add_argument('--mode', type=str, default='optuna', choices=['dry', 'optuna'],
                        help='Mode to run the script')
    parser.add_argument('--path_gt_dir', type=str, default='../data/GT_OF',
                        help='Path to the directory containing the ground truth optical flow')
    parser.add_argument('--path_frames_dir', type=str, default='../data/FRAMES_OF',
                        help='Path to the directory containing the frames')
    parser.add_argument('--frame', type=str, default='000045',
                        help='Frame number to process')

    args = parser.parse_args()
    return args


estimate_flow = {
    'maskflownet': flow_maskflownet,
    'unimatch': flow_unimatch,
}


def run_dry(gt_flow, frame_prev, frame_next):

    start = time.time()

    pred_flow, extra_out = estimate_flow[args.method](frame_prev, frame_next)

    end = time.time()
    print(f"Time: {end - start}")

    msen, sen = OF_MSEN(gt_flow, pred_flow, output_dir="output/test", verbose=False)
    pepn = OF_PEPN(sen)

    print(f"MSEN: {msen}\nPEPN: {pepn}%")
    visualize_optical_flow_error(gt_flow, pred_flow, args.frame)
    plot_optical_flow_hsv(pred_flow[:,:,:2], pred_flow[:,:,2])
    plot_optical_flow_quiver(pred_flow, frame_prev)
    plot_optical_flow_quiver(pred_flow, frame_prev, flow_with_camera=True)
    plot_optical_flow_surface(pred_flow, frame_prev)


def main(args: argparse.Namespace):
    # Load frames in "data/FRAMES_OF/XXXXXX_XX.png"
    frame_prev = cv2.imread(os.path.join(args.path_frames_dir, f"{args.frame}_10.png"))
    frame_next = cv2.imread(os.path.join(args.path_frames_dir, f"{args.frame}_11.png"))

    gt_flow = load_optical_flow(os.path.join(args.path_gt_dir, f"{args.frame}_10.png"))

    run_dry(gt_flow, frame_prev, frame_next)


if __name__ == "__main__":
    matplotlib.use('Agg') # For headless servers
    args = __parse_args()
    main(args)
