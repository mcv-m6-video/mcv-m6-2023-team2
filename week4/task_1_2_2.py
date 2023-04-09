import sys
import os
import argparse
import cv2
import time
import matplotlib
import numpy as np

from utils_w4 import load_optical_flow
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

sys.path.append("unimatch")
from evaluate_flow import flow_unimatch


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Road Traffic Monitoring Analysis for Video Surveillance. MCV-M6-Project, week 4, task 1.2.2 Team 2'
    )

    parser.add_argument('--method', type=str, default='', choices=['maskflownet', 'unimatch'],
                        help='Method to compute OF.')
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


def run_dry(gt_flow, frame_prev, frame_next, args):

    start = time.time()

    pred_flow, extra_out = estimate_flow[args.method](frame_prev, frame_next)
    if pred_flow.shape[2] == 2:
        pred_flow = np.stack((pred_flow[...,0], pred_flow[...,1], np.ones_like(pred_flow[...,0])), axis=2)

    end = time.time()
    print(f"Time: {end - start}")

    msen, sen = OF_MSEN(gt_flow, pred_flow, output_dir=f"output/{args.method}", verbose=False)
    pepn = OF_PEPN(sen)

    print(f"MSEN: {msen}\nPEPN: {pepn}%")
    visualize_optical_flow_error(gt_flow, pred_flow, args.frame)
    output_dir = os.path.join("task_1_2_2", args.method)
    plot_optical_flow_hsv(pred_flow[:,:,:2], pred_flow[:,:,2], output_dir=output_dir)
    plot_optical_flow_quiver(pred_flow, frame_prev, output_dir=output_dir)
    plot_optical_flow_quiver(pred_flow, frame_prev, flow_with_camera=True, output_dir=output_dir)
    plot_optical_flow_surface(pred_flow, frame_prev, output_dir=output_dir)


def main(args: argparse.Namespace):
    # Load frames in "data/FRAMES_OF/XXXXXX_XX.png"
    frame_prev = cv2.imread(os.path.join(args.path_frames_dir, f"{args.frame}_10.png"))
    frame_next = cv2.imread(os.path.join(args.path_frames_dir, f"{args.frame}_11.png"))

    gt_flow = load_optical_flow(os.path.join(args.path_gt_dir, f"{args.frame}_10.png"))

    run_dry(gt_flow, frame_prev, frame_next, args)


if __name__ == "__main__":
    matplotlib.use('Agg') # For headless servers
    args = __parse_args()
    main(args)
