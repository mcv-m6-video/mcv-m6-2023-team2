import os
import argparse
import cv2
import logging
import time

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
    logging.basicConfig(filename='task_1_1.log', level=logging.INFO)
    logging.info(f"Carrying out Grid Search on frame {args.frame}")

    if os.path.exists('task_1_1.csv'):
        with open('task_1_1.csv', 'r') as f:
            # Check existing results so we don't repeat them
            existing_results = " ".join(f.read().splitlines())
    else:
        with open('task_1_1.csv', 'w') as f:
            f.write('block_size,search_window_size,estimation_type,error_function,OF_MSEN,OF_PEPN,time\n')
        existing_results = ""

    block_sizes = [8, 16, 32, 64, 128]
    search_window_sizes = [8, 16, 32, 64, 128]
    estimation_types = ['forward', 'backward']
    error_functions = ['mse', 'mae']

    # Load frames in "data/FRAMES_OF/XXXXXX_XX.png"
    frame_prev = cv2.imread(os.path.join(args.path_frames_dir, f"{args.frame}_10.png"), cv2.IMREAD_GRAYSCALE)
    frame_next = cv2.imread(os.path.join(args.path_frames_dir, f"{args.frame}_11.png"), cv2.IMREAD_GRAYSCALE)

    gt_flow = load_optical_flow(os.path.join(args.path_gt_dir, f"{args.frame}_10.png"))
    
    #TODO: Save metrics and plot results of the grid search
    for block_size in block_sizes:
        for search_window_size in search_window_sizes:
            for estimation_type in estimation_types:
                for error_function in error_functions:
                    if f'{block_size},{search_window_size},{estimation_type},{error_function}' in existing_results:
                        logging.info(f"Skipping block_size={block_size}, search_window_size={search_window_size}, estimation_type={estimation_type}, error_function={error_function}")
                        continue

                    logging.info(f"Processing with block_size={block_size}, search_window_size={search_window_size}, estimation_type={estimation_type}, error_function={error_function}")
                    output_dir = os.path.join('output', f'block_size={block_size}_search_window_size={search_window_size}_estimation_type={estimation_type}_error_function={error_function}')

                    block_matching = BlockMatching(block_size=block_size, search_window_size=search_window_size, estimation_type=estimation_type, error_function=error_function)
                    start = time.time()
                    pred_flow = block_matching.estimate_optical_flow(frame_prev, frame_next)
                    end = time.time()
                    logging.info(f"Elapsed time: {end - start} seconds")

                    msen, sen = OF_MSEN(gt_flow, pred_flow, output_dir=output_dir, verbose=False)
                    pepn = OF_PEPN(sen)

                    logging.info(f"MSEN: {msen}")
                    logging.info(f"PEPN: {pepn}%")

                    visualize_optical_flow_error(gt_flow, pred_flow, output_dir=output_dir)
                    plot_optical_flow_hsv(pred_flow[:,:,:2], pred_flow[:,:,2], output_dir=output_dir)
                    plot_optical_flow_quiver(pred_flow, frame_prev, output_dir=output_dir)

                    with open('task_1_1.csv', 'a') as results_csv:
                        results_csv.write(f'{block_size},{search_window_size},{estimation_type},{error_function},{msen},{pepn},{end - start}\n')

    # block_matching = BlockMatching()
    # pred_flow = block_matching.estimate_optical_flow(frame_prev, frame_next)

    # msen, sen = OF_MSEN(gt_flow, pred_flow, frame=args.frame, verbose=False)
    # pepn = OF_PEPN(sen)

    # print(f"MSEN: {msen}\nPEPN: {pepn} %")
    # visualize_optical_flow_error(gt_flow, pred_flow, args.frame)
    # plot_optical_flow_hsv(pred_flow[:,:,:2], pred_flow[:,:,2])
    # plot_optical_flow_quiver(pred_flow, frame_prev)

    # plot_optical_flow_hsv(gt_flow[:,:,:2], gt_flow[:,:,2])
    # plot_optical_flow_quiver(gt_flow, frame_prev)


if __name__ == "__main__":
    args = __parse_args()
    main(args)
