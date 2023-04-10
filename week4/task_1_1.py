import os
import argparse
import cv2
import logging
import time
import optuna
import matplotlib
import numpy as np

from of.optical_flow import BlockMatching
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


ESTIMATION_TYPES = ['forward', 'backward']
ERROR_FUNCTIONS = ['mse', 'mae', 'nccorr', 'nccoeff']


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Road Traffic Monitoring Analysis for Video Surveillance. MCV-M6-Project, week 4, task 1.1. Team 2'
    )

    parser.add_argument('--mode', type=str, default='dry', choices=['dry', 'grid_search', 'optuna'],
                        help='Mode to run the script')
    parser.add_argument('--path_gt_dir', type=str, default='./data/GT_OF', 
                        help='Path to the directory containing the ground truth optical flow')
    parser.add_argument('--path_frames_dir', type=str, default='./data/FRAMES_OF', 
                        help='Path to the directory containing the frames')
    parser.add_argument('--frame', type=str, default='000045',
                        help='Frame number to process')
    parser.add_argument('--optuna_trials', type=int, default=100,
                        help='Number of trials to run for Optuna')

    args = parser.parse_args()
    return args


def run_dry(gt_flow, frame_prev, frame_next):
    block_size = 24
    search_window_size = 76
    block_matching = BlockMatching(
        estimation_type='forward',
        error_function='nccorr',
        block_size=block_size,
        search_window_size=search_window_size,
    )

    times = []
    start = time.time()
    pred_flow = block_matching.estimate_optical_flow(frame_prev, frame_next)
    pred_flow = block_matching.postprocess(pred_flow)
    end = time.time()
    times.append(end - start)

    print(f"Average time: {np.mean(times)} seconds. Standard deviation: {np.std(times)} seconds")

    msen, sen = OF_MSEN(gt_flow, pred_flow, output_dir="output/test", verbose=False)
    pepn = OF_PEPN(sen)

    print(f"MSEN: {msen}\nPEPN: {pepn}%")
    visualize_optical_flow_error(gt_flow, pred_flow)
    plot_optical_flow_hsv(pred_flow[:,:,:2], pred_flow[:,:,2])
    plot_optical_flow_quiver(pred_flow, frame_prev)
    plot_optical_flow_quiver(pred_flow, frame_prev, flow_with_camera=True)
    plot_optical_flow_surface(pred_flow, frame_prev)


def run_grid_search(gt_flow, frame_prev, frame_next):
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

    for block_size in block_sizes:
        for search_window_size in search_window_sizes:
            for estimation_type in ESTIMATION_TYPES:
                for error_function in ERROR_FUNCTIONS:
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
                    plot_optical_flow_quiver(pred_flow, frame_prev, output_dir=output_dir, flow_with_camera=True)
                    plot_optical_flow_surface(pred_flow, frame_prev, output_dir=output_dir)

                    with open('task_1_1.csv', 'a') as results_csv:
                        results_csv.write(f'{block_size},{search_window_size},{estimation_type},{error_function},{msen},{pepn},{end - start}\n')


def run_optuna_search(gt_flow, frame_prev, frame_next, trials: int = 100, study_name: str = 'task_1_1_optuna_study'):
    def objective(trial):
        block_size = trial.suggest_int('block_size', 8, 128, step=4)
        search_window_size = trial.suggest_int('search_window_size', 8, 128, step=4)
        estimation_type = trial.suggest_categorical('estimation_type', ESTIMATION_TYPES)
        error_function = trial.suggest_categorical('error_function', ERROR_FUNCTIONS)

        logging.info(f"Processing with block_size={block_size}, search_window_size={search_window_size}, estimation_type={estimation_type}, error_function={error_function}")
        output_dir = os.path.join('output', 'task_1_1', f'block_size={block_size}_search_window_size={search_window_size}_estimation_type={estimation_type}_error_function={error_function}')

        block_matching = BlockMatching(
            block_size=block_size, 
            search_window_size=search_window_size, 
            estimation_type=estimation_type, 
            error_function=error_function
        )

        start = time.time()
        pred_flow = block_matching.estimate_optical_flow(frame_prev, frame_next)
        end = time.time()
        eta = end - start
        logging.info(f"Elapsed time: {eta} seconds")

        msen, sen = OF_MSEN(gt_flow, pred_flow, output_dir=output_dir, verbose=False)
        pepn = OF_PEPN(sen)

        logging.info(f"MSEN: {msen}")
        logging.info(f"PEPN: {pepn}%")

        visualize_optical_flow_error(gt_flow, pred_flow, output_dir=output_dir)
        plot_optical_flow_hsv(pred_flow[:,:,:2], pred_flow[:,:,2], output_dir=output_dir)
        plot_optical_flow_quiver(pred_flow, frame_prev, output_dir=output_dir)
        plot_optical_flow_quiver(pred_flow, frame_prev, output_dir=output_dir, flow_with_camera=True)
        plot_optical_flow_surface(pred_flow, frame_prev, output_dir=output_dir)

        with open('task_1_1.csv', 'a') as results_csv:
            results_csv.write(f'{block_size},{search_window_size},{estimation_type},{error_function},{msen},{pepn},{eta}\n')

        return msen
    
    logging.basicConfig(filename='task_1_1.log', level=logging.INFO)
    logging.info(f"Carrying out Optuna Search on frame {args.frame}")

    if not os.path.exists('task_1_1.csv'):
        with open('task_1_1.csv', 'w') as f:
            f.write('block_size,search_window_size,estimation_type,error_function,OF_MSEN,OF_PEPN,time\n')

    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(
        study_name=study_name,
        direction='minimize', 
        storage=storage_name,
        load_if_exists=True,
        )
    study.optimize(objective, n_trials=trials)

    logging.info(f"Best trial: {study.best_trial}")
    logging.info(f"Best trial params: {study.best_params}")
    logging.info(f"Best trial value: {study.best_value}")


def main(args: argparse.Namespace):
    # Load frames in "data/FRAMES_OF/XXXXXX_XX.png"
    frame_prev = cv2.imread(os.path.join(args.path_frames_dir, f"{args.frame}_10.png"), cv2.IMREAD_GRAYSCALE)
    frame_next = cv2.imread(os.path.join(args.path_frames_dir, f"{args.frame}_11.png"), cv2.IMREAD_GRAYSCALE)

    gt_flow = load_optical_flow(os.path.join(args.path_gt_dir, f"{args.frame}_10.png"))
    # print(f"Max displacement in x: {np.max(np.abs(gt_flow[:,:,0]))}")
    # print(f"Max displacement in y: {np.max(np.abs(gt_flow[:,:,1]))}")
    # plot_optical_flow_hsv(gt_flow[:,:,:2], gt_flow[:,:,2])
    # plot_optical_flow_quiver(gt_flow, frame_prev)
    # plot_optical_flow_surface(gt_flow, frame_prev)

    if args.mode == "dry":
        run_dry(gt_flow, frame_prev, frame_next)
    elif args.mode == "grid_search":
        run_grid_search(gt_flow, frame_prev, frame_next)
    elif args.mode == "optuna":
        run_optuna_search(gt_flow, frame_prev, frame_next, trials=args.optuna_trials)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    matplotlib.use('Agg') # For headless servers
    args = __parse_args()
    main(args)
