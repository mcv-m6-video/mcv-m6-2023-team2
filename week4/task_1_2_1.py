import os
import argparse
import cv2
import logging
import time
import optuna
import matplotlib 
import matplotlib.pyplot as plt

from of.optical_flow import Coarse2FineFlow
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


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Road Traffic Monitoring Analysis for Video Surveillance. MCV-M6-Project, week 4, task 1.1. Team 2'
    )

    parser.add_argument('--mode', type=str, default='optuna', choices=['dry', 'optuna'],
                        help='Mode to run the script')
    parser.add_argument('--path_gt_dir', type=str, default='./data/GT_OF', 
                        help='Path to the directory containing the ground truth optical flow')
    parser.add_argument('--path_frames_dir', type=str, default='./data/FRAMES_OF', 
                        help='Path to the directory containing the frames')
    parser.add_argument('--frame', type=str, default='000045',
                        help='Frame number to process')
    parser.add_argument('--optuna_trials', type=int, default=200,
                        help='Number of trials to run for Optuna')

    args = parser.parse_args()
    return args


def run_dry(gt_flow, frame_prev, frame_next):
    estimator = Coarse2FineFlow()

    start = time.time()
    pred_flow = estimator.estimate_optical_flow(frame_prev, frame_next)
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


def run_optuna_search(gt_flow, frame_prev, frame_next, trials=100, study_name: str = 'task_1_2_1_optuna_study'):
    def objective(trial):
        alpha = trial.suggest_float("alpha", 0.001, 0.5)
        ratio = trial.suggest_float("ratio", 0.5, 0.9)
        minWidth = trial.suggest_int("minWidth", 10, 50)
        nOuterFPIterations = trial.suggest_int("nOuterFPIterations", 1, 10)
        nInnerFPIterations = trial.suggest_int("nInnerFPIterations", 1, 10)
        nSORIterations = trial.suggest_int("nSORIterations", 1, 40)

        logging.info(f"Processing with params: alpha={alpha}, ratio={ratio}, minWidth={minWidth}, nOuterFPIterations={nOuterFPIterations}, nInnerFPIterations={nInnerFPIterations}, nSORIterations={nSORIterations}")
        output_dir = os.path.join("output", "task_1_2_1", f"alpha={alpha},ratio={ratio},minWidth={minWidth},nOuterFPIterations={nOuterFPIterations},nInnerFPIterations={nInnerFPIterations},nSORIterations={nSORIterations}")

        estimator = Coarse2FineFlow(
            alpha=alpha,
            ratio=ratio,
            minWidth=minWidth,
            nOuterFPIterations=nOuterFPIterations,
            nInnerFPIterations=nInnerFPIterations,
            nSORIterations=nSORIterations,
            colType=1,
        )

        start = time.time()
        pred_flow = estimator.estimate_optical_flow(frame_prev, frame_next)
        end = time.time()
        eta = end - start
        
        msen, sen = OF_MSEN(gt_flow, pred_flow, output_dir=output_dir, verbose=False)
        pepn = OF_PEPN(sen)

        logging.info(f"MSEN: {msen}")
        logging.info(f"PEPN: {pepn}%")
        logging.info(f"Time: {eta}")

        visualize_optical_flow_error(gt_flow, pred_flow, output_dir=output_dir)
        plot_optical_flow_hsv(pred_flow[:,:,:2], pred_flow[:,:,2], output_dir=output_dir)
        plot_optical_flow_quiver(pred_flow, frame_prev, output_dir=output_dir)
        plot_optical_flow_quiver(pred_flow, frame_prev, output_dir=output_dir, flow_with_camera=True)
        plot_optical_flow_surface(pred_flow, frame_prev, output_dir=output_dir)
        plt.close('all')
        
        with open('task_1_2_1.csv', 'a') as f:
            f.write(f"{alpha},{ratio},{minWidth},{nOuterFPIterations},{nInnerFPIterations},{nSORIterations},{msen},{pepn},{eta}\n")

        return msen, eta

    logging.basicConfig(filename='task_1_2_1.log', level=logging.INFO)
    logging.info(f"Carrying out Optuna Search on frame {args.frame}")

    if not os.path.exists('task_1_2_1.csv'):
        with open('task_1_2_1.csv', 'w') as f:
            f.write('alpha,ratio,minWidth,nOuterFPIterations,nInnerFPIterations,nSORIterations,msen,pepn,time\n')

    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(
        study_name=study_name,
        directions=["minimize", "minimize"],
        storage=storage_name,
        load_if_exists=True,
        )
    study.optimize(objective, n_trials=trials)

    logging.info(f"Best trial: {study.best_trial}")
    logging.info(f"Best trial params: {study.best_params}")
    logging.info(f"Best trial value: {study.best_value}")


def main(args: argparse.Namespace):
    # Load frames in "data/FRAMES_OF/XXXXXX_XX.png"
    frame_prev = cv2.imread(os.path.join(args.path_frames_dir, f"{args.frame}_10.png"))
    frame_next = cv2.imread(os.path.join(args.path_frames_dir, f"{args.frame}_11.png"))

    gt_flow = load_optical_flow(os.path.join(args.path_gt_dir, f"{args.frame}_10.png"))

    if args.mode == "dry":
        run_dry(gt_flow, frame_prev, frame_next)
    elif args.mode == "optuna":
        run_optuna_search(gt_flow, frame_prev, frame_next, trials=args.optuna_trials)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    matplotlib.use('Agg') # For headless servers
    args = __parse_args()
    main(args)
