import cv2
import sys

from Gaussian_background_model import fit, eval
from utils import save_metrics
from utils import plot_3d_surface

import optuna
import logging


def task1(args):

    args_t1 = {
        'path_video': args.path_video,
        'path_roi': args.path_roi,
        'path_GT': args.path_GT,
        'path_results': args.path_results,
        'viz_bboxes': args.viz_bboxes,
        'store_results': args.store_results,
        'bg_model': args.bg_model,
        'alpha': args.alpha,
        'rho': args.rho,
        'color_space': args.color_space,
        'voting': args.voting,
        'frames_range': args.frames_range,
        'make_gifs': args.make_gifs,
    }

    video = cv2.VideoCapture(args_t1['path_video'])

    N = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Video length (frames): ", N)

    W = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = [H, W]

    N_train = int(0.25 * N)
    N_val = N - N_train

    print("Train sequence length: ", N_train)
    print("Test sequence length: ", N_val)

    # Fit per-pixel Gaussian
    mean, std = fit(video, frame_size, N_train, args_t1)

    # Evaluate
    recall, precision, F1, AP, IoU = eval(video, frame_size, mean, std, N_val, args_t1)
    print(f'alpha: {args.alpha}, recall: {recall[-1]}, precision: {precision[-1]}, F1: {F1[-1]}, AP: {AP}, IoU: {IoU}')

    if args.optuna_trials is None:
        save_metrics(args_t1, recall, precision, F1, AP, IoU)

    return recall, precision, F1, AP, IoU


def task2(args):
    def objective(trial):
        alpha = trial.suggest_float('alpha', 0, 10, step=0.5)
        rho = trial.suggest_float('rho', 0, 1)
        args.alpha = alpha
        args.rho = rho
        recall, precision, F1, AP, IoU = task1(args)
        return recall[-1], precision[-1], F1[-1], AP, IoU

    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = args.optuna_study_name  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(study_name=study_name, storage=storage_name,
                                directions=['maximize', 'maximize', 'maximize', 'maximize', 'maximize'],
                                load_if_exists=True)
    if args.optuna_trials is not None:
        study.optimize(objective, n_trials=args.optuna_trials)
    else:
        plot_3d_surface(args, study, metric='recall')
        plot_3d_surface(args, study, metric='precision')
        plot_3d_surface(args, study, metric='F1')
        plot_3d_surface(args, study, metric='AP')
        plot_3d_surface(args, study, metric='IoU')

def task3(args):

    args_t2 = {
        'path_video': args.path_video,
        'path_roi': args.path_roi,
        'path_GT': args.path_GT,
        'path_results': args.path_results,
        'viz_bboxes': args.viz_bboxes,
        'store_results': args.store_results,
        'bg_model': args.bg_model,
        'alpha': args.alpha,
        'rho': args.rho,
        'color_space': args.color_space,
        'voting': args.voting,
        'frames_range': args.frames_range,
        'make_gifs': args.make_gifs,
        'subs': args.subs

    }

    video = cv2.VideoCapture(args_t2['path_video'])

    N = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Video length (frames): ", N)

    W = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = [H, W]

    N_train = int(0.25 * N)
    N_val = N - N_train

    if args_t2['subs'] == 'MOG2':
        backSub = cv2.createBackgroundSubtractorMOG2()

    elif args_t2['subs'] == 'LSBP':
        backSub = cv2.createBackgroundSubtractorLSBP()
    
    else:
        backSub = cv2.createBackgroundSubtractorKNN()
