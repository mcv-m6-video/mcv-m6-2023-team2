import cv2
import sys

from Gaussian_background_model import fit, eval
from utils import save_metrics


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
        'color_space': 'grayscale',
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
    save_metrics(args_t1, recall, precision, F1, AP, IoU)
    print(f'alpha: {args.alpha}, recall: {recall[-1]}, precision: {precision[-1]}, F1: {F1[-1]}, AP: {AP}, IoU: {IoU}')
