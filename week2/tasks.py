import cv2
import sys

from Gaussian_background_model import fit, eval


def task1(args, alpha=3):

    args_t1 = {
        'path_video': args.path_video,
        'path_roi': args.path_roi,
        'path_GT': args.path_GT,
        'path_results': args.path_results,
        'N_eval': args.N_eval,
        'viz_bboxes': args.viz_bboxes,
        'store_results': args.store_results,
        'bg_model': 'non_adaptive',
        'alpha': alpha,
        'rho': 0,
        'color_space': 'grayscale',
        'voting': None,  # simple voting
    }


    video = cv2.VideoCapture(args['path_video'])

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
    return eval(video, frame_size, mean, std, args_t1)
