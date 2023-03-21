import argparse
from tasks import (
    task1,
    task2,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Road Traffic Monitoring Analysis for Video Surveillance. MCV-M6-Project. Team 2'
    )

    parser.add_argument('--t1', action='store_true',
                        help='Task1 - background estimation with non-adaptive Gaussian model')

    parser.add_argument('--t2', action='store_true',
                        help='Task2 - background estimation with adaptive Gaussian model')

    parser.add_argument('--t3', action='store_true',
                        help='Task 3 - explore and evaluate a SOTA method')

    parser.add_argument('--t4', action='store_true',
                        help='Task 4 - background estimation with non-adaptive, color-aware, multidimensional Gaussian model')

    parser.add_argument('--path_video', type=str, default="../data/AICity_S03_c010/vdo.avi",
                        help='The path to the video file to be processed.')

    parser.add_argument('--path_roi', type=str, default="../data/AICity_S03_c010/roi.jpg",
                        help='The path to the ROI file corresponding to the video to be processed.')

    parser.add_argument('--path_GT', type=str, default="../data/AICity_S03_c010/ai_challenge_s03_c010-full_annotation.xml",
                        help='The path to the ground truth file corresponding to the video to be processed.')

    parser.add_argument('--path_results', type=str, default="./results/",
                        help='The path to the directory where the results will be stored.')

    parser.add_argument('--bg_model', type=str, default='non_adaptive',
                        help='Model to be used for background estimation (non_adaptive or adaptive).')

    parser.add_argument('--alpha', type=float, default=3,
                        help='alpha parameter')

    parser.add_argument('--rho', type=float, default=0.03,
                        help='rho parameter')

    parser.add_argument('--viz_bboxes', action='store_true',
                        help='Whether to visualize the bounding boxes.')

    parser.add_argument('--color_space', type=str, default='grayscale',
                        help='Color space to be used for background estimation.')

    parser.add_argument('--voting', type=str, default='unanimous',
                        help='Voting scheme to be used for background estimation.')

    parser.add_argument('--store_results', action='store_true',
                        help='Whether to store the intermediate results.')

    parser.add_argument('--make_gifs', action='store_true',
                        help='Whether to store make GIFs of the intermediate results.')

    parser.add_argument('--frames_range', type=tuple, default=(1169, 1229),
                        help='Start and end frame bitmaps to be saved (eg. for GIF creation).')  # default=(1169, 1229)

    parser.add_argument('--optuna_study_name', type=str, default='t2-study',
                        help='Name of the Optuna study (will be saved in disk as a DB).')

    parser.add_argument('--optuna_trials', type=int, default=None,
                        help='Number of trials for Optuna optimization.')

    args = parser.parse_args()

    print(args)

    if args.t1:
        print('Launching Task 1')
        task1(args)

    if args.t2:
        print('Launching Task 2')
        task2(args)

    # if args.t3:
    #     print('Launching Task 3')
    #     task3()

    # if args.t4:
    #     print('Launching Task 4')
    #     t4_run()
