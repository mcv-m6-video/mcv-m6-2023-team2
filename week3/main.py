import argparse
from task_1 import task_1_1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Road Traffic Monitoring Analysis for Video Surveillance. MCV-M6-Project. Team 2'
    )

    parser.add_argument('--t1_1', action='store_true',
                        help='Task1.1 - Inference with off-the-shelf methods')

    parser.add_argument('--t1.2', action='store_true',
                        help='Task1.2 - Data annotation')

    parser.add_argument('--t1.3', action='store_true',
                        help='Task 1.3 - Fine-tune pretrained models with our own data')

    parser.add_argument('--t1.4', action='store_true',
                        help='Task 1.4 - K-Fold Cross-Validation')

    parser.add_argument('--path_results', type=str, default="./results/",
                    help='The path to the directory where the results will be stored.')

    parser.add_argument('--path_video', type=str, default="../data/AICity_S03_c010/vdo.avi",
                        help='The path to the video file to be processed.')

    parser.add_argument('--path_roi', type=str, default="../data/AICity_S03_c010/roi.jpg",
                        help='The path to the ROI file corresponding to the video to be processed.')

    parser.add_argument('--path_GT', type=str, default="../data/AICity_S03_c010/ai_challenge_s03_c010-full_annotation.xml",
                        help='The path to the ground truth file corresponding to the video to be processed.')

    parser.add_argument('--store_results', action='store_true',
                        help='Whether to store the intermediate results.')

    parser.add_argument('--make_gifs', action='store_true',
                        help='Whether to store make GIFs of the intermediate results.')

    parser.add_argument('--frames_range', type=tuple, default=(1169, 1229),
                        help='Start and end frame bitmaps to be saved (eg. for GIF creation).')  # default=(1169, 1229)

    parser.add_argument('--model', type=str, default='retina',
                        help='Which model to do inference with. Can be "faster", "retina", "YOLO", "transformer".')

    parser.add_argument('--make_video', type=bool, default=True,
                        help='Make video from segmentation.')

    parser.add_argument('--num_frames', type=int, default=99999999,
                       help='Number of frames to process.')


    args = parser.parse_args()

    print(args)

    if args.t1_1:
        print('Launching Task 1.1')
        task_1_1(args)
    elif args.t1_2:
        print('Launching Task 1.2')
        task2(args)
    elif args.t1_3:
        print('Launching Task 1.3')
        task3(args)
