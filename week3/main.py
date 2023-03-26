import argparse
from task_1_inference import task_1_1_inference
from task_1_evaluation import task_1_1_evaluation


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

    parser.add_argument('--path_det', type=str, default="./results/faster/detections.txt",
                        help='The path to the detection file corresponding to the video to be processed.')

    parser.add_argument('--store_results', action='store_true',
                        help='Whether to store the intermediate results.')

    parser.add_argument('--make_gifs', action='store_true',
                        help='Whether to store make GIFs of the intermediate results.')

    parser.add_argument('--mode', type=str, default='inference',
                        help='Which mode to execute in task 1.1. Can be "inference", "evaluation", "visualization".')

    parser.add_argument('--model', type=str, default='retina',
                        help='Which model to do inference with. Can be "faster", "retina", "yolo", "transformer".')

    parser.add_argument('--min_iou', type=float, default=0.5,
                       help='Minimum IoU for a detection to be considered a TP.')

    parser.add_argument('--min_conf', type=float, default=0.5,
                       help='Minimum confidence of a detection for it to be considere for evaluation.')

    parser.add_argument('--format', type=str, default='aicity',
                        help='Which format to use to store detections. Can be "aicity", "kitti". The latter is only suppoted for "faster" and "retina" models.')

    parser.add_argument('--num_frames', type=int, default=99999999,
                       help='Number of frames to process.')


    args = parser.parse_args()

    print(args)

    if args.t1_1:
        print(f'Launching Task 1.1. Mode: {args.mode}. Model: {args.model}')
        if args.mode == 'inference':
            task_1_1_inference(args)
        elif args.mode == 'evaluation':
            task_1_1_evaluation(args)
        elif args.mode == 'visualization':
            pass
    elif args.t1_2:
        print('Launching Task 1.2')
        task2(args)
    elif args.t1_3:
        print('Launching Task 1.3')
        task3(args)
