
import sys
sys.path.append("unimatch")
# from evaluate_flow import setup_model
# from evaluate_flow import inference_flow
from of.optical_flow import BlockMatching
from utils_w4 import convert_image_to_optical_flow

import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Road Traffic Monitoring Analysis for Video Surveillance. MCV-M6-Project, week 3, task 2.2. Team 2'
    )

    # parser.add_argument('--path_sequence', type=str, default="../data/AICity_S03_c010/vdo.avi",
    #                     help='Path to the directory where the sequence is stored.')
    parser.add_argument('--path_sequence', type=str, default="./data/AICity_data/train/S03/c010/vdo.avi",
                        help='Path to the directory where the sequence is stored.')
    parser.add_argument('--path_results', type=str, default="./results/video_of",
                        help='The path to the directory where the results will be stored.')

    args = parser.parse_args()
    return args


def save_optical_flow(args, video_max_frames: int = 9999, video_frame_sampling: int = 1):
    os.makedirs(args.path_results, exist_ok=True)
    video = cv2.VideoCapture(args.path_sequence)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    max_frames = min(video_max_frames, total_frames)

    _, frame_prev = video.read()

    # frames_of = []
    # filename = os.path.join(args.path_results, f"s03_optical_flow_unimatch_{max_frames}_frames.npy")
    # results_of_file = open(filename, "wb")

    block_matching = BlockMatching(
        estimation_type='forward',
        error_function='nccorr',
        block_size=24,
        search_window_size=76,
    )

    for idx_frame in tqdm(range(0, max_frames - 1, video_frame_sampling), desc="Saving optical flow..."):
        # read the frame
        # video.set(cv2.CAP_PROP_POS_FRAMES, idx_frame)
        ret, frame = video.read()

        if not ret:
            break
            
        # Canvia el model
        pred_flow = block_matching.estimate_optical_flow(frame_prev, frame)
        pred_flow = block_matching.postprocess(pred_flow)
        pred_flow = convert_image_to_optical_flow(pred_flow)

        # Save image
        cv2.imwrite(os.path.join(args.path_results, f"{idx_frame}.png"), pred_flow)
        # frames_of.append(pred_flow)

        frame_prev = frame

    # np.save(results_of_file, frames_of)
    # results_of_file.close()
    # print(f"Optical flow saved successfully! at {filename}")


# def inference_of_video():
#     model = setup_model()

#     inference_flow(
#         model=model,
#         inference_dir=None,
#         inference_video=args.path_sequence,
#         save_flo_flow=True,
#         output_path=args.path_results,
#     )


if __name__ == "__main__":
    args = __parse_args()
    save_optical_flow(
        args,
        video_max_frames=5,
        video_frame_sampling=1,
    )
    # inference_of_video()

