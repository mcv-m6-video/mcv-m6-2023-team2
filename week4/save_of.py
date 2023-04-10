
import sys

from of.optical_flow import BlockMatching
from utils_w4 import convert_image_to_optical_flow

sys.path.append("unimatch")
from evaluate_flow import (
    setup_model,
    flow_unimatch_single,
    inference_flow,
)

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
    parser.add_argument('--path_sequence', type=str, default="../data/aic19/train/S03/c010/vdo.avi",
                        help='Path to the directory where the sequence is stored.')
    parser.add_argument('--path_results', type=str, default="./results/",
                        help='The path to the directory where the results will be stored.')

    args = parser.parse_args()
    return args


def save_optical_flow_blockmatching(args, video_max_frames: int = 9999, video_frame_sampling: int = 1):
    path_results = os.path.join(args.path_results, "video_of_block_matching")
    os.makedirs(path_results, exist_ok=True)
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
    print(f"Optical flow (BlockMatching) saved successfully at {args.path_results} !")


def save_optical_flow_unimatch(args, video_max_frames: int = 9999, video_frame_sampling: int = 1):
    path_results = os.path.join(args.path_results, "video_of_block_unimatch")
    os.makedirs(path_results, exist_ok=True)
    video = cv2.VideoCapture(args.path_sequence)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    max_frames = min(video_max_frames, total_frames)

    _, frame_prev = video.read()

    unimatch_model = setup_model()

    for idx_frame in tqdm(range(0, max_frames - 1, video_frame_sampling), desc="Saving optical flow (Unimatch)..."):
        ret, frame = video.read()

        if not ret:
            break

        pred_flow, _ = flow_unimatch_single(frame_prev, frame, unimatch_model,)
        if pred_flow.shape[2] == 2:
            pred_flow = np.stack((pred_flow[...,0], pred_flow[...,1], np.ones_like(pred_flow[...,0])), axis=2)

        # Save image
        pred_flow = convert_image_to_optical_flow(pred_flow)
        cv2.imwrite(os.path.join(args.path_results, f"{idx_frame}.png"), pred_flow)

        frame_prev = frame

    print(f"Optical flow (Unimatch) saved successfully at {args.path_results} !")


def inference_of_video_unimatch():
    model = setup_model()

    inference_flow(
        model=model,
        inference_dir=None,
        inference_video=args.path_sequence,
        save_flo_flow=True,
        output_path=args.path_results,
    )
    print(f"Optical flow (Unimatch) for the video saved succesfully!")


if __name__ == "__main__":
    args = __parse_args()
    # save_optical_flow_blockmatching(
    #     args,
    #     video_max_frames=5,
    #     video_frame_sampling=1,
    # )
    save_optical_flow_unimatch(
        args,
        video_max_frames=5,
        video_frame_sampling=1,
    )
    inference_of_video_unimatch()
