
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
    parser.add_argument('--sequence', type=str, help='Sequence to process, e.g. "S03"')
    parser.add_argument('--cameras', help='List of cameras to process, e.g. "c010,c011,c012,c013,c014,c015"',
                        type=lambda s: [item for item in s.split(',')])
    parser.add_argument('--max_frames', type=int, help='Maximum number of frames to process')
    parser.add_argument('--path_results', type=str, default="./results/",
                        help='The path to the directory where the results will be stored.')

    args = parser.parse_args()
    return args


def save_optical_flow_blockmatching(
    args,
    sequence: str,
    camera: str,
    video_max_frames: int = 9999,
    video_frame_sampling: int = 1,
    ):
    path_results = os.path.join(args.path_results, f"video_of_unimatch_{sequence}_{camera}")
    # path_results = os.path.join(args.path_results, "video_of_block_matching")
    os.makedirs(path_results, exist_ok=True)
    video = cv2.VideoCapture(args.path_sequence)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    max_frames = min(video_max_frames, total_frames)

    _, frame_prev = video.read()
    frame_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)

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
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pred_flow = block_matching.estimate_optical_flow(frame_prev, frame)
        pred_flow = block_matching.postprocess(pred_flow)
        pred_flow = convert_image_to_optical_flow(pred_flow)

        # Save image
        cv2.imwrite(os.path.join(path_results, f"{idx_frame}.png"), pred_flow)
        # frames_of.append(pred_flow)

        frame_prev = frame

    # np.save(results_of_file, frames_of)
    # results_of_file.close()
    # print(f"Optical flow saved successfully! at {filename}")
    print(f"Optical flow (BlockMatching) saved successfully at {path_results} !")


def save_optical_flow_unimatch(
    path_sequence: str,
    path_results: str,
    video_max_frames: int = 9999,
    video_frame_sampling: int = 1,
    ):
    os.makedirs(path_results, exist_ok=True)
    video = cv2.VideoCapture(path_sequence)
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
        cv2.imwrite(os.path.join(path_results, f"{idx_frame}.png"), pred_flow)

        frame_prev = frame

    print(f"Optical flow (Unimatch) saved successfully at {path_results} !")


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


def batch_inference_unimatch(args):
    sequence = args.sequence
    cameras = args.cameras

    for camera in cameras:
        print("Processing sequence: ", sequence, ", camera: ", camera)

        path_results = os.path.join(args.path_results, f"video_of_unimatch_{sequence}_{camera}")
        print("Saving results at: ", path_results)

        path_sequence = f"../data/aic19/train/{sequence}/{camera}/vdo.avi"
        print("Processing video at: ", path_sequence)

        save_optical_flow_unimatch(
            path_sequence,
            path_results,
            args.max_frames,
            video_frame_sampling=1,
        )

        print("Done processing sequence: ", sequence, ", camera: ", camera)
        print("Done with video at: ", path_sequence)
        print("--------------------------------------------------")


if __name__ == "__main__":

    args = __parse_args()

    # save_optical_flow_blockmatching(
    #     args,
    #     video_max_frames=5,
    #     video_frame_sampling=1,
    # )

    print(args)

    batch_inference_unimatch(args)

    # save_optical_flow_unimatch(
    #     args,
    #     video_max_frames=args.max_frames,
    #     video_frame_sampling=1,
    # )
    # inference_of_video_unimatch()
