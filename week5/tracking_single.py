"""
Performs multi-target tracking using a single camera.

It scans for all sequences and cameras in the dataset and performs tracking on each of them.
"""

import sys
import argparse

from utils import load_config

import os
import argparse
import numpy as np
import time
import cv2
from tqdm import tqdm
from typing import Dict, List

from utils import (
    load_config,
    load_predictions,
    group_annotations_by_frame,
    filter_annotations,
    non_maxima_suppression,
    load_optical_flow,
)
from tracking.tracking_utils import (
    viz_tracking,
    filter_by_area,
    filter_parked,
    store_trackers_list,
    TrackHandlerOverlap,
)
from bounding_box import BoundingBox


def post_tracking(cfg, video_width, video_height, fps, trackers_list, frames_list):
    if cfg["filter_by_area"]:
        trackers_list = filter_by_area(cfg, trackers_list)
    if cfg["filter_parked"]:
        trackers_list = filter_parked(cfg, trackers_list)

    store_trackers_list(trackers_list, cfg["save_tracking_path"])
    # visualize tracking
    viz_tracking(cfg["save_video_path"], video_width, video_height, fps, trackers_list, frames_list)


def tracking_by_maximum_overlap(
    cfg: Dict,
    detections: List[BoundingBox],
    max_frame_skip: int = 0,
    min_iou: float = 0.5,
):
    track_handler = TrackHandlerOverlap(max_frame_skip=max_frame_skip, min_iou=min_iou)
    video = cv2.VideoCapture(cfg["path_sequence"])
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    total_time = 0.0
    trackers_list = []
    frames_list = []

    for frame_id in tqdm(range(total_frames-1)):
        ret, frame = video.read()
        if not ret:
            break

        # Read detections
        if len(detections) <= frame_id:
            frame_detections = []
        else:
            frame_detections = detections[frame_id]

        start_time = time.time()
        track_handler.update_tracks(frame_detections, frame_id)
        cycle_time = time.time() - start_time
        total_time += cycle_time

        # Visualize tracking
        frame_detections = []
        for track in track_handler.live_tracks:
            detection, _ = track.last_detection()
            frame_detections.append(detection)
        trackers_list.append(frame_detections)
        frames_list.append(frame)

        total_frames += 1

    print("Total Tracking took: %.3f for %d frames, @ %.1f FPS" % (total_time, total_frames, total_frames/total_time))
    post_tracking(cfg, video_width, video_height, fps, trackers_list, frames_list)


def tracking_by_kalman_filter(
    cfg,
    detections,
    video_max_frames: int = 9999,
    video_frame_sampling: int = 1,
    tracking_max_age: int = 1,
    tracking_min_hits: int = 3,
    tracking_iou_threshold: float = 0.3,
    of_use: bool = False,
    of_data_path: str = None,
):
    # Reference: https://github.com/telecombcn-dl/2017-persontyle/blob/master/sessions/tracking/tracking_kalman.ipynb

    total_time = 0.0
    trackers_list = []
    frames_list = []

    # Only for display

    video = cv2.VideoCapture(cfg["path_sequence"])
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    max_frames = min(video_max_frames, total_frames)

    if of_use:
        from tracking.sort_of import Sort
        print("Imported SORT with Optical Flow.")
    else:
        from tracking.sort import Sort
        print("Imported SORT.")
    mot_tracker = Sort(max_age=tracking_max_age, min_hits=tracking_min_hits, iou_threshold=tracking_iou_threshold)

    for idx_frame in tqdm(range(0, max_frames-1, video_frame_sampling), desc="Computing tracking..."):
        # read the frame
        ret, frame = video.read()

        if not ret:
            break

        # Read detections
        if len(detections) <= idx_frame:
            dets = []
        else:
            dets = detections[idx_frame]

        # Convert to proper array for the tracker input
        dets = np.array([d.coordinates for d in dets])
        # If no detections, add empty array
        if len(dets) == 0:
            dets = np.empty((0, 5))

        start_time = time.time()
        # Update tracker
        if of_use and of_data_path is not None:
            pred_flow = load_optical_flow(os.path.join(of_data_path, f"{idx_frame}.png"))

            # Update tracker
            trackers = mot_tracker.update(dets, pred_flow)
        else:
            # Update tracker
            trackers = mot_tracker.update(dets)
        cycle_time = time.time() - start_time
        total_time += cycle_time

        trackers = [BoundingBox(*t, int(idx_frame)) for t in trackers]
        trackers_list.append(trackers)
        frames_list.append(frame)

        total_frames += 1

    print("Total Tracking took: %.3f for %d frames, @ %.1f FPS" % (total_time, total_frames, total_frames/total_time))
    post_tracking(cfg, video_width, video_height, fps, trackers_list, frames_list)


# Detections are assumed to be stored in the following directory structure:
# | detections
#     | seqXXX
#         | cYYY
#         | cZZZ
#     | seqYYY
#         | cYYY
#     ....
def scan_sequences(cfg):
    # For each tracked video, the output path will be like this:
    # ./week5/data/trackers/mot_challenge/parabellum-train/<tracking_type>/data/{seq}_{camera}.txt

    # Scan all sequences in directory cfg["detections_dir"]
    for seq in os.listdir(cfg["detections_dir"]):
        seq_name = os.path.basename(seq)
        # Scan all cameras in directory cfg["detections_dir"]/seq
        for camera in os.listdir(os.path.join(cfg["detections_dir"], seq)):
            camera_name = os.path.basename(camera)
            # Load detections

            detections_path = f"{os.path.join(cfg['detections_dir'], seq, camera)}/detections.txt"

            # Load and process detections
            confidence_threshold = 0.6
            detections = load_predictions(detections_path)
            detections = filter_annotations(detections, confidence_thr=confidence_threshold)
            detections = group_annotations_by_frame(detections)
            detections = non_maxima_suppression(detections)

            exp_name = f'{seq_name.lower()}_{camera_name}'
            method_name = cfg["tracking_type"] + f'_filtArea{cfg["filter_by_area"]}' + f'_filtParked{cfg["filter_parked"]}'
            cfg["save_tracking_path"] = os.path.join(cfg["path_tracking_data"], method_name, "data", exp_name + ".txt")
            os.makedirs(os.path.dirname(cfg["save_tracking_path"]), exist_ok=True)
            cfg["path_sequence"] = os.path.join(cfg["dataset_dir"], seq_name, camera_name, "vdo.avi")

            cfg["save_video_path"] = os.path.join(cfg["path_results"], f"tracking_single_{method_name}_{exp_name}.mp4")
            os.makedirs(cfg["path_results"], exist_ok=True)

            if cfg["tracking_type"] == "kalman":

                # BEST Kalman filter Tracking parameters (found in WK 3)
                max_age = 50
                min_hits = 3
                min_iou = 0.3

                tracking_by_kalman_filter(
                    cfg=cfg,
                    detections=detections,
                    # video_max_frames=45,  # TODO: Comment this line to use all frames!
                    tracking_max_age=max_age,
                    tracking_min_hits=min_hits,
                    tracking_iou_threshold=min_iou,
                )
            elif cfg["tracking_type"] == "kalman_of":
                # BEST Kalman filter Tracking parameters (found in WK 3)
                max_age = 50
                min_hits = 3
                min_iou = 0.3

                tracking_by_kalman_filter(
                    cfg=cfg,
                    detections=detections,
                    # video_max_frames=45,  # TODO: Comment this line to use all frames!
                    tracking_max_age=max_age,
                    tracking_min_hits=min_hits,
                    tracking_iou_threshold=min_iou,
                    of_use=True,
                    of_data_path=cfg["of_data_path"],
                )
            elif cfg["tracking_type"] == "overlap":
                # BEST Tracking by Overlap parameters (found in WK 3)
                max_frame_skip = 10
                min_iou = 0.4
                tracking_by_maximum_overlap(
                    cfg=cfg,
                    detections=detections,
                    max_frame_skip=max_frame_skip,
                    min_iou=min_iou,
                )
            else:
                raise ValueError(f"Unknown tracking type: {cfg['tracking_type']}. Valid values: 'kalman', 'kalman_of'.")

            print(f"Tracking results saved in: {cfg['save_tracking_path']}")
            print(f"Tracking video saved in: {cfg['save_video_path']}")
            print("-" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/tracking_single.yaml")
    args = parser.parse_args(sys.argv[1:])

    cfg = load_config(args.config)

    scan_sequences(cfg)
