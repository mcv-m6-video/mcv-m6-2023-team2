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
from tracking.tracking_utils import TrackingViz
from bounding_box import BoundingBox


def store_trackers_list(trackers_list: List[List[BoundingBox]], save_tracking_path: str):
    # trackers_list is a list of lists, where each list contains the bounding boxes of a frame
    results_file = open(save_tracking_path, "w")
    for trackers in trackers_list:
        for d in trackers:
            # Save tracking with bounding boxes in MOT Challenge format:
            # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, -1, -1, -1, -1
            results_file.write(
                f"{d.frame+1},{d.track_id},{d.x1},{d.y1},{d.x2-d.x1},{d.y2-d.y1},{d.confidence if d.confidence else '-1'},-1,-1,-1\n"
            )
    results_file.close()


def filter_by_id(keep_id, trackers_list: List[List[BoundingBox]]):
    filtered_trackers_list = []
    for trackers in trackers_list:
        trackers_filt = [d for d in trackers if d.track_id in keep_id]
        filtered_trackers_list.append(trackers_filt)
    return filtered_trackers_list


def filter_by_area(cfg: Dict, trackers_list: List[List[BoundingBox]]):
    # keep track of the area of each track over time
    trackId_area = {}
    for trackers in trackers_list:
        for d in trackers:
            if d.track_id not in trackId_area:
                trackId_area[d.track_id] = []
            trackId_area[d.track_id].append(d.area)

    keep_id = set()
    # Compute the average area of each track
    for track_id in trackId_area:
        trackId_area[track_id] = np.mean(trackId_area[track_id])
        # Keep only the tracks with an area above a threshold
        if trackId_area[track_id] >= cfg["filter_area_threshold"]:
            keep_id.add(track_id)

    # Finally, store only the tracks that are not parked
    filtered_trackers_list = filter_by_id(keep_id, trackers_list)
    return filtered_trackers_list


def filter_parked(cfg: Dict, trackers_list: List[List[BoundingBox]]):
    """ Discards parked vehicles """
    # Compute the center of the bounding box for each frame and track
    bbox_center = {}  # track_id -> list of (x,y) coordinates
    for trackers in trackers_list:
        for d in trackers:
            if d.track_id not in bbox_center:
                bbox_center[d.track_id] = []
            bbox_center[d.track_id].append([d.center_x, d.center_y])

    # Compute the std of the bounding boxes center for each track
    keep_id = set()
    for track_id in bbox_center:
        bbox_center[track_id] = np.std(bbox_center[track_id], axis=0)
        if bbox_center[track_id][0] >= cfg["filter_parked_threshold"] or bbox_center[track_id][1] >= cfg["filter_parked_threshold"]:
            keep_id.add(track_id)

    # Finally, store only the tracks that are not parked
    filtered_trackers_list = filter_by_id(keep_id, trackers_list)
    return filtered_trackers_list


def viz_tracking(
    output_video_path: str,
    video_width: int,
    video_height: int,
    fps: int,
    trackers_list: List[List[BoundingBox]],
    frames_list: List[np.ndarray],
):
    tracking_viz = TrackingViz(output_video_path, video_width, video_height, fps)
    for trackers, frame in zip(trackers_list, frames_list):
        tracking_viz.draw_tracks(frame, trackers)
        tracking_viz.draw_trajectories(frame)
        tracking_viz.write_frame(frame)


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

        # TODO: posar aquesta visu abans i despres de filtrar per area, per aparcats, etc.
        trackers = [BoundingBox(*t, int(idx_frame)) for t in trackers]
        frames_list.append(frame)

        trackers_list.append(trackers)
        total_frames += 1

    print("Total Tracking took: %.3f for %d frames, @ %.1f FPS" % (total_time, total_frames, total_frames/total_time))

    if cfg["filter_by_area"]:
        trackers_list = filter_by_area(cfg, trackers_list)
    if cfg["filter_parked"]:
        trackers_list = filter_parked(cfg, trackers_list)

    store_trackers_list(trackers_list, cfg["save_tracking_path"])
    # visualize tracking
    viz_tracking(cfg["save_video_path"], video_width, video_height, fps, trackers_list, frames_list)


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
