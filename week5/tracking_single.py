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

from utils import (
    load_config,
    load_predictions,
    group_annotations_by_frame,
    filter_annotations,
    non_maxima_suppression,
)
from tracking.tracking_utils import TrackingViz
from tracking.sort import Sort
from bounding_box import BoundingBox


def tracking_by_kalman_filter(
    cfg,
    detections,
    model_name,
    save_video_path,
    save_tracking_path,
    video_max_frames: int = 9999,
    video_frame_sampling: int = 1,
    tracking_max_age: int = 1,
    tracking_min_hits: int = 3,
    tracking_iou_threshold: float = 0.3,
):
    # Reference: https://github.com/telecombcn-dl/2017-persontyle/blob/master/sessions/tracking/tracking_kalman.ipynb

    os.makedirs(save_tracking_path, exist_ok=True)

    total_time = 0.0
    out = []

    # Only for display
    output_video_path = os.path.join(save_video_path, f"tracking_single_{model_name}.mp4")
    os.makedirs(cfg["path_results"], exist_ok=True)

    video = cv2.VideoCapture(cfg["path_sequence"])
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    max_frames = min(video_max_frames, total_frames)

    tracking_viz = TrackingViz(output_video_path, video_width, video_height, fps)
    results_file = open(os.path.join(save_tracking_path, f"res.txt"), "w")

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
                # If no detections, add empty array

        tracking_viz.draw_detections(frame, dets)

        # Convert to proper array for the tracker input
        dets = np.array([d.coordinates for d in dets])
        # If no detections, add empty array
        if len(dets) == 0:
            dets = np.empty((0, 5))

        start_time = time.time()
        # Update tracker
        trackers = mot_tracker.update(dets)

        cycle_time = time.time() - start_time
        total_time += cycle_time

        trackers = [BoundingBox(*t, int(idx_frame)) for t in trackers]
        tracking_viz.draw_tracks(frame, trackers)
        tracking_viz.draw_trajectories(frame)
        tracking_viz.write_frame(frame)

        # Save tracking with bounding boxes in MOT Challenge format:
        # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, -1, -1, -1, -1
        for d in trackers:
            results_file.write(f"{d.frame+1},{d.track_id},{d.x1},{d.y1},{d.x2-d.x1},{d.y2-d.y1},{d.confidence if d.confidence else '-1'},-1,-1,-1\n")

        out.append(trackers)
        total_frames += 1

    print("Total Tracking took: %.3f for %d frames, @ %.1f FPS" % (total_time, total_frames, total_frames/total_time))


# Detections are stored in the following directory structure:
# | detections
#     | seqXXX
#         | cYYY
#         | cZZZ
#     | seqYYY
#         | cYYY
#     ....
def scan_sequences(cfg):
    # For each tracked video, the output path will be like this:
    # ./week5/data/trackers/mot_challenge/parabellum-train/MODEL_NAME/data/{seq}_{camera}.txt

    # Scan all sequences in directory cfg["detections_dir"]
    for seq in os.listdir(cfg["detections_dir"]):
        seq_name = os.path.basename(seq)
        # Scan all cameras in directory cfg["detections_dir"]/seq
        for camera in os.listdir(os.path.join(cfg["detections_dir"], seq)):
            camera_name = os.path.basename(camera)
            # Load detections

            detections_path = f"{os.path.join(cfg['detections_dir'], seq, camera)}/detections.txt"

            # Load and process detections
            # TODO: afegir filtrat per mida, i el que haviem comentat
            confidence_threshold = 0.6
            detections = load_predictions(detections_path)
            detections = filter_annotations(detections, confidence_thr=confidence_threshold)
            detections = group_annotations_by_frame(detections)
            detections = non_maxima_suppression(detections)

            exp_name = f'{cfg["tracking_type"]}_{seq_name}_{camera_name}'
            save_tracking_path = os.path.join(cfg["path_tracking_data"], exp_name, "data")
            save_video_path = cfg["path_results"]
            cfg["path_sequence"] = os.path.join(cfg["dataset_dir"], seq_name, camera_name, "vdo.avi")

            if cfg["tracking_type"] == "kalman":

                # BEST Kalman filter Tracking parameters (found in WK 3)
                max_age = 50
                min_hits = 3
                min_iou = 0.3

                tracking_by_kalman_filter(
                    cfg=cfg,
                    detections=detections,
                    model_name=exp_name,
                    save_video_path=save_video_path,
                    save_tracking_path=save_tracking_path,
                    # video_max_frames=45,    # TODO: Comment this line to use all frames!
                    tracking_max_age=max_age,
                    tracking_min_hits=min_hits,
                    tracking_iou_threshold=min_iou,
                )
            else:
                raise ValueError(f"Unknown tracking type: {cfg['tracking_type']}. Valid values: 'kalman'.")

            print(f"Tracking results saved in: {save_tracking_path}")
            print(f"Tracking video saved in: {save_video_path}")
            print("-" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/tracking_single.yaml")
    args = parser.parse_args(sys.argv[1:])

    cfg = load_config(args.config)

    scan_sequences(cfg)
