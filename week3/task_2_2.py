from __future__ import print_function

import os
import argparse
import numpy as np
import time
import matplotlib.pyplot as plt  
import matplotlib.patches as patches
import cv2

from tqdm import tqdm
from skimage import io
from IPython import display as dp

from utils import (
    load_predictions,
    load_annotations,
    group_annotations_by_frame,
    filter_annotations,
)
from tracking_utils import TrackingViz
from class_utils import BoundingBox
from sort import Sort


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Road Traffic Monitoring Analysis for Video Surveillance. MCV-M6-Project, week 3, task 2.2. Team 2'
    )

    parser.add_argument('--path_sequence', type=str, default="data/AICity_data/train/S03/c010/vdo.avi",
                        help='Path to the directory where the sequence is stored.')

    parser.add_argument('--path_results', type=str, default="./week3/results/",
                    help='The path to the directory where the results will be stored.')

    parser.add_argument('--path_tracking_data', type=str, default="./week3/data/trackers/mot_challenge/parabellum-train",
                    help='The path to the directory where the results will be stored.')
    
    parser.add_argument('--use_ground_truth', action='store_true',
                    help='Whether to use the ground truth for evaluation.')
    
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                    help='Confidence threshold for detections.')
    
    args = parser.parse_args()
    return args


def tracking_by_kalman_filter(detections, model_name, save_video_path, save_tracking_path):
    # Reference: https://github.com/telecombcn-dl/2017-persontyle/blob/master/sessions/tracking/tracking_kalman.ipynb
    save_tracking_path = os.path.join(save_tracking_path, model_name, "data")
    os.makedirs(save_tracking_path, exist_ok=True)
    
    total_time = 0.0
    total_frames = 0
    out = []

    # Only for display
    output_video_path = os.path.join(save_video_path, f"task_2_2_{model_name}.mp4")
    os.makedirs(args.path_results, exist_ok=True)

    video = cv2.VideoCapture(args.path_sequence)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    
    tracking_viz = TrackingViz(output_video_path, video_width, video_height, fps)
    results_file = open(os.path.join(save_tracking_path, "s03.txt"), "w")

    mot_tracker = Sort() 

    for idx_frame in tqdm(range(0, total_frames)):  
        dets = detections[idx_frame]   

        # read the frame
        # video.set(cv2.CAP_PROP_POS_FRAMES, idx_frame)
        ret, frame = video.read()

        if not ret:
            break

        tracking_viz.draw_detections(frame, dets)
                
        # Convert to proper array for the tracker input
        dets = np.array([d.coordinates for d in dets])

        # If no detections, add empty array
        if len(dets) == 0:
            dets = np.empty((0, 5))

        start_time = time.time()
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

    print("Total Tracking took: %.3f for %d frames or %.1f FPS" % (total_time, total_frames, total_frames/total_time))


def main(args: argparse.Namespace):
    for model_name in ["yolo", "ssd", "detr", "retina"]:
        # Path will be like this: ./week3/data/gt/mot_challenge/parabellum-train/MODEL_NAME/data/s03.txt
        detections_path = f"week3/results/{model_name}/detections.txt"
        detections = load_predictions(detections_path)
        detections = filter_annotations(detections, confidence_thr=args.confidence_threshold)
        detections = group_annotations_by_frame(detections)
        model_name = f"{model_name}_thr_{int(args.confidence_threshold*100)}"
        tracking_by_kalman_filter(detections, model_name, args.path_results, args.path_tracking_data)


if __name__ == "__main__":
    args = __parse_args()
    main(args)
