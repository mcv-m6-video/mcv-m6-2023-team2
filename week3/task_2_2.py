from __future__ import print_function

import os
import argparse
import numpy as np
import time
import cv2

from tqdm import tqdm

from utils import (
    load_predictions,
    group_annotations_by_frame,
    filter_annotations,
    non_maxima_suppression
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
    
    args = parser.parse_args()
    return args


def tracking_by_kalman_filter(
        detections, 
        model_name, 
        save_video_path, 
        save_tracking_path, 
        max_age: int = 1,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ):
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

    mot_tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold) 

    for idx_frame in tqdm(range(0, total_frames)):  
        if len(detections) <= idx_frame:
            dets = []
        else:
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
    base_confidence_threshold = 0.5
    base_min_iou = 0.3
    base_max_age = 10

    # model_name = "faster_finetune"
    # detections_path = f"week3/results/{model_name}/detections.txt"
    # detections = load_predictions(detections_path)    
    # detections = filter_annotations(detections, confidence_thr=base_confidence_threshold)
    # detections = group_annotations_by_frame(detections)
    # detections = non_maxima_suppression(detections)
    # model_name_for_file = f"kalman_{model_name}_thr_{int(base_confidence_threshold*100)}_nms_{True}_maxage_{base_max_age}"
    # tracking_by_kalman_filter(
    #     detections, 
    #     model_name_for_file, 
    #     args.path_results, 
    #     args.path_tracking_data,
    #     max_age=base_max_age,
    #     min_hits=3,
    #     iou_threshold=0.3,
    #     )

    # Path will be like this: ./week3/data/trackers/mot_challenge/parabellum-train/MODEL_NAME/data/s03.txt
    for model_name in ["faster", "faster_finetune"]:
        detections_path = f"week3/results/{model_name}/detections.txt"

        for max_age in [1, 50]:
            detections = load_predictions(detections_path)
            detections = filter_annotations(detections, confidence_thr=base_confidence_threshold)
            detections = group_annotations_by_frame(detections)
            detections = non_maxima_suppression(detections)
            model_name_for_file = f"kalman_{model_name.replace('_', '-')}_miniou_{int(base_min_iou*100)}_nms_{True}_maxage_{max_age}"
            tracking_by_kalman_filter(
                detections, 
                model_name_for_file, 
                args.path_results, 
                args.path_tracking_data,
                max_age=max_age,
                min_hits=3,
                iou_threshold=base_min_iou,
                )
            
        for min_iou in [0.1, 0.3, 0.5]:
            detections = load_predictions(detections_path)
            detections = filter_annotations(detections, confidence_thr=base_confidence_threshold)
            detections = group_annotations_by_frame(detections)
            detections = non_maxima_suppression(detections)
            model_name_for_file = f"kalman_{model_name.replace('_', '-')}_miniou_{int(min_iou*100)}_nms_{True}_maxage_{base_max_age}"
            tracking_by_kalman_filter(
                detections, 
                model_name_for_file, 
                args.path_results, 
                args.path_tracking_data,
                max_age=base_max_age,
                min_hits=3,
                iou_threshold=min_iou,
                )
        
        


if __name__ == "__main__":
    args = __parse_args()
    main(args)
