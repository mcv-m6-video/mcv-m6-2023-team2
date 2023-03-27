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
)
from tracking_utils import TrackingViz
from sort import Sort


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Road Traffic Monitoring Analysis for Video Surveillance. MCV-M6-Project, week 3, task 2.2. Team 2'
    )
    
    parser.add_argument('--path_annotations', type=str, default="data/AICity_data/train/S03/c010/ai_challenge_s03_c010-full_annotation.xml",
                    help='Path to the directory where the annotations are stored.')
    
    parser.add_argument('--path_sequence', type=str, default="data/AICity_data/train/S03/c010/vdo.avi",
                        help='Path to the directory where the sequence is stored.')

    parser.add_argument('--path_results', type=str, default="./results/",
                    help='The path to the directory where the results will be stored.')
    
    parser.add_argument('--use_ground_truth', action='store_true', default=True,
                    help='Whether to use the ground truth for evaluation.')
    
    args = parser.parse_args()
    return args


def main(args: argparse.Namespace):
    if args.use_ground_truth:
        detections = load_annotations(args.path_annotations) 
        detections = group_annotations_by_frame(detections)
    else:
        raise NotImplementedError
    
    # Reference: https://github.com/telecombcn-dl/2017-persontyle/blob/master/sessions/tracking/tracking_kalman.ipynb
    
    display = True
    total_time = 0.0
    total_frames = 0
    out = []

    # Only for display
    output_video_path = os.path.join(args.path_results, "task_2_2.mp4")
    os.makedirs(args.path_results, exist_ok=True)

    video = cv2.VideoCapture(args.path_sequence)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    
    tracking_viz = TrackingViz(output_video_path, video_width, video_height, fps)

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
        
        tracking_viz.draw_tracks(frame, trackers)  
        tracking_viz.write_frame(frame)      
        
        out.append(trackers)
        total_frames += 1

    print("Total Tracking took: %.3f for %d frames or %.1f FPS" % (total_time, total_frames, total_frames/total_time))


if __name__ == "__main__":
    args = __parse_args()
    main(args)
