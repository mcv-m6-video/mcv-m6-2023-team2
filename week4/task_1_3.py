from __future__ import print_function

import sys
import os
import argparse
import numpy as np
import time
import cv2

from tqdm import tqdm

from utils_w4 import (
    load_predictions,
    group_annotations_by_frame,
    filter_annotations,
    non_maxima_suppression
)
from tracking.tracking_utils import TrackingViz
from class_utils import BoundingBox
# from tracking.sort import Sort
from tracking.sort_of import Sort

from of.optical_flow import BlockMatching


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Road Traffic Monitoring Analysis for Video Surveillance. MCV-M6-Project, week 3, task 2.2. Team 2'
    )

    parser.add_argument('--path_sequence', type=str, default="data/AICity_data/train/S03/c010/vdo.avi",
                        help='Path to the directory where the sequence is stored.')

    parser.add_argument('--path_of_data', type=str, default="./week4/results/video_of/",
                        help='Path to the directory where the sequence is stored.')

    parser.add_argument('--path_results', type=str, default="./week4/results/",
                    help='The path to the directory where the results will be stored.')

    parser.add_argument('--path_tracking_data', type=str, default="./week4/data/trackers/mot_challenge/parabellum-train",
                    help='The path to the directory where the results will be stored.')
    
    args = parser.parse_args()
    return args


def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
            # Reshape testdata into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))


def tracking_by_kalman_filter_with_optical_flow(
        detections, 
        model_name, 
        save_video_path, 
        save_tracking_path,
        video_max_frames: int = 9999,
        video_frame_sampling: int = 1,
        tracking_max_age: int = 1,
        tracking_min_hits: int = 3,
        tracking_iou_threshold: float = 0.3,
        of_data_path: str = None,
        of_block_size: int = 16,
        of_search_window_size: int = 64,
        of_estimation_type: str = "mean",
        of_error_function: str = "mse",
        of_color_space: str = "gray",
):
    # Reference: https://github.com/telecombcn-dl/2017-persontyle/blob/master/sessions/tracking/tracking_kalman.ipynb
    save_tracking_path = os.path.join(save_tracking_path, model_name, "data")
    os.makedirs(save_tracking_path, exist_ok=True)
    
    total_time = 0.0
    out = []

    # Only for display
    output_video_path = os.path.join(save_video_path, f"task_1_3_{model_name}.mp4")
    os.makedirs(args.path_results, exist_ok=True)

    video = cv2.VideoCapture(args.path_sequence)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    max_frames = min(video_max_frames, total_frames)
    
    tracking_viz = TrackingViz(output_video_path, video_width, video_height, fps)
    results_file = open(os.path.join(save_tracking_path, "s03.txt"), "w")

    mot_tracker = Sort(max_age=tracking_max_age, min_hits=tracking_min_hits, iou_threshold=tracking_iou_threshold)

    # block_matching = BlockMatching(block_size=of_block_size, search_window_size=of_search_window_size,
    #                                estimation_type=of_estimation_type, error_function=of_error_function)

    for idx_frame in tqdm(range(0, max_frames, video_frame_sampling), desc="Computing Kalman filter Tracking with "
                                                                             "Optical Flow..."):
        # Read detections
        if len(detections) <= idx_frame:
            dets = []
        else:
            dets = detections[idx_frame]   

        # read the frame
        video.set(cv2.CAP_PROP_POS_FRAMES, idx_frame)
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

        # if of_color_space == "gray":
        #     pred_flow = block_matching.estimate_optical_flow(
        #         cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY),
        #         cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
        #         leave_tqdm=False
        #     )
        # else:
        #     pred_flow = block_matching.estimate_optical_flow(
        #         frame_prev,
        #         frame,
        #         leave_tqdm=False
        #     )

        # Read optical flow file of the current frame (saved in disk)
        if of_data_path is not None:
            pred_flow = readFlow(os.path.join(of_data_path, f"{idx_frame:04d}_pred.flo"))
        else:
            pred_flow = np.zeros((video_width, video_height, 2))

        trackers = mot_tracker.update(dets, pred_flow)
        # trackers = mot_tracker.update(dets)
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

    # Path will be like this: ./week3/data/trackers/mot_challenge/parabellum-train/MODEL_NAME/data/s03.txt
    model_name = "retina_of"
    detections_path = f"week4/results/{model_name}/detections.txt"
    confidence_threshold = 0.5
    # Tracking parameters
    max_age = 50
    min_hits = 3
    min_iou = 0.3
    # Optical Flow parameters
    block_size = 24
    search_window_size = 76
    estimation_type = "forward"
    error_function = "nccorr"
    color_space = "gray"

    detections = load_predictions(detections_path)
    detections = filter_annotations(detections, confidence_thr=confidence_threshold)
    detections = group_annotations_by_frame(detections)
    detections = non_maxima_suppression(detections)
    # model_name_for_file = f"kalman_{model_name.replace('_', '-')}_block_{block_size}_window" \
    #                       f"_{search_window_size}_type_{estimation_type}_error_{error_function}_color_{color_space}"

    model_name_for_file = f"kalman_{model_name.replace('_', '-')}_unimatch" \

    # FIXME: Not optimal (at least with BlockMatching). ETA: +23 hours
    tracking_by_kalman_filter_with_optical_flow(
        detections=detections,
        model_name=model_name_for_file,
        save_video_path=args.path_results,
        save_tracking_path=args.path_tracking_data,
        # video_max_frames=45,
        tracking_max_age=max_age,
        tracking_min_hits=min_hits,
        tracking_iou_threshold=min_iou,
        of_data_path=args.path_of_data,
        of_block_size=block_size,
        of_search_window_size=search_window_size,
        of_estimation_type=estimation_type,
        of_error_function=error_function,
        of_color_space=color_space,
    )


if __name__ == "__main__":
    args = __parse_args()
    main(args)
