from __future__ import print_function

import os
import argparse
import numpy as np
import time
import cv2
from typing import List

from tqdm import tqdm

from utils_w4 import (
    load_predictions,
    group_annotations_by_frame,
    filter_annotations,
    non_maxima_suppression,
    load_optical_flow,
    read_flow_unimatch,
)
from tracking.tracking_utils import TrackHandlerOverlap, TrackingViz, Track
from class_utils import BoundingBox


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Road Traffic Monitoring Analysis for Video Surveillance. MCV-M6-Project, week 4, task 1.3. Team 2'
    )

    parser.add_argument('--path_sequence', type=str, default="data/AICity_data/train/S03/c010/vdo.avi",
                        help='Path to the directory where the sequence is stored.')

    parser.add_argument('--tracking_type', type=str, default="kalman",
                        help='Type of tracking to be used. Options: "overlap", "kalman".')

    parser.add_argument('--model', type=str, default="faster_finetune",
                        help='Path to the directory where the detections are stored.')

    parser.add_argument('--of_use', action='store_true', default=True,
                        help='Whether to use optical flow.')

    parser.add_argument('--of_type', type=str, default="unimatch",
                        help='Type of optical flow to be used. Options: "block_match", "unimatch".')

    parser.add_argument('--path_of_data', type=str, default="./week4/results/video_of_unimatch/",
                        help='Path to the directory where the optical flow is stored.')

    parser.add_argument('--path_results', type=str, default="./week4/results/",
                        help='The path to the directory where the results will be stored.')

    parser.add_argument('--path_tracking_data', type=str, default="./week4/data/trackers/mot_challenge/parabellum-train",
                        help='The path to the directory where the results will be stored.')
    
    args = parser.parse_args()
    return args


def save_track_results(tracks: List[Track], results_file):
    for t in tracks:
        detections = t.detections
        for d in detections:
            results_file.write(
                f"{d.frame + 1},{d.track_id},{d.x1},{d.y1},{d.x2 - d.x1},{d.y2 - d.y1},{d.confidence if d.confidence else '-1'},-1,-1,-1\n")


def tracking_by_maximum_overlap(
        video_path: str,
        detections: List[BoundingBox],
        model_name: str,
        save_path: str,
        save_tracking_path: str,
        max_frame_skip: int = 0,
        min_iou: float = 0.5,
):
    track_handler = TrackHandlerOverlap(max_frame_skip=max_frame_skip, min_iou=min_iou)
    pass

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video_path = os.path.join(save_path, f"{model_name}.mp4")
    tracking_viz = TrackingViz(output_video_path, width, height, fps=10)

    for frame_id in tqdm(range(total_frames)):
        ret, frame = video.read()
        if not ret:
            break

        frame_detections = detections[frame_id]
        track_handler.update_tracks(frame_detections, frame_id)

        # Visualize tracking
        viz_detections = []
        for track in track_handler.live_tracks:
            detection, _ = track.last_detection()
            viz_detections.append(detection)

        tracking_viz.draw_tracks(frame, viz_detections)
        tracking_viz.draw_trajectories(frame)
        tracking_viz.write_frame(frame)

    # Save tracking results
    save_tracking_path = os.path.join(save_tracking_path, model_name, "data")
    os.makedirs(save_tracking_path, exist_ok=True)
    results_file = open(os.path.join(save_tracking_path, "s03.txt"), "w")
    save_track_results(track_handler.terminated_tracks, results_file)
    save_track_results(track_handler.live_tracks, results_file)

    total_terminated_tracks = len(track_handler.terminated_tracks)
    total_live_tracks = len(track_handler.live_tracks)
    print(f"Total terminated tracks: {total_terminated_tracks}")
    print(f"Total live tracks: {total_live_tracks}")
    print(f"Total tracks: {total_terminated_tracks + total_live_tracks}")


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
        of_use: bool = False,
        of_data_path: str = None,   # Path to the optical flow data. Only used if of_use is True.
        of_type: str = "block_match",  # Type of optical flow. Options: "block_match", "unimatch"
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

    if of_use:
        from tracking.sort_of import Sort
        print("Imported SORT with Optical Flow.")
    else:
        from tracking.sort import Sort
        print("Imported SORT.")

    mot_tracker = Sort(max_age=tracking_max_age, min_hits=tracking_min_hits, iou_threshold=tracking_iou_threshold)

    for idx_frame in tqdm(range(0, max_frames-1, video_frame_sampling), desc="Computing tracking..."):
        # Read detections
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

        # Read optical flow file of the current frame (saved in disk)
        if of_use and of_data_path is not None:
            if of_type == "block_match":
                pred_flow = load_optical_flow(os.path.join(of_data_path, f"{idx_frame}.png"))
            elif of_type == "unimatch":
                # pred_flow = read_flow_unimatch(os.path.join(of_data_path, f"{idx_frame:04d}_pred.flo"))
                # pred_flow = load_optical_flow(os.path.join(of_data_path, f"{idx_frame:04d}_flow.png"))
                pred_flow = load_optical_flow(os.path.join(of_data_path, f"{idx_frame}.png"))
                print(pred_flow.shape)
            else:
                raise ValueError(f"Invalid optical flow type: {of_type}. Options: 'block_matching', 'unimatch'")

            # Update tracker
            trackers = mot_tracker.update(dets, pred_flow)
        else:
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

    print("Total Tracking took: %.3f for %d frames or %.1f FPS" % (total_time, total_frames, total_frames/total_time))


def main(args: argparse.Namespace):

    # Path will be like this: ./week3/data/trackers/mot_challenge/parabellum-train/MODEL_NAME/data/s03.txt
    detections_path = f"week4/results/{args.model}/detections.txt"

    # Load and process detections
    confidence_threshold = 0.5
    detections = load_predictions(detections_path)
    detections = filter_annotations(detections, confidence_thr=confidence_threshold)
    detections = group_annotations_by_frame(detections)
    detections = non_maxima_suppression(detections)

    model_name_for_file = f"{args.tracking_type}_{args.model.replace('_', '-')}_OF_" \
                          f"{args.of_type if args.of_use else 'none'}"

    if args.tracking_type == "kalman":

        # BEST Kalman filter Tracking parameters (found in WK 3)
        max_age = 50
        min_hits = 3
        min_iou = 0.3

        tracking_by_kalman_filter_with_optical_flow(
            detections=detections,
            model_name=model_name_for_file,
            save_video_path=args.path_results,
            save_tracking_path=args.path_tracking_data,
            # video_max_frames=45,    # TODO: Comment this line to use all frames!
            tracking_max_age=max_age,
            tracking_min_hits=min_hits,
            tracking_iou_threshold=min_iou,
            of_use=args.of_use,
            of_data_path=args.path_of_data,
            of_type=args.of_type,
        )

    elif args.tracking_type == "overlap":

        if args.of_use:
            # TODO: Implement this in "tracking > overlap_of.py"
            raise ValueError("Overlap tracking does not support optical flow.")

        else:
            # BEST Overlap Tracking parameters (found in WK 3)
            max_frame_skip = 10
            min_iou = 0.5

            tracking_by_maximum_overlap(
                video_path=args.path_sequence,
                detections=detections,
                model_name=model_name_for_file,
                save_path=args.path_results,
                save_tracking_path=args.path_tracking_data,
                max_frame_skip=max_frame_skip,
                min_iou=min_iou,
            )

    else:
        raise ValueError(f"Unknown tracking type: {args.tracking_type}. Valid values: 'kalman', 'overlap'.")

    print(f"Tracking results saved in: {os.path.join(args.path_tracking_data, model_name_for_file)}")
    print(f"Tracking video saved in: {os.path.join(args.path_results, f'task_1_3_{model_name_for_file}.mp4')}")


if __name__ == "__main__":
    args = __parse_args()
    main(args)
