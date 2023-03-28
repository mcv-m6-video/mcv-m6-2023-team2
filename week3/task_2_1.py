import os
import argparse
import numpy as np

import cv2
from tqdm import tqdm

from utils import load_annotations

from tracking_utils import TrackHandlerOverlap, TrackingViz
from typing import List
from class_utils import BoundingBox


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Road Traffic Monitoring Analysis for Video Surveillance. MCV-M6-Project, week 3, task 2.2. Team 2'
    )

    parser.add_argument('--path_annotations', type=str,
                        default="../data/AICity_data/train/S03/c010/ai_challenge_s03_c010-full_annotation.xml",
                        help='Path to the directory where the annotations are stored.')

    parser.add_argument('--path_results', type=str, default="./results/",
                        help='The path to the directory where the results will be stored.')

    parser.add_argument('--use_ground_truth', action='store_true', default=True,
                        help='Whether to use the ground truth for evaluation.')

    parser.add_argument('--path_video', type=str, default="../data/AICity_data/train/S03/c010/vdo.avi",
                        help='The path to the video file to be processed.')

    args = parser.parse_args()
    return args


def tracking_by_maximum_overlap(
        video_path: str,
        annotations: List[BoundingBox],
        save_path: str,
        results_name: str,
        confidence_threshold: float = 0.5,
):
    track_handler = TrackHandlerOverlap(max_frame_skip=0, min_iou=0.4)
    pass

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # TODO: Remove this FAKE part.
    #  Just used to 'invent' random confidences for each GT annotation :)
    for frame_annotation in annotations:
        for annotation in frame_annotation:
            annotation.confidence = np.random.uniform(0, 1)
    # TODO: Until here.

    output_video_path = os.path.join(save_path, f"{results_name}.mp4")
    tracking_viz = TrackingViz(output_video_path, width, height, fps=4)

    for frame_id in tqdm(range(total_frames)):
        ret, frame = video.read()
        if not ret:
            break

        frame_annotations = annotations[frame_id]

        # Filter detections by confidence, if needed
        filtered_frame_annotations = [ann for ann in frame_annotations if ann.confidence > confidence_threshold]

        track_handler.update_tracks(filtered_frame_annotations, frame_id)

        # Visualize tracking
        detections = []
        for track in track_handler.live_tracks:
            detection, _ = track.last_detection()
            detections.append(detection)
        tracking_viz.draw_tracks(frame, detections)

        with open(os.path.join(save_path, f"{results_name}.csv"), 'a') as f:
            # Save tracking with bounding boxes in MOT Challenge format:
            # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, -1, -1, -1, -1
            # TODO: Should we save 'terminated' tracks instead of 'live' tracks?
            #  for track in track_handler.live_tracks:
            for track in track_handler.live_tracks:
                detection, _ = track.last_detection()
                f.write(
                    f"{frame_id},{track.id},{detection.x1},{detection.y1},{detection.x2},{detection.y2},-1,-1,-1,-1\n")
                # print(f"frame: {frame_id}, id: {track.id}, bbox: {detection}")

    print("Total tracks: ", len(track_handler.terminated_tracks))


def main(args: argparse.Namespace):
    video_path = args.path_video
    save_path = args.path_results
    os.makedirs(save_path, exist_ok=True)

    if args.use_ground_truth:
        annotations = load_annotations(args.path_annotations, grouped=True)
    else:
        raise NotImplementedError

    tracking_by_maximum_overlap(video_path, annotations, save_path, "task_2_1_tracking_by_overlap")


if __name__ == "__main__":
    args = __parse_args()
    main(args)
