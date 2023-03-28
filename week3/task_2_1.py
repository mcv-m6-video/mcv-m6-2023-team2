import os
import argparse

import cv2
from tqdm import tqdm

from utils import load_predictions
from utils import group_annotations_by_frame
from utils import filter_annotations
from utils import non_maxima_suppression

from tracking_utils import TrackHandlerOverlap, TrackingViz, Track
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

    parser.add_argument('--path_tracking_data', type=str,
                        default="data/trackers/mot_challenge/parabellum-train",
                        help='The path to the directory where the results will be stored.')

    parser.add_argument('--use_ground_truth', action='store_true', default=False,
                        help='Whether to use the ground truth for evaluation.')

    parser.add_argument('--path_video', type=str, default="../data/AICity_data/train/S03/c010/vdo.avi",
                        help='The path to the video file to be processed.')

    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                        help='Confidence threshold for detections.')

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


def main(args: argparse.Namespace):
    video_path = args.path_video
    save_path = args.path_results
    path_tracking_data = args.path_tracking_data
    os.makedirs(save_path, exist_ok=True)

    # Path will be like this: ./week3/data/gt/mot_challenge/parabellum-train/MODEL_NAME/data/s03.txt
    for model_name in ["yolo", "retina"]:
        detections_path = f"results/{model_name}/detections.txt"

        for min_iou in [0.5, 0.75]:
            for max_frame_skip in [5, 10]:
                for confidence_threshold in [0.0, 0.4]:
                    detections = load_predictions(detections_path)
                    detections = filter_annotations(detections, confidence_threshold)
                    detections = group_annotations_by_frame(detections)
                    detections = non_maxima_suppression(detections)

                    model_name_for_file = f"overlap_{model_name}_thr_{int(confidence_threshold * 100)}_iou_{int(min_iou * 100)}_skip_{max_frame_skip}"
                    tracking_by_maximum_overlap(video_path, detections, model_name_for_file, save_path, path_tracking_data, max_frame_skip, min_iou)
                    print(f"Model: {model_name_for_file} - Done!")


if __name__ == "__main__":
    args = __parse_args()
    main(args)
