import os
import cv2
import argparse

from tqdm import tqdm

from utils import load_predictions, group_annotations_by_frame


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Road Traffic Monitoring Analysis for Video Surveillance. MCV-M6-Project, week 5, Metric Learning Creation Team 2'
    )

    parser.add_argument('--data_path', type=str, default='../datasets/aic19/train/',
                        help='Path to the dataset')
    parser.add_argument('--output_path', type=str, default='./metric_learning_dataset/',
                        help='Path to the output folder')

    args = parser.parse_args()
    return args


def create_siamese_dataset(args, sequence, camera):
    aic19_video_path = os.path.join(args.data_path, sequence, camera, 'vdo.avi')
    output_dataset_path = args.output_path
    # Path like data_path/SXX/c_00X/gt/gt.txt
    detections_file = os.path.join(args.data_path, sequence, camera, 'gt/gt.txt')
    detections = load_predictions(detections_file)
    detections = group_annotations_by_frame(detections)

    video = cv2.VideoCapture(aic19_video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    for idx_frame in tqdm(range(0, total_frames)):  
        if len(detections) <= idx_frame:
            break

        ret, frame = video.read()

        if not ret:
            break

        frame_name = f"frame{idx_frame:04d}.jpg"

        for detection in detections[idx_frame]:
            xtl, ytl, w, h = detection.coordinates_dim
            xtl, ytl, w, h = int(xtl), int(ytl), int(w), int(h)

            class_path = f"{sequence}_{str(detection.track_id)}"
            os.makedirs(os.path.join(output_dataset_path, class_path), exist_ok=True)

            cropped_frame = frame[ytl:ytl + h, xtl:xtl + w]
            cv2.imwrite(os.path.join(output_dataset_path, class_path, frame_name), cropped_frame)


def main(args: argparse.Namespace):
    for sequence in os.listdir(args.data_path):
        for camera in os.listdir(os.path.join(args.data_path, sequence)):
            create_siamese_dataset(args, sequence, camera)


if __name__ == "__main__":
    args = __parse_args()
    main(args)
