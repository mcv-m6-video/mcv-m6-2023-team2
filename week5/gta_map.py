import numpy as np
import argparse
import os
import cv2
import numpy.linalg as LA

from tqdm import tqdm
from typing import Tuple
from scipy.ndimage import map_coordinates

from utils import load_predictions, group_annotations_by_frame


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Road Traffic Monitoring Analysis for Video Surveillance. MCV-M6-Project, week 5, Metric Learning Creation Team 2'
    )

    parser.add_argument('--sequence_path', type=str, default='./data/aic19/train/S03',
                        help='Path to the sequence')

    args = parser.parse_args()
    return args


def resize_image(img, size=(28,28)):
    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape)>2 else 1

    if h == w: 
        return cv2.resize(img, size, cv2.INTER_AREA)

    dif = h if h > w else w

    interpolation = cv2.INTER_AREA if dif > (size[0]+size[1])//2 else cv2.INTER_CUBIC

    x_pos = (dif - w)//2
    y_pos = (dif - h)//2

    if len(img.shape) == 2:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]

    return cv2.resize(mask, size, interpolation)


def filter_points(predictions_per_camera: dict, threshold_factor: float = 3) -> None:    
    all_points = []
    for camera_name, camera_predictions in predictions_per_camera.items():
        camera_points = [(x, y) for preds in camera_predictions for (x, y, _) in preds]
        all_points.extend(camera_points)
    
    # Calculate the centroid of all points
    centroid = np.mean(all_points, axis=0)
    
    # Calculate the distance between each point and the centroid
    distances = [np.linalg.norm(np.array(point) - centroid) for point in all_points]
    
    # Calculate the median distance
    median_distance = np.median(distances)
    
    # Remove all points that are further away than the threshold
    for camera_name, camera_predictions in predictions_per_camera.items():
        for preds in camera_predictions:
            preds[:] = [pred for pred in preds if np.linalg.norm(np.array((pred[0], pred[1])) - centroid) < threshold_factor * median_distance]


def main(args):
    cameras = os.listdir(args.sequence_path)
    cameras = [camera for camera in cameras if camera.startswith('c')]
    cameras.sort()
    print('Cameras:', cameras)

    print("Mapping predictions to GPS coordinates...")
    predictions_in_gps = {}
    for camera in cameras:
        predictions_in_gps[camera] = []
        calibration_file = os.path.join(args.sequence_path, camera, 'calibration.txt')
        # Calibration file format: x y z;x y z;x y z;x y z
        with open(calibration_file, 'r') as f:
            line = f.readline()
            calibration = np.array([val.split() for val in line.split(';')]).astype(np.float32)

        predictions = load_predictions(os.path.join(args.sequence_path, camera, 'gt/gt.txt'))
        predictions = group_annotations_by_frame(predictions)

        for idx_frame, frame_predictions in enumerate(predictions):
            predictions_in_gps[camera].append([])

            for prediction in frame_predictions:
                # Convert to GPS
                gps = calibration @ np.array([prediction.center_x, prediction.center_y, 1]).T
                gps = gps / gps[2]
                predictions_in_gps[camera][idx_frame].append((gps[0], gps[1], prediction.track_id))

    print(f"Found {len(predictions_in_gps[cameras[0]])} frames.")

    # Filter points
    print("Filtering points...")
    filter_points(predictions_in_gps)

    # Generate 100 random colors
    colors = np.random.randint(0, 255, (100, 3), dtype=np.uint8)

    # Create map 

    min_x, min_y, max_x, max_y = 0, 0, 0, 0
    max_frame = 0
    for camera in cameras:
        max_frame = max(max_frame, len(predictions_in_gps[camera]))
        for frame_predictions in predictions_in_gps[camera]:
            for prediction in frame_predictions:
                min_x = min(min_x, prediction[0])
                min_y = min(min_y, prediction[1])
                max_x = max(max_x, prediction[0])
                max_y = max(max_y, prediction[1])

    map_size = (int(np.ceil(max_y - min_y)), int(np.ceil(max_x - min_x)), 3)
    print(f"Map size: {map_size}")

    camera_map = np.zeros((map_size[0], map_size[1], 3), dtype=np.uint8)

    # Draw grayish background for all predictions
    camera_colors = np.random.randint(0, 255, (len(cameras), 3), dtype=np.uint8)
    for camera in cameras:
        for frame_predictions in predictions_in_gps[camera]:
            for prediction in frame_predictions:
                # x, y = int(np.ceil((prediction[0] - min_x) / (max_x - min_x) * camera_map.shape[1])), \
                #           int(np.ceil((prediction[1] - min_y) / (max_y - min_y) * camera_map.shape[0]))  
                x, y = int(np.ceil((prediction[0] - min_x))), int(np.ceil((prediction[1] - min_y)))
                color = camera_colors[cameras.index(camera)]
                color = (int(color[0] * 0.5), int(color[1] * 0.5), int(color[2] * 0.5))
                cv2.circle(camera_map, (x, y), 24, color, -1)
                # Write camera name
                # cv2.putText(camera_map, camera, (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # camera_map = cv2.VideoCapture(os.path.join(args.sequence_path, camera, 'vdo.avi')).read()[1]
    # Apply calibration to camera map
    # camera_map = apply_H(camera_map, calibration)[0]

    # Dilate camera map
    # kernel = np.ones((50, 50), np.uint8)
    # camera_map = cv2.dilate(camera_map, kernel, iterations=1)

    # Resize camera map to fit in the output video
    # camera_map = cv2.resize(camera_map, (1920, 1080))

    # Draw predictions in a video
    # video = cv2.VideoWriter('map.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, camera_map.shape[:2][::-1])
    video = cv2.VideoWriter('map.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, (1920, 1080))

    print("Generating video...")

    for idx_frame in tqdm(range(max_frame)):
        map_gps = camera_map.copy()

        if idx_frame > 100:
            break

        # Draw predictions as circles
        for camera in cameras:
            if idx_frame >= len(predictions_in_gps[camera]):
                continue

            for prediction in predictions_in_gps[camera][idx_frame]:
                color = colors[prediction[2] % 100]
                color = (int(color[0]), int(color[1]), int(color[2]))
                # y, x = int(np.ceil(prediction[1] - min_y)), int(np.ceil(prediction[0] - min_x))
                # Map GPS coordinates so that they fit in the camera map image
                # x, y = int(np.ceil((prediction[0] - min_x) / (max_x - min_x) * camera_map.shape[1])), \
                #           int(np.ceil((prediction[1] - min_y) / (max_y - min_y) * camera_map.shape[0])) 
                x, y = int(np.ceil((prediction[0] - min_x))), int(np.ceil((prediction[1] - min_y)))
                cv2.circle(map_gps, (x, y), 8, color, -1)
                # Write the track ID with a white background
                cv2.putText(map_gps, str(prediction[2]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(map_gps, str(prediction[2]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # cv2.imwrite(os.path.join(args.sequence_path, f'frame_{idx_frame:04d}.jpg'), map_gps)
        # Resize to 640x480, keeping aspect ratio
        map_gps = resize_image(map_gps, size=(1920, 1080))
        video.write(map_gps)

    video.release()


if __name__ == '__main__':
    args = __parse_args()
    main(args)
