import numpy as np
import argparse
import os
import cv2

from tqdm import tqdm

from utils import load_timestamps, draw_bounding_box
from gps_utils import predictions_to_gps


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Road Traffic Monitoring Analysis for Video Surveillance. MCV-M6-Project, week 5, Metric Learning Creation Team 2'
    )

    parser.add_argument('--sequence_path', type=str, default='./data/aic19/train/S03',
                        help='Path to the sequence')
    parser.add_argument('--detections_path', type=str, default='./data/aic19/train/S03',
                        help='Path to the sequence')
    parser.add_argument('--timestamps_path', type=str, default='./data/aic19/cam_timestamp/S03.txt',
                        help='Path to the timestamps file')

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


def filter_points(predictions_per_camera: dict, threshold_factor: float = 2.0) -> None:    
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

    # Load timestamps
    start_timestamps = load_timestamps(args.timestamps_path)

    print("Mapping predictions to GPS coordinates...")
    predictions_in_gps = predictions_to_gps(cameras, args.sequence_path, args.detections_path)

    print(f"Found {len(predictions_in_gps[cameras[0]])} frames.")

    # Filter points
    # print("Filtering points...")
    # filter_points(predictions_in_gps)

    # Generate 100 random colors
    colors = np.random.randint(0, 255, (100, 3), dtype=np.uint8)

    # Create map 
    subcamera_size = 400
    camera_size = (1080, 1920)
    camera_map = np.zeros((camera_size[0]+subcamera_size, camera_size[1], 3), dtype=np.uint8)
    min_x, min_y, max_x, max_y = np.inf, np.inf, -np.inf, -np.inf
    max_frame = 0
    for camera in cameras:
        max_frame = max(max_frame, len(predictions_in_gps[camera]))
        for frame_predictions in predictions_in_gps[camera]:
            for prediction in frame_predictions:
                min_x = min(min_x, prediction[0])
                min_y = min(min_y, prediction[1])
                max_x = max(max_x, prediction[0])
                max_y = max(max_y, prediction[1])

    # Draw grayish background for all predictions
    camera_colors = np.random.randint(0, 255, (len(cameras), 3), dtype=np.uint8)
    for camera in cameras:
        for frame_predictions in predictions_in_gps[camera]:
            for prediction in frame_predictions:
                x, y = int(np.ceil((prediction[0] - min_x) / (max_x - min_x) * camera_size[1])), \
                          int(np.ceil((prediction[1] - min_y) / (max_y - min_y) * camera_size[0]))  
                color = camera_colors[cameras.index(camera)]
                color = (int(color[0] * 0.5), int(color[1] * 0.5), int(color[2] * 0.5))
                cv2.circle(camera_map, (x, y), 24, color, -1)

    # Apply homography to camera images
    # for camera in cameras:
    #     # Load ROI image
    #     roi = cv2.imread(os.path.join(args.sequence_path, camera, 'roi.jpg'), cv2.IMREAD_GRAYSCALE)
    #     # Load camera image
    #     video = cv2.VideoCapture(os.path.join(args.sequence_path, camera, 'vdo.avi'))
    #     ret, camera_image = video.read()
    #     video.release()

    #     # Apply ROI as mask
    #     camera_image = cv2.bitwise_and(camera_image, camera_image, mask=roi)

    #     # Load homography
    #     homography_file = os.path.join(args.sequence_path, camera, 'calibration.txt')
    #     with open(homography_file, 'r') as f:
    #         homography_line = f.readline()
    #         homography = np.array([val.split() for val in homography_line.split(';')]).astype(np.float32)

    #     homography = LA.inv(homography)
    #     camera_image, (mx, my) = apply_H(camera_image, homography, min_x, min_y, max_x, max_y)
    #     # Add camera image to map by max x and y
    #     camera_map = np.maximum(camera_map, camera_image)

    # Write camera name
    # camera_plotted = []
    # for camera in cameras:
    #     for frame_predictions in predictions_in_gps[camera]:
    #         for prediction in frame_predictions:
    #             if camera not in camera_plotted:
    #                 x, y = int(np.ceil((prediction[0] - min_x) / (max_x - min_x) * camera_size[1])), \
    #                       int(np.ceil((prediction[1] - min_y) / (max_y - min_y) * camera_size[0]))  
    #                 camera_plotted.append(camera)
    #                 cv2.putText(camera_map, camera, (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw predictions in a video
    video = cv2.VideoWriter('map.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, camera_map.shape[:2][::-1])

    # Read camera videos
    camera_videos = {
        camera: cv2.VideoCapture(os.path.join(args.sequence_path, camera, 'vdo.avi')) for camera in cameras
    }

    print("Generating video...")

    num_cameras = len(cameras)
    for idx_frame in tqdm(range(max_frame)):
        map_gps = camera_map.copy()

        if idx_frame > 100:
            break

        # Draw predictions as circles
        for idx_camera, camera in enumerate(cameras):
            if idx_frame >= len(predictions_in_gps[camera]) or start_timestamps[camera] > idx_frame:
                continue

            idx_frame_camera = int(idx_frame - start_timestamps[camera])

            # Draw camera frame in map
            _, camera_image = camera_videos[camera].read()

            for prediction in predictions_in_gps[camera][idx_frame_camera]:
                color = colors[prediction[2] % 100]
                color = (int(color[0]), int(color[1]), int(color[2]))
                # Map GPS coordinates so that they fit in the camera map image
                x, y = int(np.ceil((prediction[0] - min_x) / (max_x - min_x) * camera_size[1])), \
                          int(np.ceil((prediction[1] - min_y) / (max_y - min_y) * camera_size[0])) 
                cv2.circle(map_gps, (x, y), 8, color, -1)
                # Write the track ID with a white background
                cv2.putText(map_gps, f"{camera} - {str(prediction[2])}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(map_gps, f"{camera} - {str(prediction[2])}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                # Draw bounding box
                draw_bounding_box(camera_image, prediction[3], f"{camera} - {str(prediction[2])}", color)
                
            # Space cameras evenly
            camera_image = cv2.resize(camera_image, (int(camera_size[1] // num_cameras), subcamera_size))
            map_gps[map_gps.shape[0] - camera_image.shape[0]:, idx_camera * camera_image.shape[1]:(idx_camera + 1) * camera_image.shape[1]] = camera_image

        video.write(map_gps)

    video.release()


if __name__ == '__main__':
    args = __parse_args()
    main(args)
