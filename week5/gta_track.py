import numpy as np
import argparse
import os
import cv2

from tqdm import tqdm

from utils import load_timestamps
from homographies import apply_H
from gps_utils import predictions_to_gps
import random
import uuid
import copy

def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Road Traffic Monitoring Analysis for Video Surveillance. MCV-M6-Project, week 5, Metric Learning Creation Team 2'
    )

    parser.add_argument('--sequence_path', type=str, default='/home/adri/Desktop/master/M6/mcv-m6-2023-team2/week5/aic19/train/S03',
                        help='Path to the sequence')
    parser.add_argument('--detections_path', type=str, default='/home/adri/Desktop/master/M6/mcv-m6-2023-team2/week5/aic19/train/S03',
                        help='Path to the sequence')
    parser.add_argument('--timestamps_path', type=str, default='/home/adri/Desktop/master/M6/mcv-m6-2023-team2/week5/aic19/cam_timestamp/S03.txt',
                        help='Path to the timestamps file')
    parser.add_argument('-v', '--vel_window', default=5)

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

    # Create map 
    camera_map = np.zeros((1080, 1920, 3), dtype=np.uint8)
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
                x, y = int(np.ceil((prediction[0] - min_x) / (max_x - min_x) * camera_map.shape[1])), \
                          int(np.ceil((prediction[1] - min_y) / (max_y - min_y) * camera_map.shape[0]))  
                color = camera_colors[cameras.index(camera)]
                color = (int(color[0] * 0.5), int(color[1] * 0.5), int(color[2] * 0.5))
                cv2.circle(camera_map, (x, y), 24, color, -1)

    cars = {}
    for camera in cameras:
        cars[camera] = {}
        print(len(predictions_in_gps[camera]))
        for frame, frame_predictions in enumerate(predictions_in_gps[camera]):
            for pred, prediction in enumerate(frame_predictions):

                x, y = int(np.ceil((prediction[0] - min_x) / (max_x - min_x) * camera_map.shape[1])), \
                        int(np.ceil((prediction[1] - min_y) / (max_y - min_y) * camera_map.shape[0]))
                
                id_ =  prediction[2] if prediction[2] != -1 else uuid.uuid4()
                if not id_ in cars[camera]: cars[camera][id_] = {}
                cars[camera][id_][frame] = np.array([x, y])
    
    window = args.vel_window
    cars_features = copy.deepcopy(cars)
    if window:
        for camera in cameras:
            for car in cars[camera]:
                frames = list(cars[camera][car].keys())
                for idx, frame in  enumerate(cars[camera][car]):
                    

                    idx_org = max(0, idx - window)
                    if idx_org == idx: moduluus, angle = 0, 0 # Mirar que fem aquí
                    else:
                        vel_vector = cars[camera][car][frames[idx]] - cars[camera][car][frames[idx_org]]
                        moduluus = np.sqrt(vel_vector.dot(vel_vector))
                        angle = vel_vector[0] / vel_vector[1]
                    motion = np.array([moduluus, angle])
                    cars_features[camera][car][frame] = np.concatenate([cars_features[camera][car][frame], motion])

                    cars_features[camera][car][frame][0] /= camera_map.shape[1]
                    cars_features[camera][car][frame][1] /= camera_map.shape[0]

    # {c010: cotxe1: {1: V, 2: V, 3: V}}
    correspondences = {}
    a = []
    THR = 0.975
    dot, norm = np.dot, np.linalg.norm # nah nah nah truquitos que poca gente sabe
    done = []
    for cam in cameras:
        for cam_2 in cameras:
            if cam == cam_2 or (f"{cam_2}-{cam}" in done): continue
            done.append(f"{cam}-{cam_2}" )
            for frame in range(max_frame):
                #if not frame in correspondences: correspondences[frame] = {}


                ## ARA VULL TOTS ELS COTXES D'AQUELL FRAME A CADA CAMERA ###

                cam1_candidates = [car for car in cars_features[cam] if frame in cars_features[cam][car]]
                cam2_candidates = [car for car in cars_features[cam_2] if frame in cars_features[cam_2][car]]

                # [car1, car2, car3]
                # [car4, car5]
                if not (len(cam1_candidates) * len(cam2_candidates)): continue

                feats = np.array([cars_features[cam][car][frame] for car in cam1_candidates])
                feats2 = np.array([cars_features[cam_2][car][frame] for car in cam2_candidates])

                coormatrix = np.zeros((len(feats), len(feats2)))
                for n, f in enumerate(feats):
                    for m, f2 in enumerate(feats2):

                        coormatrix[n, m] = dot(f, f2) / (norm(f) * norm(f2) + 1e-3)
                
                assigned = {}
                for idx, car_id in enumerate(cam1_candidates):
                    idx_max = np.argmax(coormatrix[idx])
                    if coormatrix[idx, idx_max] >= THR and (not idx in assigned):

                        correspondences[car_id] = cam2_candidates[idx_max]

    print(correspondences)

if __name__ == '__main__':
    args = __parse_args()
    main(args)
