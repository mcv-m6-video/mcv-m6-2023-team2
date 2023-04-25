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


def forward_warping(p: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Forward warp a point with a given Homography H.
    """
    x1, x2, x3 = H @ p.T
    return x1/x3, x2/x3


def backward_warping(p: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Backward warp a point with a given Homography H.
    """
    x1, x2, x3 = LA.inv(H) @ np.array(p)
    return x1/x3, x2/x3
    

def find_max_size(m: int, n: int, H: np.ndarray) -> Tuple[int, int, int, int]:
    corners = np.array([[0, 0, 1], [n, 0, 1], [0, m, 1], [n, m, 1]])
    corners = np.array(forward_warping(corners, H))

    min_x = np.ceil(corners[0].min())
    max_x = np.floor(corners[0].max())
    min_y = np.ceil(corners[1].min())
    max_y = np.floor(corners[1].max())

    return max_x, min_x, max_y, min_y


def apply_H(I: np.ndarray, H: np.ndarray) -> Tuple[np.uint, tuple]:
    """
    Applies a homography to an image.

    Args:
        I (np.array): Image to be transformed.
        H (np.array): Homography matrix. The homography is defined as
            H = [[h11, h12, h13],
                [h21, h22, h23],
                [h31, h32, h33]]

    Returns:
        np.array: Transformed image.
    """
    m, n, C = I.shape
    max_x, min_x, max_y, min_y = find_max_size(m, n, H)

    # Compute size of output image
    width_canvas, height_canvas = max_x - min_x, max_y - min_y

    # Create grid in the output space
    X, Y = np.meshgrid(np.arange(min_x, max_x), np.arange(min_y, max_y))
    X_flat, Y_flat = X.flatten(), Y.flatten()

    # Generate matrix with output points in homogenous coordinates
    dest_points = np.array([X_flat, Y_flat, np.ones_like(X_flat)])

    # Backward warp output points to their source points
    source_x, source_y = backward_warping(dest_points, H)

    # Get src_x and src_y in meshgrid-like coordinates
    source_x = np.reshape(source_x, X.shape)
    source_y = np.reshape(source_y, Y.shape)

    # Set up output image.
    out = np.zeros((int(height_canvas), int(width_canvas), 3))

    # Map source coordinates to their corresponding value.
    # Interpolation is needed as coordinates may be real numbers.
    for i in range(C):
        out[:,:,i] = map_coordinates(I[:,:,i], [source_y, source_x])

    return np.uint8(out), (min_x, min_y)


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

    # Draw predictions in a video
    camera_map = np.zeros((1080, 1920, 3), dtype=np.uint8)
    video = cv2.VideoWriter('map.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, camera_map.shape[::-1], 0)

    # camera_map = cv2.VideoCapture(os.path.join(args.sequence_path, 'vdo.avi')).read()[1]
    # Apply calibration to camera map
    # camera_map = apply_H(camera_map, calibration)[0]

    # Dilate camera map
    # kernel = np.ones((50, 50), np.uint8)
    # camera_map = cv2.dilate(camera_map, kernel, iterations=1)

    print("Generating video...")

    for idx_frame in tqdm(range(max_frame)):
        map_gps = camera_map.copy()

        if idx_frame > 400: # TODO REMOVE!!
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
                x, y = int(np.ceil((prediction[0] - min_x) / (max_x - min_x) * camera_map.shape[1])), \
                          int(np.ceil((prediction[1] - min_y) / (max_y - min_y) * camera_map.shape[0]))                
                cv2.circle(map_gps, (x, y), 32, color, -1)

        # cv2.imwrite(os.path.join(args.sequence_path, f'frame_{idx_frame:04d}.jpg'), map_gps)
        # Resize to 640x480, keeping aspect ratio
        # map_gps = resize_image(map_gps, size=(1920, 1080))
        video.write(map_gps)

    video.release()


if __name__ == '__main__':
    args = __parse_args()
    main(args)
