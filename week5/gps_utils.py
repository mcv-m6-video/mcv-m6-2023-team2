import numpy as np
import argparse
import os
import numpy.linalg as LA

from typing import Dict, List

from utils import load_predictions, group_annotations_by_frame


def predictions_to_gps(cameras: List[str], sequence_path: str) -> Dict[str, List[List[tuple]]]:
    """
    Convert predictions from image coordinates to GPS coordinates.
    
    Args:
        cameras: List of cameras in the sequence.
        sequence_path: Path to the sequence.

    Returns:
        predictions_in_gps: Dictionary with the predictions in GPS coordinates for each camera.
        {
            'camera_00': [
                [(x, y, track_id), (x, y, track_id), ...],  # frame 0
                [(x, y, track_id), (x, y, track_id), ...],  # frame 1
                ...
            ],
        }
    """
    predictions_in_gps = {}

    for camera in cameras:
        predictions_in_gps[camera] = []
        homography_file = os.path.join(sequence_path, camera, 'calibration.txt')
        # Calibration file format
        # First line (homography): x y z;x y z;x y z;x y z
        # Second line (distortion coefficients, optional): k1 k2 p1 p2
        with open(homography_file, 'r') as f:
            homography_line = f.readline()
            homography = np.array([val.split() for val in homography_line.split(';')]).astype(np.float32)
            # distorion_coeffs = f.readline()
            # if distorion_coeffs:
            #     continue

        # Invert homography                
        homography = LA.inv(homography)

        # Load predictions
        predictions = load_predictions(os.path.join(sequence_path, camera, 'gt/gt.txt'))
        predictions = group_annotations_by_frame(predictions)

        for idx_frame, frame_predictions in enumerate(predictions):
            predictions_in_gps[camera].append([])

            for prediction in frame_predictions:
                # Convert to GPS
                gps = homography @ np.array([prediction.center_x, prediction.center_y, 1]).T
                gps = gps / gps[2]
                predictions_in_gps[camera][idx_frame].append((gps[0], gps[1], prediction.track_id))

    return predictions_in_gps