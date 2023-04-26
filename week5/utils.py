import cv2
import yaml
import numpy as np
from typing import List, Dict

from bounding_box import BoundingBox

import torch
import kornia.augmentation as K


def load_config(cfg_path):
    with open(cfg_path, 'r') as f:
        config_yaml = yaml.safe_load(f)
    return config_yaml


def group_annotations_by_track(annotations: List[BoundingBox]) -> Dict[int, List[BoundingBox]]:
    """
    Groups the given list of annotations by track.

    Parameters:
    annotations (list): List of annotations to group by track.

    Returns:
    A list of lists of annotations grouped by track.
    """
    group_annotations = dict()
    for annotation in annotations:
        if annotation.track_id not in group_annotations:
            group_annotations[annotation.track_id] = []
        else:
            group_annotations[annotation.track_id].append(annotation)

    return group_annotations


def group_annotations_by_frame(annotations: List[BoundingBox]) -> List[List[BoundingBox]]:
    """
    Groups the given list of annotations by frame.

    Parameters:
    annotations (list): List of annotations to group by frame.

    Returns:
    A list of lists of annotations grouped by frame.
    """
    grouped_annotations = []

    for box in annotations:
        if len(grouped_annotations) <= box.frame:
            for _ in range(box.frame - len(grouped_annotations) + 1):
                grouped_annotations.append([])

        grouped_annotations[box.frame].append(box)

    return grouped_annotations


def load_predictions(csv_file_path: str, grouped: bool = False) -> List[BoundingBox]:
    """
    Loads the predictions from the given CSV file.

    Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    We checked the format in https://github.com/mcv-m6-video/mcv-m6-2021-team4/blob/main/W1/aicity_reader.py
    Also, solved the frame-1 issue :)
    """
    with open(csv_file_path) as f:
        lines = f.readlines()

    bboxes = []

    for line in lines:
        frame, track_id, xtl, ytl, width, height, confidence, _, _, _ = line.split(',')
        xbr = float(xtl) + float(width)
        ybr = float(ytl) + float(height)
        bboxes.append(BoundingBox(
            x1=float(xtl),
            y1=float(ytl),
            x2=xbr,
            y2=ybr,
            frame=int(frame)-1,
            track_id=int(track_id),
            label='car',
            parked=False,
            confidence=float(confidence),
        ))

    if grouped:
        return group_annotations_by_frame(bboxes)

    return bboxes


def filter_annotations(annotations: List[BoundingBox], confidence_thr: float = 0.0) -> List[BoundingBox]:
    return [x for x in annotations if x.confidence >= confidence_thr]


def non_maxima_suppression(bboxes_per_frame: List[List[BoundingBox]], iou_threshold: float = 0.7) -> List[BoundingBox]:
    """
    Perform Non Maxima Suppression (NMS) on a list of bounding boxes.

    :param bboxes: a list of BoundingBox objects per frame.
    :param iou_threshold: the IoU threshold for overlapping bounding boxes.
    :return: a list of selected BoundingBox objects after NMS.
    """
    new_bboxes_per_frame = []

    for bboxes in bboxes_per_frame:
        # Sort the bounding boxes by decreasing confidence scores.
        bboxes_sorted = sorted(bboxes, key=lambda bbox: bbox.confidence or 0, reverse=True)

        selected_bboxes = []

        while bboxes_sorted:
            # Select the bounding box with the highest confidence score.
            bbox = bboxes_sorted[0]
            selected_bboxes.append(bbox)

            # Remove the selected bounding box from the list.
            bboxes_sorted = bboxes_sorted[1:]

            # Compute the IoU between the selected bounding box and the remaining bounding boxes.
            ious = [bbox.IoU(other) for other in bboxes_sorted]

            # Remove the bounding boxes with IoU > threshold.
            bboxes_sorted = [b for i, b in enumerate(bboxes_sorted) if ious[i] <= iou_threshold]

        new_bboxes_per_frame.append(selected_bboxes)

    return new_bboxes_per_frame


def convert_optical_flow_to_image(flow: np.ndarray) -> np.ndarray:
    # The 3-channel uint16 PNG images that comprise optical flow maps contain information
    # on the u-component in the first channel, the v-component in the second channel,
    # and whether a valid ground truth optical flow value exists for a given pixel in the third channel.
    # A value of 1 in the third channel indicates the existence of a valid optical flow value
    # while a value of 0 indicates otherwise. To convert the u- and v-flow values from
    # their original uint16 format to floating point values, one can do so by subtracting 2^15 from the value,
    # converting it to float, and then dividing the result by 64.

    img_u = (flow[:, :, 2] - 2 ** 15) / 64
    img_v = (flow[:, :, 1] - 2 ** 15) / 64

    img_available = flow[:, :, 0]  # whether a valid GT optical flow value is available
    img_available[img_available > 1] = 1

    img_u[img_available == 0] = 0
    img_v[img_available == 0] = 0

    optical_flow = np.dstack((img_u, img_v, img_available))
    return optical_flow


def load_optical_flow(file_path: str):
    # channels arranged as BGR
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED).astype(np.double)
    return convert_optical_flow_to_image(img)


def return_image_full_range(image):
    return (torch.clamp(K.Normalize(mean=[-0.4850, -0.4560, -0.4060], std=[1/0.2290, 1/0.2240, 1/0.2250])(image), min = 0, max = 1) * 255).squeeze().cpu().numpy().astype(np.uint8).transpose(1, 2,  0)


def load_timestamps(timestamps_path: str):
    start_timestamps = {}

    with open(timestamps_path, 'r') as f:
        for line in f:
            camera, timestamp = line.split()
            fps = 10 if camera != 'c015' else 8
            start_timestamps[camera] = float(timestamp) * fps

    return start_timestamps


def draw_bounding_box(image: np.ndarray, bbox: tuple, track_id: str = "unknown", color: tuple = (0, 255, 0)):
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    name_text = f"Track {track_id}"      
    text_size, _ = cv2.getTextSize(name_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    text_width, text_height = text_size
    # Draw filled rectangle behind text for better visibility
    cv2.rectangle(image, (x1, y1 - text_height - 4), (x1 + text_width, y1), (0, 0, 0), cv2.FILLED)
    # Draw text shadow in contrasting color
    cv2.putText(image, name_text, (x1 + 1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)