from typing import List
import yaml
from bounding_box import BoundingBox


def load_config(cfg_path):
    with open(cfg_path, 'r') as f:
        config_yaml = yaml.safe_load(f)
    return config_yaml


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