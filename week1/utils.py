import cv2
import xmltodict
import numpy as np

from typing import List

from class_utils import BoundingBox


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


def load_annotations(xml_file_path: str) -> List[BoundingBox]:
    """
    Loads the annotations from the given XML file.
    """
    with open(xml_file_path) as f:
        annotations = xmltodict.parse(f.read())

    tracks = annotations['annotations']['track']
    bboxes = []

    for track in tracks:
        for box in track['box']:
            bboxes.append(BoundingBox(
                x1=float(box['@xtl']),
                y1=float(box['@ytl']),
                x2=float(box['@xbr']),
                y2=float(box['@ybr']),
                frame=int(box['@frame']),
                track_id=int(track['@id']),
                label=track['@label'],
                parked='attribute' in box and box['attribute']['#text'] == 'true'
            ))

    return bboxes


def load_predictions(csv_file_path: str) -> List[BoundingBox]:
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

    return bboxes


def add_noise_to_bbox(box, noise: float = 0.1) -> BoundingBox:
    """
    Adds normal noise to the size and position of the bounding box.
    
    Parameters:
    coordinates (list): List of coordinates for the bounding box.
    noise (float): Amount of noise to add to the bounding box size and position.
    
    Returns:
    A new list of coordinates with noise added to the bounding box size and position.
    """
    xtl, ytl, xbr, ybr = box.x1, box.y1, box.x2, box.y2
    xtl = xtl + np.random.normal(0, noise)
    ytl = ytl + np.random.normal(0, noise)
    xbr = xbr + np.random.normal(0, noise)
    ybr = ybr + np.random.normal(0, noise)
    return BoundingBox(xtl, ytl, xbr, ybr, box.track_id, box.frame, box.label, box.parked)


def create_fake_track_predictions(
        bboxes: List[BoundingBox], 
        noise: float = 0.1, 
        prob_generate: float = 0.1, 
        prob_delete: float = 0.1
        ):
    """
    Adds noise to the size and position of the bounding boxes in the given list of tracks.
    Introduces probability to generate/delete bounding boxes.
    
    Parameters:
    tracks (list): List of tracks to add noise and probability to.
    noise (float): Amount of noise to add to the bounding box size and position.
    prob_generate (float): Probability of generating a new bounding box for each frame.
    prob_delete (float): Probability of deleting an existing bounding box for each frame.
    
    Returns:
    A new list of tracks with noise and probability added to the bounding boxes.
    """
    new_bboxes = []

    for box in bboxes:
        if np.random.random() < prob_delete:
            new_bboxes.append(box)

        new_box = box.clone()
        new_box = add_noise_to_bbox(box, noise)

        new_bboxes.append(new_box)
        
        if np.random.random() < prob_generate:
            extra_box = new_box.clone()
            extra_box = add_noise_to_bbox(box, noise)
            new_bboxes.append(extra_box)    
    
    # TODO: Add new predictions randomly

    return new_bboxes


def load_optical_flow(file_path: str):
    # channels arranged as BGR
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED).astype(np.double)

    # The 3-channel uint16 PNG images that comprise optical flow maps contain information
    # on the u-component in the first channel, the v-component in the second channel,
    # and whether a valid ground truth optical flow value exists for a given pixel in the third channel.
    # A value of 1 in the third channel indicates the existence of a valid optical flow value
    # while a value of 0 indicates otherwise. To convert the u- and v-flow values from
    # their original uint16 format to floating point values, one can do so by subtracting 2^15 from the value,
    # converting it to float, and then dividing the result by 64.

    img_u = (img[:, :, 2] - 2 ** 15) / 64
    img_v = (img[:, :, 1] - 2 ** 15) / 64

    img_available = img[:, :, 0]  # whether a valid GT optical flow value is available
    img_available[img_available > 1] = 1

    img_u[img_available == 0] = 0
    img_v[img_available == 0] = 0

    optical_flow = np.dstack((img_u, img_v, img_available))

    return optical_flow
