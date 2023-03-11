import xml.etree.ElementTree as ET
import xmltodict
import numpy as np

from copy import deepcopy
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
            grouped_annotations.append([])
            
        grouped_annotations[box.frame].append(box)

    return grouped_annotations


def load_annotations(xml_file_path: str):
    with open(xml_file_path) as f:
        annotations = xmltodict.parse(f.read())

    tracks = annotations['annotations']['track']
    bboxes = []

    for track in tracks:
        for box in track['box']:
            bboxes.append(BoundingBox(
                x1=int(box['@xtl']),
                y1=int(box['@ytl']),
                x2=int(box['@xbr']),
                y2=int(box['@ybr']),
                frame=int(box['@frame']),
                track_id=int(track['@id']),
                label=track['@label'],
                parked=box['attribute']['#text'] == 'true'
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
    xtl = int(xtl + np.random.normal(0, noise))
    ytl = int(ytl + np.random.normal(0, noise))
    xbr = int(xbr + np.random.normal(0, noise))
    ybr = int(ybr + np.random.normal(0, noise))
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