import numpy as np

from typing import List

from class_utils import BoundingBox


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