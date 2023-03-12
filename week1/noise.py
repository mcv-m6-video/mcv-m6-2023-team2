import numpy as np

from typing import List, Optional

from class_utils import BoundingBox


def add_noise_to_bbox(box, std: float = 0.1) -> BoundingBox:
    """
    Adds normal noise to the size and position of the bounding box.
    
    Parameters:
    box (BoundingBox): Bounding box to add noise to.
    std (float): Amount of noise to add to the bounding box size and position.
    
    Returns:
    A new bounding box with noise added to the size and position.
    """
    xtl, ytl, xbr, ybr = box.x1, box.y1, box.x2, box.y2
    xtl = xtl * np.random.normal(1, std)
    ytl = ytl * np.random.normal(1, std)
    xbr = xbr * np.random.normal(1, std)
    ybr = ybr * np.random.normal(1, std)
    return BoundingBox(xtl, ytl, xbr, ybr, box.track_id, box.frame, box.label, box.parked)


def add_position_noise_to_bbox(box, std: float = 0.1) -> BoundingBox:
    """
    Adds normal noise to the position of the bounding box.
    
    Parameters:
    box (BoundingBox): Bounding box to add noise to.
    std (float): Amount of noise to add to the bounding box position.
    
    Returns:
    A new bounding box with noise added to the position.
    """
    x_noise = np.random.normal(0, std)
    y_noise = np.random.normal(0, std)

    move_x = box.width * x_noise
    move_y = box.height * y_noise

    new_box = box.clone()
    new_box.move(move_x, move_y)  

    return new_box


def add_size_noise_to_bbox(box, std: float = 0.1) -> BoundingBox:
    """
    Adds normal noise to the size of the bounding box.
    
    Parameters:
    box (BoundingBox): Bounding box to add noise to.
    std (float): Amount of noise to add to the bounding box size.
    
    Returns:
    A new bounding box with noise added to the size.
    """
    x_noise = np.random.normal(1, std)
    y_noise = np.random.normal(1, std)

    new_width = box.width * x_noise
    new_height = box.height * y_noise

    new_box = box.clone()
    new_box.resize(new_width, new_height)

    return new_box


def create_fake_track_predictions(
        bboxes: List[BoundingBox],
        height: int,
        width: int,
        std_size: float = 0.1,
        std_position: float = 0.1,
        prob_delete: float = 0.3,
        prob_similar: float = 0.1,
        std_similar: float = 0.2,
        min_random: int = 0,
        max_random: int = 2,
        similar_statistic: Optional[str] = None
    ) -> List[BoundingBox]:
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
    num_frames = max([box.frame for box in bboxes])
    new_bboxes = []

    for box in bboxes:
        if np.random.random() < prob_delete:
            continue

        new_box = box.clone()
        new_box = add_size_noise_to_bbox(new_box, std_size)
        new_box = add_position_noise_to_bbox(new_box, std_position)
        new_bboxes.append(new_box)
        
        if np.random.random() < prob_similar:
            extra_box = new_box.clone()
            extra_box = add_noise_to_bbox(box, std_similar)
            new_bboxes.append(extra_box)
    
    if similar_statistic is None:
        for frame in range(num_frames):
            num_random = np.random.randint(min_random, max_random+1)

            for _ in range(num_random):
                xtl = np.random.randint(0, width-1)
                ytl = np.random.randint(0, height-1)
                xbr = np.random.randint(xtl, width)
                ybr = np.random.randint(ytl, height)
                new_box = BoundingBox(xtl, ytl, xbr, ybr, -1, frame)
                new_bboxes.append(new_box)

    elif similar_statistic == "mean":
        mean_width = np.mean([box.width for box in bboxes])
        mean_height = np.mean([box.height for box in bboxes])

        for frame in range(num_frames):
            num_random = np.random.randint(min_random, max_random+1)

            for _ in range(num_random):
                xtl = np.random.randint(0, width-mean_width)
                ytl = np.random.randint(0, height-mean_height)
                xbr = xtl + mean_width
                ybr = ytl + mean_height
                new_box = BoundingBox(xtl, ytl, xbr, ybr, -1, frame)
                new_bboxes.append(new_box)

    return new_bboxes