import cv2
import numpy as np
from typing import List, Dict
import time
import os
from tqdm import tqdm

from bounding_box import BoundingBox
from week5.utils import load_optical_flow


class Track(object):
    """
    Class to represent a track (object or instance trajectory).
    """
    def __init__(self, id: int, first_detection: BoundingBox, first_frame_number: int):
        first_detection.track_id = id
        self.id = id
        self.detections = [first_detection]
        self.frame_numbers = [first_frame_number]
        self.terminated = False

    def get_track_boxes(self):
        return self.detections

    def add_detection(self, detection: BoundingBox, frame_number: int):
        detection.track_id = self.id
        if self.frame_numbers[-1] == frame_number:
            return
        self.detections.append(detection)
        self.frame_numbers.append(frame_number)

    def last_detection(self):
        return self.detections[-1], self.frame_numbers[-1]


class TrackHandler:
    def __init__(self):
        self.track_id_counter = 0

    def update_tracks(self, frame_detections: List[BoundingBox], frame_number: int):
        pass

    def create_new_track(self, first_detection: BoundingBox, first_frame_number: int):
        pass


class TrackHandlerOverlap(TrackHandler):
    def __init__(self, max_frame_skip: int = 0, min_iou: float = 0.4):
        """
        :param max_frame_skip: Maximum number of frames to skip before terminating a track
        :param min_iou: Minimum IoU to consider a detection as a match
        """
        super().__init__()
        self.max_frame_skip = max_frame_skip
        self.min_iou = min_iou
        self.live_tracks = []
        self.terminated_tracks = []

    def create_new_track(self, first_detection: BoundingBox, first_frame_number: int):
        new_track = Track(self.track_id_counter, first_detection, first_frame_number)
        self.track_id_counter += 1
        return new_track

    def update_tracks(self, frame_detections: List[BoundingBox], frame_number: int):
        new_live_tracks = []

        # Update live / terminated tracks
        for track in self.live_tracks:
            _, last_frame_number = track.last_detection()
            if abs(frame_number - last_frame_number) <= self.max_frame_skip + 1:
                new_live_tracks.append(track)
            else:
                self.terminated_tracks.append(track)

        self.live_tracks = new_live_tracks
        new_tracks = []
        # Update tracks
        for detection in frame_detections:
            max_iou, best_idx = 0, -1
            for idx in range(len(self.live_tracks)):
                last_track_detection, last_frame_number = self.live_tracks[idx].last_detection()
                iou = detection.IoU(last_track_detection)
                if iou > self.min_iou and iou > max_iou:
                    max_iou = iou
                    best_idx = idx

            # If match : update track detection
            if best_idx != -1:
                self.live_tracks[best_idx].add_detection(detection, frame_number)
            else:  # Otherwise create new track
                new_track = self.create_new_track(detection, frame_number)
                new_tracks.append(new_track)

        # Add new tracks to live tracks
        for new_track in new_tracks:
            self.live_tracks.append(new_track)


def post_tracking(cfg, video_width, video_height, fps, trackers_list, frames_list):
    if cfg["filter_by_area"]:
        trackers_list = filter_by_area(cfg, trackers_list)
    if cfg["filter_parked"]:
        trackers_list = filter_parked(cfg, trackers_list)

    store_trackers_list(trackers_list, cfg["save_tracking_path"])
    # visualize tracking
    viz_tracking(cfg["save_video_path"], video_width, video_height, fps, trackers_list, frames_list)


def tracking_by_maximum_overlap(
    cfg: Dict,
    detections: List[BoundingBox],
    max_frame_skip: int = 0,
    min_iou: float = 0.5,
):
    track_handler = TrackHandlerOverlap(max_frame_skip=max_frame_skip, min_iou=min_iou)
    video = cv2.VideoCapture(cfg["path_sequence"])
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    total_time = 0.0
    trackers_list = []
    frames_list = []

    for frame_id in tqdm(range(total_frames-1)):
        ret, frame = video.read()
        if not ret:
            break

        # Read detections
        if len(detections) <= frame_id:
            frame_detections = []
        else:
            frame_detections = detections[frame_id]

        start_time = time.time()
        track_handler.update_tracks(frame_detections, frame_id)
        cycle_time = time.time() - start_time
        total_time += cycle_time

        # Visualize tracking
        frame_detections = []
        for track in track_handler.live_tracks:
            detection, _ = track.last_detection()
            frame_detections.append(detection)
        trackers_list.append(frame_detections)
        frames_list.append(frame)

        total_frames += 1

    print("Total Tracking took: %.3f for %d frames, @ %.1f FPS" % (total_time, total_frames, total_frames/total_time))
    post_tracking(cfg, video_width, video_height, fps, trackers_list, frames_list)


def tracking_by_kalman_filter(
    cfg,
    detections,
    video_max_frames: int = 9999,
    video_frame_sampling: int = 1,
    tracking_max_age: int = 1,
    tracking_min_hits: int = 3,
    tracking_iou_threshold: float = 0.3,
    of_use: bool = False,
    of_data_path: str = None,
):
    # Reference: https://github.com/telecombcn-dl/2017-persontyle/blob/master/sessions/tracking/tracking_kalman.ipynb

    total_time = 0.0
    trackers_list = []
    frames_list = []

    # Only for display

    video = cv2.VideoCapture(cfg["path_sequence"])
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    max_frames = min(video_max_frames, total_frames)

    if of_use:
        from sort_of import Sort
        print("Imported SORT with Optical Flow.")
    else:
        from sort import Sort
        print("Imported SORT.")
    mot_tracker = Sort(max_age=tracking_max_age, min_hits=tracking_min_hits, iou_threshold=tracking_iou_threshold)

    for idx_frame in tqdm(range(0, max_frames-1, video_frame_sampling), desc="Computing tracking..."):
        # read the frame
        ret, frame = video.read()

        if not ret:
            break

        # Read detections
        if len(detections) <= idx_frame:
            dets = []
        else:
            dets = detections[idx_frame]

        # Convert to proper array for the tracker input
        dets = np.array([d.coordinates for d in dets])
        # If no detections, add empty array
        if len(dets) == 0:
            dets = np.empty((0, 5))

        start_time = time.time()
        # Update tracker
        if of_use and of_data_path is not None:
            pred_flow = load_optical_flow(os.path.join(of_data_path, f"{idx_frame}.png"))

            # Update tracker
            trackers = mot_tracker.update(dets, pred_flow)
        else:
            # Update tracker
            trackers = mot_tracker.update(dets)
        cycle_time = time.time() - start_time
        total_time += cycle_time

        trackers = [BoundingBox(*t, int(idx_frame)) for t in trackers]
        trackers_list.append(trackers)
        frames_list.append(frame)

        total_frames += 1

    print("Total Tracking took: %.3f for %d frames, @ %.1f FPS" % (total_time, total_frames, total_frames/total_time))
    post_tracking(cfg, video_width, video_height, fps, trackers_list, frames_list)


def viz_tracking(
    output_video_path: str,
    video_width: int,
    video_height: int,
    fps: int,
    trackers_list: List[List[BoundingBox]],
    frames_list: List[np.ndarray],
):
    tracking_viz = TrackingViz(output_video_path, video_width, video_height, fps)
    for trackers, frame in zip(trackers_list, frames_list):
        tracking_viz.draw_tracks(frame, trackers)
        tracking_viz.draw_trajectories(frame)
        tracking_viz.write_frame(frame)


class TrackingViz:
    TRACKER_NAMES = ['John Wick', 'Viggo Tarasov', 'Iosef Tarasov', 'Marcus', 'Winston', 'Ms. Perkins', 'Aurelio', 'Charon', 'Harry', 'Jimmy', 'Francisco', 'Kirill', 'Abram Tarasov', 'Wolfgang', 'Perkins', 'Avi', 'Wick\'s neighbor', 'Priest', 'Helen Wick', 'Julius', 'Mrs. Tarasov', 'Baba Yaga', 'Viggo\'s lawyer', 'Mrs. Perkins', 'Doctor', 'Continental hotel clerk', 'Continental hotel doctor', 'Continental hotel sommelier', 'Continental hotel bartender', 'Continental hotel concierge', 'Continental hotel housekeeper', 'Continental hotel doorman', 'Continental hotel security']

    def __init__(self, 
                 output_video_path: str, 
                 video_width: int, 
                 video_height: int, 
                 fps: int, 
                 fourcc: str = cv2.VideoWriter_fourcc(*'XVID'),
                 max_trackers: int = 32,
                 max_trajectory_length: int = 100,
                 ) -> None:
        self.out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (video_width, video_height))
        self.video_width = video_width
        self.video_height = video_height
        self.colors = np.random.rand(max_trackers, 3) * 255
        self.max_trackers = max_trackers
        self.max_trajectory_length = max_trajectory_length
        self.frame_trajectories = []

    def draw_detections(self, frame: np.ndarray, detections: List[BoundingBox]) -> None:
        """
        Draw the detection boxes on the frame.

        Args:
            frame: image frame to draw on
            detections: list of bounding boxes in format BoundingBox
        """
        for detection in detections:
            detection = np.array(detection.coordinates)
            detection[0] = np.clip(detection[0], 0, self.video_width)
            detection[1] = np.clip(detection[1], 0, self.video_height)
            detection[2] = np.clip(detection[2], 0, self.video_width)
            detection[3] = np.clip(detection[3], 0, self.video_height)
            detection = detection.astype(np.uint32)

            x1, y1, x2, y2 = detection
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)

    def draw_tracks(self, frame: np.ndarray, trackers: List[BoundingBox]) -> None:
        """
        Draw the tracking boxes on the frame.

        Args:
            frame: image frame to draw on
            trackers: list of bounding boxes in format [x1, y1, x2, y2, track_id]
        """
        self.frame_trajectories.append([])

        for tracker in trackers:
            track_id = tracker.track_id
            if isinstance(track_id, str):
                track_id = int(track_id)
            track_idx = track_id % self.max_trackers

            self.frame_trajectories[-1].append((tracker.center_x, tracker.center_y, track_id))

            tracker = np.array(tracker.coordinates)
            tracker[0] = np.clip(tracker[0], 0, self.video_width)
            tracker[1] = np.clip(tracker[1], 0, self.video_height)
            tracker[2] = np.clip(tracker[2], 0, self.video_width)
            tracker[3] = np.clip(tracker[3], 0, self.video_height)
            tracker = tracker.astype(np.uint32)    

            x1, y1, x2, y2 = tracker

            cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors[track_idx,:], 2)

            name_text = self.TRACKER_NAMES[track_idx-1]            
            text_color = (255, 255, 255) if sum(self.colors[track_idx]) < 382 else (0, 0, 0)
            text_size, _ = cv2.getTextSize(name_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_width, text_height = text_size
            # Draw filled rectangle behind text for better visibility
            cv2.rectangle(frame, (x1, y1 - text_height - 4), (x1 + text_width, y1), self.colors[track_idx], -1)
            # Draw text shadow in contrasting color
            cv2.putText(frame, name_text, (x1 + 1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    def draw_trajectories(self, frame: np.ndarray) -> None:
        """
        Draw the trajectories on the frame.

        Args:
            frame: image frame to draw on
        """
        if len(self.frame_trajectories) < 2:
            return
        
        for i in range(len(self.frame_trajectories)):
            if i == 0:
                continue

            current_trajectories = self.frame_trajectories[i]
            previous_trajectories = self.frame_trajectories[i-1]

            for current_trajectory in current_trajectories:
                current_x, current_y, current_id = current_trajectory

                for previous_trajectory in previous_trajectories:
                    previous_x, previous_y, previous_id = previous_trajectory

                    if current_id == previous_id:
                        track_idx = current_id % self.max_trackers
                        cv2.line(frame, (int(previous_x), int(previous_y)), (int(current_x), int(current_y)), self.colors[track_idx,:], 2)

    def write_frame(self, frame: np.ndarray) -> None:
        self.out_video.write(frame)

        if len(self.frame_trajectories) > self.max_trajectory_length:
            self.frame_trajectories.pop(0)


def store_trackers_list(
    trackers_list: List[List[BoundingBox]],
    save_tracking_path: str,
    file_mode: str = "a",
    ):
    # trackers_list is a list of lists, where each list contains the bounding boxes of a frame
    used_frame_track = set()
    results_file = open(save_tracking_path, file_mode)
    for trackers in trackers_list:
        for d in trackers:
            # Save tracking with bounding boxes in MOT Challenge format:
            # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, -1, -1, -1, -1
            if not (d.frame+1, d.track_id) in used_frame_track:  # avoid duplicates
                used_frame_track.add((d.frame+1, d.track_id))
                results_file.write(
                    f"{d.frame+1},{d.track_id},{d.x1},{d.y1},{d.x2-d.x1},{d.y2-d.y1},{d.confidence if d.confidence else '-1'},-1,-1,-1\n"
                )
    results_file.close()


def filter_by_id(keep_id, trackers_list: List[List[BoundingBox]]):
    filtered_trackers_list = []
    for trackers in trackers_list:
        trackers_filt = [d for d in trackers if d.track_id in keep_id]
        filtered_trackers_list.append(trackers_filt)
    return filtered_trackers_list


def filter_by_area(cfg: Dict, trackers_list: List[List[BoundingBox]]):
    # keep track of the area of each track over time
    trackId_area = {}
    for trackers in trackers_list:
        for d in trackers:
            if d.track_id not in trackId_area:
                trackId_area[d.track_id] = []
            trackId_area[d.track_id].append(d.area)

    keep_id = set()
    # Compute the average area of each track
    for track_id in trackId_area:
        trackId_area[track_id] = np.mean(trackId_area[track_id])
        # Keep only the tracks with an area above a threshold
        if trackId_area[track_id] >= cfg["filter_area_threshold"]:
            keep_id.add(track_id)

    # Finally, store only the tracks that are not parked
    filtered_trackers_list = filter_by_id(keep_id, trackers_list)
    return filtered_trackers_list


def filter_parked(cfg: Dict, trackers_list: List[List[BoundingBox]]):
    """ Discards parked vehicles """
    # Compute the center of the bounding box for each frame and track
    bbox_center = {}  # track_id -> list of (x,y) coordinates
    for trackers in trackers_list:
        for d in trackers:
            if d.track_id not in bbox_center:
                bbox_center[d.track_id] = []
            bbox_center[d.track_id].append([d.center_x, d.center_y])

    # Compute the std of the bounding boxes center for each track
    keep_id = set()
    for track_id in bbox_center:
        bbox_center[track_id] = np.std(bbox_center[track_id], axis=0)
        if bbox_center[track_id][0] >= cfg["filter_parked_threshold"] or bbox_center[track_id][1] >= cfg["filter_parked_threshold"]:
            keep_id.add(track_id)

    # Finally, store only the tracks that are not parked
    filtered_trackers_list = filter_by_id(keep_id, trackers_list)
    return filtered_trackers_list


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
