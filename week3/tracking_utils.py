import cv2
import numpy as np

from typing import List
from class_utils import BoundingBox


class Track(object):
    """
    Class to represent a track (object or instance trajectory).
    """
    def __init__(self, id: int, first_detection: BoundingBox, first_frame_number: int):
        self.id = id
        self.detections = [first_detection]
        self.frame_numbers = [first_frame_number]
        self.terminated = False

    def get_track_boxes(self):
        return self.detections

    def add_detection(self, detection: BoundingBox, frame_number: int):
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


class TrackingViz:
    TRACKER_NAMES = ['John Wick', 'Viggo Tarasov', 'Iosef Tarasov', 'Marcus', 'Winston', 'Ms. Perkins', 'Aurelio', 'Charon', 'Harry', 'Jimmy', 'Francisco', 'Kirill', 'Abram Tarasov', 'Wolfgang', 'Perkins', 'Avi', 'Wick\'s neighbor', 'Priest', 'Helen Wick', 'Julius', 'Mrs. Tarasov', 'Baba Yaga', 'Viggo\'s lawyer', 'Mrs. Perkins', 'Doctor', 'Continental hotel clerk', 'Continental hotel doctor', 'Continental hotel sommelier', 'Continental hotel bartender', 'Continental hotel concierge', 'Continental hotel housekeeper', 'Continental hotel doorman', 'Continental hotel security']

    def __init__(self, 
                 output_video_path: str, 
                 video_width: int, 
                 video_height: int, 
                 fps: int, 
                 fourcc: str = cv2.VideoWriter_fourcc(*'FMP4'),
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