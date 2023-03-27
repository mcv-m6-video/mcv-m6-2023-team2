import cv2
import numpy as np

from typing import List

from class_utils import BoundingBox


class TrackingViz:
    TRACKER_NAMES = ['John Wick', 'Viggo Tarasov', 'Iosef Tarasov', 'Marcus', 'Winston', 'Ms. Perkins', 'Aurelio', 'Charon', 'Harry', 'Jimmy', 'Francisco', 'Kirill', 'Abram Tarasov', 'Wolfgang', 'Perkins', 'Avi', 'Wick\'s neighbor', 'Priest', 'Helen Wick', 'Julius', 'Mrs. Tarasov', 'Baba Yaga', 'Viggo\'s lawyer', 'Mrs. Perkins', 'Doctor', 'Continental hotel clerk', 'Continental hotel doctor', 'Continental hotel sommelier', 'Continental hotel bartender', 'Continental hotel concierge', 'Continental hotel housekeeper', 'Continental hotel doorman', 'Continental hotel security']

    def __init__(self, 
                 output_video_path: str, 
                 video_width: int, 
                 video_height: int, 
                 fps: int, 
                 fourcc: str = cv2.VideoWriter_fourcc(*'xvid'), 
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