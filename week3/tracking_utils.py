import cv2
import numpy as np

from typing import List


class TrackingViz:
    TRACKER_NAMES = ['John Wick', 'Viggo Tarasov', 'Iosef Tarasov', 'Marcus', 'Winston', 'Ms. Perkins', 'Aurelio', 'Charon', 'Harry', 'Jimmy', 'Francisco', 'Kirill', 'Abram Tarasov', 'Wolfgang', 'Perkins', 'Avi', 'Wick\'s neighbor', 'Priest', 'Helen Wick', 'Julius', 'Mrs. Tarasov', 'Baba Yaga', 'Viggo\'s lawyer', 'Mrs. Perkins', 'Doctor', 'Continental hotel clerk', 'Continental hotel doctor', 'Continental hotel sommelier', 'Continental hotel bartender', 'Continental hotel concierge', 'Continental hotel housekeeper', 'Continental hotel doorman', 'Continental hotel security']

    def __init__(self, 
                 output_video_path: str, 
                 video_width: int, 
                 video_height: int, 
                 fps: int, 
                 fourcc: str = "xvid", 
                 max_trackers: int = 32
                 ) -> None:
        self.out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (video_width, video_height))
        self.video_width = video_width
        self.video_height = video_height
        self.colors = np.random.rand(max_trackers, 3) * 255
        self.max_trackers = max_trackers

    def draw_tracks(self, frame: np.ndarray, trackers: List[np.ndarray]) -> None:
        """
        Draw bounding boxes on the frame and write it to the output video.

        Args:
            frame: image frame to draw on
            trackers: list of bounding boxes in format [x1, y1, x2, y2, track_id]
        """
        for tracker in trackers:
            tracker = np.array(tracker)
            tracker[0] = np.clip(tracker[0], 0, self.video_width)
            tracker[1] = np.clip(tracker[1], 0, self.video_height)
            tracker[2] = np.clip(tracker[2], 0, self.video_width)
            tracker[3] = np.clip(tracker[3], 0, self.video_height)
            tracker = tracker.astype(np.uint32)    

            x1, y1, x2, y2, track_id = tracker
            track_idx = track_id%self.max_trackers

            cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors[track_idx,:], 2)

            name_text = self.TRACKER_NAMES[track_idx-1]            
            text_color = (255, 255, 255) if sum(self.colors[track_idx]) < 382 else (0, 0, 0)
            text_size, _ = cv2.getTextSize(name_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_width, text_height = text_size
            # Draw filled rectangle behind text for better visibility
            cv2.rectangle(frame, (x1, y1 - text_height - 4), (x1 + text_width, y1), self.colors[track_idx], -1)
            # Draw text shadow in contrasting color
            cv2.putText(frame, name_text, (x1 + 1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

        self.out_video.write(frame)