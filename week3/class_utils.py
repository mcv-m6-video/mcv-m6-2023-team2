from dataclasses import dataclass
from typing import Optional


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    track_id: int
    frame: int
    label: Optional[str] = None
    parked: Optional[bool] = None
    confidence: Optional[float] = None

    def __post_init__(self):
        self.width = self.x2 - self.x1
        self.height = self.y2 - self.y1
        self.__update_center()

    def __repr__(self):
        return f'BoundingBox(x1={self.x1}, y1={self.y1}, x2={self.x2}, y2={self.y2}, frame={self.frame}, track_id={self.track_id}, label={self.label})'

    def __eq__(self, other):
        return (self.x1 == other.x1 and self.y1 == other.y1
                and self.x2 == other.x2 and self.y2 == other.y2
                and self.frame == other.frame
                and self.track_id == other.track_id
                and self.label == other.label
                )

    def __hash__(self):
        return hash((self.x1, self.y1, self.x2, self.y2, self.frame, self.track_id, self.label))

    @property
    def coordinates(self):
        return [self.x1, self.y1, self.x2, self.y2]

    @property
    def coordinates_dim(self):
        return [self.x1, self.y1, self.width, self.height]

    @property
    def area(self):
        return self.width * self.height

    def __update_center(self):
        self.center_x = self.x1 + self.width / 2
        self.center_y = self.y1 + self.height / 2

    def resize(self, width, height):
        """
        Resize the bounding box to the given width and height and keep the center point.
        """
        self.x1 = self.center_x - width / 2
        self.x2 = self.center_x + width / 2
        self.y1 = self.center_y - height / 2
        self.y2 = self.center_y + height / 2
        self.width = width
        self.height = height

    def move(self, x, y):
        """
        Move the bounding box by the given x and y values.
        """
        self.x1 += x
        self.x2 += x
        self.y1 += y
        self.y2 += y
        self.__update_center()

    def clone(self):
        return BoundingBox(self.x1, self.y1, self.x2, self.y2, self.track_id, self.frame, self.label, self.parked)


    def intersection_bboxes(self, bboxB):
        """Computes the intersection area of two bounding boxes."""
        xA = max(self.x1, bboxB.x1)
        yA = max(self.y1, bboxB.y1)
        xB = min(self.x2, bboxB.x2)
        yB = min(self.y2, bboxB.y2)
        return max(0, xB - xA) * max(0, yB - yA)


    def IoA(self, bboxB):
        """Computes the intersection over areas of two bounding boxes."""
        intersecArea = self.intersection_bboxes(bboxB)
        return intersecArea / self.area, intersecArea / bboxB.area


    def IoU(self, bboxB):
        """Computes the intersection over union of two bounding boxes."""
        interArea = self.intersection_bboxes(bboxB)
        iou = interArea / float(self.area + bboxB.area - interArea)
        return iou
