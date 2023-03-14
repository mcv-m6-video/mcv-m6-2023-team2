import math
from typing import List
import cv2
import xmltodict
import numpy as np
import matplotlib.pyplot as plt

from class_utils import BoundingBox
from metrics import frame_voc_iou


def iou_over_time(video_path: str, annotations: List[BoundingBox], predictions: List[BoundingBox], show_video=False) \
        -> float:
    """
    Shows the given annotations and predictions in the given video and returns the mean IoU.

    :param video_path: Path to the video.
    :param annotations: List of annotations.
    :param predictions: List of predictions.

    """
    grouped_annotations = group_annotations_by_frame(annotations)
    grouped_predictions = group_annotations_by_frame(predictions)

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`

    my_dpi = 96
    fig = plt.figure(figsize=(width/my_dpi, height/my_dpi), dpi=my_dpi)
    plot, = plt.plot(np.arange(total_frames), np.ones(total_frames))

    miou = []
    for idx_frame in range(total_frames):
        ret, frame = video.read()
        if not ret:
            break

        if show_video:
            for box in annotations:
                if box.frame == idx_frame:
                    cv2.rectangle(frame, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), (0, 255, 0), 2)

            for box in predictions:
                if box.frame == idx_frame:
                    cv2.rectangle(frame, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), (0, 0, 255), 2)

            # progress bar
            cv2.rectangle(frame, (0, height - 25), (int(idx_frame * (width / total_frames)), width), (0, 255, 0), -1)

            # cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        iou = frame_voc_iou(grouped_annotations[idx_frame], grouped_predictions[idx_frame])
        miou.append(iou)

        plot.set_ydata(miou)
        fig.canvas.draw()
        plot_img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)
        # cv2.imshow("plot", plot_img)

        side_by_side = np.concatenate((frame, plot_img), axis=1)
        cv2.imshow("plot", side_by_side)

    video.release()
    cv2.destroyAllWindows()

    return np.mean(miou)


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


def histogram_error_distribution(error, GT):
    max_range = int(math.ceil(np.amax(error)))

    plt.title('Mean square error distribution')
    plt.ylabel('Density')
    plt.xlabel('Mean square error')
    plt.hist(error[GT[2] == 1].ravel(), bins=30, range=(0.0, max_range))
