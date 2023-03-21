import cv2
import tqdm

import matplotlib.pyplot as plt

from utils import (
    load_annotations,
    draw_boxes,
    draw_legend,
    save_image,
    create_gif,
    group_by_frame,
    group_annotations_by_frame,
)

def subtract_test_partition(video, test_frame_start, total_frames, model, args):
    

    gt = load_annotations(args['path_GT'], select_label_types=['car'], grouped=True, use_parked=False)
    detections, annotations = [], []

    for t in tqdm(range(total_frames), desc='Fitting and testing OpenCV background model...'):
        _, frame = video.read()
        fgMask = model.apply(frame)
        plt.imshow(fgMask)
        plt.show()
        0/0
