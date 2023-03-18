import cv2
import numpy as np
from tqdm import tqdm

from utils import (
    load_annotations,
    draw_boxes,
    group_by_frame,
)
from filtering import (
    spatial_morphology,
    extract_foreground,
    filter_detections_temporal,
)
from metrics import voc_eval

COLOR_SPACES = {
    # 'RGB': (cv2.COLOR_BGR2RGB, 3),
    # 'LAB': (cv2.COLOR_BGR2LAB, 3),
    # 'YUV': (cv2.COLOR_BGR2YUV, 3),
    # 'H': (cv2.COLOR_BGR2HSV, 1),
    # 'L': (cv2.COLOR_BGR2LAB, 1),
    # 'CbCr': (cv2.COLOR_BGR2YCrCb, 2),
    'grayscale': (cv2.COLOR_BGR2GRAY, 1),
    # 'YCrCb': (cv2.COLOR_BGR2YCrCb, 3),
    # 'HSV': (cv2.COLOR_BGR2HSV, 3),
}


def fixed_Gaussian_background(img, H_W, mean, std, args):
    alpha = args['alpha']
    h, w = H_W
    segmentation = np.zeros((h, w))
    mask = abs(img - mean) >= alpha * (std + 2)

    N_channels = COLOR_SPACES[args['color_space']][1]
    if N_channels == 1:
        segmentation[mask] = 255
    else:
        if args['voting'] == 'unanimous' or N_channels == 2:
            votes = (np.count_nonzero(mask, axis=2) / N_channels) >= 1
        elif args['voting'] == 'simple':
            votes = (np.count_nonzero(mask, axis=2) / (N_channels // 2 + 1)) >= 1
        else:
            raise ValueError('The specified Voting mechanism is not supported!')

        segmentation[votes] = 255

    return segmentation, mean


method_Gaussian_background = {
    'non_adaptive': fixed_Gaussian_background,
    # 'adaptive': # TODO: implement adaptive Gaussian background model
}


def eval(video_cv2, H_W, mean, std, args):
    GT = load_annotations(args['path_GT'], grouped=True, use_parked=False)
    init_frame_id = int(video_cv2.get(cv2.CAP_PROP_POS_FRAMES))
    frame_id = init_frame_id
    detections, annotations = [], {}
    for t in tqdm(range(args['N_eval'])):
        _, frame = video_cv2.read()
        frame = cv2.cvtColor(frame, COLOR_SPACES[args['color_space']][0])
        if args['color_space'] == 'H':
            H, S, V = np.split(frame, 3, axis=2)
            frame = np.squeeze(H)
        if args['color_space'] == 'L':
            L, A, B = np.split(frame, 3, axis=2)
            frame = np.squeeze(L)
        if args['color_space'] == 'CbCr':
            Y, Cb, Cr = np.split(frame, 3, axis=2)
            frame = np.dstack((Cb, Cr))
            
        segmentation, mean, std = method_Gaussian_background[args['bg_model']](frame, H_W, mean, std, args)
        roi = cv2.imread(args['path_roi'], cv2.IMREAD_GRAYSCALE) / 255
        segmentation = segmentation * roi
        segmentation = spatial_morphology(segmentation)

        if args['store_results'] and frame_id >= 1169 and frame_id < 1229 : # if frame_id >= 535 and frame_id < 550
            cv2.imwrite(args['path_results'] + f"seg_{str(frame_id)}_pp_{str(args['alpha'])}.bmp", segmentation.astype(int))

        detected_bboxes = extract_foreground(segmentation, frame_id, args)
        detections += detected_bboxes

        gt_bboxes = []
        if frame_id in GT:
            gt_bboxes = GT[frame_id]
        annotations[frame_id] = gt_bboxes

        if args['show_boxes']:
            segmentation = cv2.cvtColor(segmentation.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            segmentation_boxes = draw_boxes(image=segmentation, boxes=detected_bboxes, color='r', linewidth=3)
            segmentation_boxes = draw_boxes(image=segmentation_boxes, boxes=gt_bboxes, color='g', linewidth=3)

            cv2.imshow("Segmentation mask with detected and GT bboxes", segmentation_boxes)
            if cv2.waitKey() == 113:
                # press 'q' to exit
                break

        frame_id += 1

    detections = filter_detections_temporal(group_by_frame(detections), init=init_frame_id, end=frame_id)
    recall, precision, AP, IoU = voc_eval(detections, annotations, ovthresh=0.5, use_confidence=False)

    return AP


def fit(video_cv2, H_W, N_train, args):
    count = 0
    h, w = H_W
    num_ch = COLOR_SPACES[args['color_space']][1]
    if num_ch == 1:
        avg = np.zeros((h, w))
        SS = np.zeros((h, w))
    else:
        avg = np.zeros((h, w, num_ch))
        SS = np.zeros((h, w, num_ch))

    # Compute average and std
    for t in tqdm(range(N_train)):
        _, frame = video_cv2.read()
        frame = cv2.cvtColor(frame, COLOR_SPACES[args['color_space']][0])
        if args['color_space'] == 'H':
            H, S, V = np.split(frame,3,axis=2)
            frame = np.squeeze(H)
        if args['color_space'] == 'L':
            L, A, B = np.split(frame,3,axis=2)
            frame = np.squeeze(L)
        if args['color_space'] == 'CbCr':
            Y, Cb, Cr = np.split(frame,3,axis=2)
            frame = np.dstack((Cb,Cr))
        count += 1
        dev_avg = frame - avg
        avg += dev_avg / count
        dev_frame = frame - avg
        SS += dev_avg * dev_frame

    std = np.sqrt(SS / count)

    print("Mean and std have been calculated!")

    if args['save_results']:
        cv2.imwrite(args['path_results'] + "mean_train.png", avg)
        cv2.imwrite(args['path_results'] + "std_train.png", std)

    return avg, std
