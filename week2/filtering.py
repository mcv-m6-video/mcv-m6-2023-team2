from copy import deepcopy
import cv2
import numpy as np
from class_utils import (
    IoA,
    IoU,
    BoundingBox,
)


def eliminate_overlapping_boxes(BBs):
    intersec_th = 0.7
    idx_A = 0
    tmp_BBs = deepcopy(BBs)
    while idx_A < len(BBs)-1:
        discarded_idx = []
        discarded_tmp = []
        discarded_A = False

        bboxA = BBs[idx_A]
        del tmp_BBs[0]

        for idx_B, bboxB in enumerate(tmp_BBs):
            intesec_A, intesec_B = IoA(bboxA, bboxB)

            if intesec_A > intersec_th or intesec_B > intersec_th:
                if intesec_A > intesec_B:
                    discarded_A = True
                    if idx_A not in discarded_idx:
                        discarded_idx.append(idx_A)
                else:
                    discarded_idx.append(idx_A + idx_B + 1)
                    discarded_tmp.append(idx_B)

        _d = 0
        for d in sorted(discarded_idx):
            del BBs[d-discarded]
            discarded += 1

        _d = 0
        for d in sorted(discarded_tmp):
            del tmp_BBs[d-_d]
            _d += 1

        if not discarded_A:
            idx_A +=1

        if len(tmp_BBs) == 0:
            break

    return BBs


def extract_foreground(img, frame_ID, args):
    BBs = []
    ROI = cv2.imread(args['path_roi'], cv2.IMREAD_GRAYSCALE) / 255
    segm_img = img * ROI
    contours, _ = cv2.findContours(segm_img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    idx = 0
    for c in contours:
        rect = cv2.boundingRect(c)
        if rect[2] < 50 or rect[3] < 50 or rect[2]/rect[3] < 0.8:
            continue  # Discard small contours

        x, y, w, h = rect

        # TODO: pillar bicis a gt
        BBs.append(
            BoundingBox(
                x1=x,
                y1=y,
                x2=x+w,
                y2=y+h,
                frame=frame_ID,
                track_id=-1,
                label='car',  # careful, we are assuming only cars were detected
            )
        )

        idx += 1

    return eliminate_overlapping_boxes(BBs)


def spatial_morphology(img, iterations=1):

    img = cv2.erode(img, np.ones((2, 2), np.uint8), iterations=iterations)
    img = cv2.dilate(img, np.ones((3,4), np.uint8), iterations=iterations)

    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3, 6), np.uint8))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((6, 3), np.uint8))

    return img


def filter_detections_temporal(detects, init_id, end_id):
    """
    Filter detections along temporal axis.
    Discards those predictions that are not consistent (have low IoU)
    with the detections in the previous and next frames.
    """
    accepted_detects = []

    if init_id in detects:
        for d in detects[init_id]:
            accepted_detects.append(d)

    if end_id - 1 in detects:
        for d in detects[end_id - 1]:
            accepted_detects.append(d)

    iou_thr = 0.55
    for curr_frame in range(init_id + 1, end_id - 1):
        if curr_frame not in detects:
            continue
        detect_curr = detects[curr_frame]
        detect_prev = []
        detect_next = []
        if curr_frame - 1 in detects:
            detect_prev = detects[curr_frame - 1]
        if curr_frame + 1 in detects:
            detect_next = detects[curr_frame + 1]

        for d_curr in detect_curr:
            max_iou_prev = 0
            max_iou_next = 0

            for d_prev in detect_prev:
                iou_prev = IoU(d_curr, d_prev)
                max_iou_prev = max(max_iou_prev, iou_prev)

            for d_next in detect_next:
                iou_next = IoU(d_curr, d_next)
                max_iou_next = max(max_iou_next, iou_next)

            if max_iou_prev >= iou_thr or max_iou_next >= iou_thr:
                accepted_detects.append(d_curr)

    return accepted_detects
