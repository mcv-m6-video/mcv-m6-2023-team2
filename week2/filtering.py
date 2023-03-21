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
            del BBs[d-_d]
            _d += 1

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

        # TODO: take bikes from GT
        BBs.append(
            BoundingBox(
                x1=x,
                y1=y,
                x2=x+w,
                y2=y+h,
                frame=frame_ID,
                track_id=-1,
                label='car',  # we are assuming only cars were detected
            )
        )

        idx += 1

    # return BBs
    return eliminate_overlapping_boxes(BBs)


def spatial_morphology(img, iterations=1):

    img = cv2.erode(img, np.ones((2, 2), np.uint8), iterations=iterations)
    img = cv2.dilate(img, np.ones((3,4), np.uint8), iterations=iterations)

    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3, 6), np.uint8))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((6, 3), np.uint8))

    return img


def filter_detections_temporal(detects):
    """
    Filter detections along temporal axis.
    Discards those predictions that are not consistent (have low IoU)
    with the detections in the previous and next frames.
    """

    # start time counter
    import time
    start_time = time.time()

    accepted_detects = []

    accepted_detects.append(detects[0])

    iou_thr = 0.5
    for i in range(1, len(detects) - 1):

        detect_curr = detects[i]
        if len(detect_curr) == 0:
            accepted_detects.append(detect_curr)
            continue

        detect_prev = []
        detect_next = []
        try:
            if len(detects[i-1]) > 0 and detects[i-1][0].frame == detect_curr[0].frame - 1:
                detect_prev = detects[i-1]
            if len(detects[i+1]) > 0 and detects[i+1][0].frame == detect_curr[0].frame + 1:
                detect_next = detects[i+1]
        except IndexError as e:
            print(e, i, len(detects))
            pass

        _detect_curr = []
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
                _detect_curr.append(d_curr)
            else:
                print(f"Filtering out prediction for frame {d_curr.frame}")

        accepted_detects.append(_detect_curr)

    if len(detects) > 1:
        accepted_detects.append(detects[-1])

    print(f"Temporal filtering took {time.time() - start_time} seconds")

    return accepted_detects
