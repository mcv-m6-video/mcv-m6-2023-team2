
import cv2
import os
import numpy as np
from typing import List
from week5.bounding_box import BoundingBox
from tqdm import tqdm
import pickle


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def str_frame_id(id):
    return '%04d' % id


def update_data(annot, frame_id, xmin, ymin, xmax, ymax, conf, obj_id=0):
    """
    Updates the annotations dict with by adding the desired data to it
    :param annot: annotation dict
    :param frame_id: id of the framed added
    :param xmin: min position on the x axis of the bbox
    :param ymin: min position on the y axis of the bbox
    :param xmax: max position on the x axis of the bbox
    :param ymax: max position on the y axis of the bbox
    :param conf: confidence
    :return: the updated dictionary
    """

    frame_name = '%04d' % int(frame_id)
    obj_info = dict(
        name='car',
        obj_id=obj_id,
        bbox=list(map(float, [xmin, ymin, xmax, ymax])),
        confidence=float(conf)
    )

    if frame_name not in annot.keys():
        annot.update({frame_name: [obj_info]})
    else:
        annot[frame_name].append(obj_info)

    return annot


def return_bb(det_bboxes, frame, bb_id):
    for bbox in det_bboxes[str_frame_id(frame)]:
        if bbox['obj_id'] == bb_id:
            return bbox['bbox']
    return None


def interpolate_bb(bb_first, bb_last, distance):
    bb_first = np.array(bb_first)
    bb_last = np.array(bb_last)
    #interpolate new bbox depending on de distance in frames between first and last bbox
    new_bb = bb_first + (bb_last-bb_first)/distance

    return list(np.round(new_bb, 2))


def compute_iou(bb_gt, bb, resize_factor=1):
    """
    iou = compute_iou(bb_gt, bb)
    Compute IoU between bboxes from ground truth and a single bbox.
    bb_gt: Ground truth bboxes
        Array of (num, bbox), num:number of boxes, bbox:(xmin,ymin,xmax,ymax)
    bb: Detected bbox
        Array of (bbox,), bbox:(xmin,ymin,xmax,ymax)
    """

    # intersection
    bb = bb / resize_factor

    ixmin = np.maximum(bb_gt[:, 0], bb[0])
    iymin = np.maximum(bb_gt[:, 1], bb[1])
    ixmax = np.minimum(bb_gt[:, 2], bb[2])
    iymax = np.minimum(bb_gt[:, 3], bb[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
           (bb_gt[:, 2] - bb_gt[:, 0] + 1.) *
           (bb_gt[:, 3] - bb_gt[:, 1] + 1.) - inters)

    return inters / uni


def tracking_by_maximum_overlap_with_optical_flow(
        det_bboxes: List[BoundingBox],
        threshold: float = 0.5,
        remove_noise: bool = True,
        interpolate: bool = False,
        frames_paths: List[str] = None,
        of_path: str = None,
):
    id_seq = {}
    # not assuming any order
    start_frame = int(min(det_bboxes.keys()))
    num_frames = int(max(det_bboxes.keys())) - start_frame + 1

    # init the tracking by  using the first frame
    for value, detection in enumerate(det_bboxes[str_frame_id(start_frame)]):
        detection['obj_id'] = value
        id_seq.update({value: True})

    for i in tqdm(range(start_frame, num_frames - 1), 'Frames Overlapping Tracking'):
        img1 = cv2.imread(frames_paths[i - 1])
        img2 = cv2.imread(frames_paths[i])

        if os.path.isfile(os.path.join(of_path, 'flow_' + str(i) + '.pkl')):
            with open(os.path.join(of_path, 'flow_' + str(i) + '.pkl'), 'rb') as f:
                flow = pickle.load(f)

            u = flow[:, :, 0]
            v = flow[:, :, 1]

        # init
        id_seq = {frame_id: False for frame_id in id_seq}

        for detection in det_bboxes[str_frame_id(i + 1)]:
            active_frame = i
            bbox_matched = False

            OF_x = u[int(detection['bbox'][1]):int(detection['bbox'][3]),
                   int(detection['bbox'][0]):int(detection['bbox'][2])]
            OF_y = v[int(detection['bbox'][1]):int(detection['bbox'][3]),
                   int(detection['bbox'][0]):int(detection['bbox'][2])]

            mag, ang = cv2.cartToPolar(OF_x.astype(np.float32), OF_y.astype(np.float32))
            # keep the values which is found the most for mag and ang
            uniques, counts = np.unique(mag, return_counts=True)
            mc_mag = uniques[counts.argmax()]
            uniques, counts = np.unique(ang, return_counts=True)
            mc_ang = uniques[counts.argmax()]
            x, y = pol2cart(mc_mag, mc_ang)

            detection['bbox_of'] = [detection['bbox'][0] - x, detection['bbox'][1] - y,
                                    detection['bbox'][2] - x, detection['bbox'][3] - y]

            # if there is no good match on previous frame, check n-1 up to n=5
            while (bbox_matched == False) and (active_frame >= start_frame) and ((i - active_frame) < 5):
                candidates = [candidate['bbox'] for candidate in det_bboxes[str_frame_id(active_frame)]]
                # compare with all detections in previous frame
                # best match
                iou = compute_iou(np.array(candidates), np.array(detection['bbox_of']))
                while np.max(iou) > threshold:
                    # candidate found, check if free
                    matching_id = det_bboxes[str_frame_id(active_frame)][np.argmax(iou)]['obj_id']
                    if id_seq[matching_id] == False:
                        detection['obj_id'] = matching_id
                        bbox_matched = True
                        # interpolate bboxes
                        if i != active_frame and interpolate:
                            frames_skip = i - active_frame
                            for j in range(frames_skip):
                                new_bb = interpolate_bb(return_bb(det_bboxes, (active_frame + j), matching_id),
                                                        detection['bbox'], frames_skip - j + 1)
                                update_data(det_bboxes, (active_frame + 1 + j), *new_bb, 0, matching_id)
                        break
                    else:  # try next best match
                        iou[np.argmax(iou)] = 0
                active_frame = active_frame - 1

            if not bbox_matched:
                # new object
                detection['obj_id'] = max(id_seq.keys()) + 1

            id_seq.update({detection['obj_id']: True})

    # filter by number of ocurrences
    if remove_noise:
        id_ocurrence = {}
        # Count ocurrences
        for i in range(start_frame, num_frames):
            for detection in det_bboxes[str_frame_id(i)]:
                objt_id = detection['obj_id']
                if objt_id in id_ocurrence:
                    id_ocurrence[objt_id] += 1
                else:
                    id_ocurrence[objt_id] = 1
        # detectiosn to be removed
        ids_to_remove = [id_obj for id_obj in id_ocurrence if id_ocurrence[id_obj] < 4]
        for i in range(start_frame, num_frames):
            for idx, detection in enumerate(det_bboxes[str_frame_id(i)]):
                if detection['obj_id'] in ids_to_remove:
                    det_bboxes[str_frame_id(i)].pop(idx)

    return det_bboxes