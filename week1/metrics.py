import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import load_optical_flow


def voc_ap(rec, prec):
    """
    Compute VOC AP given precision and recall using the VOC 07 11-point method.

    Original code from https://github.com/facebookresearch/detectron2/blob/main/detectron2/evaluation/pascal_voc_evaluation.py
    """
    ap = 0.0
    
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(rec >= t) == 0:
            p = 0
        else:
            p = np.max(prec[rec >= t])
        ap = ap + p / 11.0

    return ap


def voc_iou(pred, gt):
    """
    Calculate IoU between detect box and gt boxes.
    """
    # compute overlaps
    # intersection
    ixmin = np.maximum(gt[:, 0], pred[0])
    iymin = np.maximum(gt[:, 1], pred[1])
    ixmax = np.minimum(gt[:, 2], pred[2])
    iymax = np.minimum(gt[:, 3], pred[3])
    iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
    ih = np.maximum(iymax - iymin + 1.0, 0.0)
    inters = iw * ih

    # union
    uni = (
        (pred[2] - pred[0] + 1.0) * (pred[3] - pred[1] + 1.0)
        + (gt[:, 2] - gt[:, 0] + 1.0) * (gt[:, 3] - gt[:, 1] + 1.0)
        - inters
    )

    return inters / uni


def voc_eval(preds, gt, ovthresh=0.5):
    """
    rec, prec, ap = voc_eval(preds,
                            gt,
                            [ovthresh],
                            )
    Top level function that does the PASCAL VOC evaluation.
    gt: Ground truth bounding boxes grouped by frames.
    preds: Predicted bounding boxes grouped by frames.
    [ovthresh]: Overlap threshold (default = 0.5)

    Original code from https://github.com/facebookresearch/detectron2/blob/main/detectron2/evaluation/pascal_voc_evaluation.py
    """
    # extract gt objects for this class
    class_recs = {}
    npos = 0
    
    for i, frame in enumerate(gt):
        bbox = np.array([bbox.coordinates for bbox in frame])
        difficult = np.array([False for bbox in frame]).astype(bool)
        det = [False] * len(frame)
        npos = npos + sum(~difficult)
        class_recs[i] = {"bbox": bbox, "difficult": difficult, "det": det}

    # read dets
    image_ids = []
    confidence = []
    BB = []

    for i, frame in enumerate(preds):
        image_ids += [i] * len(preds[i])
        confidence += [bbox.confidence for bbox in preds[i]]
        BB += [bbox.coordinates for bbox in preds[i]]

    confidence = np.array(confidence)
    BB = np.array(BB).reshape(-1, 4)

    if np.all(confidence != None): # Podria haver-hi algun confidence = None?
        sorted_ind = np.argsort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    iou = np.zeros(nd)
    
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            overlaps = voc_iou(bb, BBGT)
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)
            iou[d] = ovmax

        if ovmax > ovthresh:
            if not R["difficult"][jmax]:
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    R["det"][jmax] = 1
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec)
    iou = np.mean(iou)

    return rec, prec, ap, iou


def OF_MSEN(GT, pred, frame: str, verbose=False, visualize=True):
    """
    Computes "Mean Square Error in Non-occluded areas"
    """

    u_diff, v_diff = GT[:, :, 0] - pred[:, :, 0], GT[:, :, 1] - pred[:, :, 1]
    se = np.sqrt(u_diff ** 2 + v_diff ** 2)
    sen = se[GT[:, :, 2] == 1]
    msen = np.mean(sen)

    if verbose:
        print(GT[0, -1])
        print(pred[0, -1])
        print(u_diff[0, -1])
        print(v_diff[0, -1])
        print(se[0, -1])

    if visualize:
        se[GT[:, :, 2] == 0] = 0  # Exclude non-valid pixels
        plt.figure(figsize=(11, 4))
        img_plot = plt.imshow(se)
        img_plot.set_cmap("Blues")
        plt.title(f"Mean Square Error in Non-Occluded Areas - {frame}")
        plt.colorbar()
        plt.savefig(f'./results/OF_squareError_{frame}.png')
        plt.clf()

        pred, _ = cv2.cartToPolar(pred[:, :, 0], pred[:, :, 1])
        plt.figure(figsize=(11, 4))
        img_plot = plt.imshow(pred)
        plt.clim(0,4)
        img_plot.set_cmap("YlOrRd")
        plt.title(f"Optical Flow Prediction - {frame}")
        plt.colorbar()
        plt.savefig(f'./results/OF_pred_{frame}.png')
        plt.clf()

    return msen, sen


def OF_PEPN(sen, th=3):
    """
    Compute "Percentage of Erroneous Pixels in Non-Occluded Areas"
    """

    n_pixels_n = len(sen)
    error_count = np.sum(sen > th)
    pepn = 100 * (1 / n_pixels_n) * error_count

    return pepn
