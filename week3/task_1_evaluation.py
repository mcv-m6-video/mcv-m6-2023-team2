from utils import (
    load_annotations,
    load_predictions,
    generate_test_subset,
    group_annotations_by_frame,
)
from metrics import voc_eval


def filter_by_conf(detections, conf_thr=0.5):
    filtered_detections = []
    for det in detections:
        if det.confidence >= conf_thr:
            filtered_detections.append(det)

    return filtered_detections


def task_1_1_evaluation(args):
    gt = load_annotations(args.path_GT, grouped=False, use_parked=True)
    det = load_predictions(args.path_det, grouped=False)

    det = filter_by_conf(det, conf_thr=args.min_conf)

    test_perc = 0.75
    test_gt = generate_test_subset(gt, num_frames=2141, test_perc=test_perc)
    test_det = generate_test_subset(det, num_frames=2141, test_perc=test_perc)

    rec, prec, ap, iou = voc_eval(
        group_annotations_by_frame(test_det),
        group_annotations_by_frame(test_gt),
        args.min_iou,
    )
    print('AP' + str(args.min_iou) + ': ', ap)
    print(f'Recall: {rec}  |  Precision: {prec}  |  IoU: {iou}')
