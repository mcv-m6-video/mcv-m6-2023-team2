import cv2
from tqdm import tqdm

import matplotlib.pyplot as plt
import copy

from utils import (
    load_annotations,
    draw_boxes,
    draw_legend,
    save_image,
    create_gif,
    group_by_frame,
    group_annotations_by_frame,
)

from filtering import (
    spatial_morphology,
    extract_foreground,
    filter_detections_temporal,
)
from metrics import voc_eval


def subtract_test_partition(video, test_frame_start, total_frames, model, args, h = None, w = None):
    

    gt = load_annotations(args['path_GT'], select_label_types=['car'], grouped=True, use_parked=False)
    detections, annotations = [], []
    if args['make_video']: outvid = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), video.get(cv2.CAP_PROP_FPS), (w,h))
    roi = cv2.imread(args['path_roi'], cv2.IMREAD_GRAYSCALE) / 255

    for t in tqdm(range(total_frames), desc='Fitting and testing OpenCV background model...'):
        _, frame = video.read()
        fgMask = model.apply(frame)


        ### Evaluation zone ###

        if t > test_frame_start:


            gt_bboxes = gt[t] if len(gt[t]) else []
            
            segmentation = fgMask * roi
            detected_bboxes = extract_foreground(segmentation, t, args)
            detections.append([detected_bboxes])
            annotations.append([gt_bboxes])

        
        if args['make_video']:
            text = f'testing... {t - test_frame_start}' if t > test_frame_start else f'training... {t}'
            color = (0, 255, 0) if t > test_frame_start else (0, 0, 255)
            img = cv2.cvtColor(copy.deepcopy(fgMask), cv2.COLOR_GRAY2BGR)
            img = cv2.putText(img=img, text=text, org=(100, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=color,thickness=2)
            outvid.write(img)
    outvid.release()
    recall, precision, F1, AP, IoU = voc_eval(detections, annotations, ovthresh=0.5)
    print(
        'recall:', recall,
        'precision:', precision,
        'F1:', F1,
        'AP:', AP,
        'IoU:', IoU)
    
    return recall, precision, F1, AP, IoU


