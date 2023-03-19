import cv2
import numpy as np
from tqdm import tqdm

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

    return segmentation, mean, std


def adaptive_Gaussian_background(img, H_W, mean, std, args):
    alpha = args['alpha']
    rho = args['rho']
    h, w = H_W
    mask = abs(img - mean) >= alpha * (std + 2)

    segmentation = np.zeros((h, w))
    N_channels = COLOR_SPACES[args['color_space']][1]

    if N_channels == 1:
        segmentation[mask] = 255
    else:
        if args['voting'] == 'unanimous' or N_channels == 2:
            voting = (np.count_nonzero(mask, axis=2) / N_channels) >= 1
        elif args['voting'] == 'simple':
            voting = (np.count_nonzero(mask, axis=2) / (N_channels // 2 + 1)) >= 1
        else:
            raise ValueError('Voting method does not exist')

        segmentation[voting] = 255

    # Try update the background every N frames
    mean = np.where(mask, mean, rho * img + (1 - rho) * mean)
    std = np.where(mask, std, np.sqrt(rho * (img - mean) ** 2 + (1 - rho) * std ** 2))

    return segmentation, mean, std


method_Gaussian_background = {
    'non_adaptive': fixed_Gaussian_background,
    'adaptive': adaptive_Gaussian_background,
}


def eval(video_cv2, H_W, mean, std, N_val, args):
    GT = load_annotations(args['path_GT'], select_label_types=['car'], grouped=True, use_parked=False)
    init_frame_id = int(video_cv2.get(cv2.CAP_PROP_POS_FRAMES))
    frame_id = init_frame_id
    detections, annotations = [], []
    i_GT = frame_id
    for t in tqdm(range(N_val), desc='Evaluating Gaussian background model... (this may take a while)'):
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

        if args['store_results'] and args['frames_range'][0] <= frame_id < args['frames_range'][1]:
            save_image(segmentation.astype(int), frame_id, args, subfolder='segm', extension='.bmp')

        detected_bboxes = extract_foreground(segmentation, frame_id, args)
        detections += [detected_bboxes]

        gt_bboxes = []
        if len(GT[i_GT]) == 0:
            gt_bboxes = []
            # print('if len(GT[i_GT]) == 0:')
        else:
            gt_bboxes = GT[i_GT]
        annotations += [gt_bboxes]
        i_GT += 1
        # if frame_id < GT[i_GT][0].frame:  # GT is assumed to be sorted by frame id
        #     print('if frame_id < GT[i_GT][0].frame:' * 3)
        #     gt_bboxes = []
        # elif GT[i_GT][0].frame < frame_id:
        #     while GT[i_GT][0].frame < frame_id:
        #         i_GT += 1
        #     gt_bboxes = []
        # if frame_id == GT[i_GT][0].frame:
        #     print(frame_id, GT[i_GT][0].frame)
        #     gt_bboxes = GT[i_GT]
        #     i_GT += 1

        segmentation = cv2.cvtColor(segmentation.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        segmentation_boxes = draw_boxes(image=segmentation, boxes=detected_bboxes, color='r', linewidth=3)
        segmentation_boxes = draw_boxes(image=segmentation_boxes, boxes=gt_bboxes, color='g', linewidth=3)
        segmentation_boxes = draw_legend(image=segmentation_boxes, labels=['Ground truth', 'Detected'], colors=['g','r'])

        if args['store_results'] and args['frames_range'][0] <= frame_id < args['frames_range'][1]:
            save_image(segmentation_boxes.astype(int), frame_id, args, subfolder='segm_bbox', extension='.png')

        if args['viz_bboxes']:
            cv2.imshow("Segmentation mask with detected and GT bboxes", segmentation_boxes)
            if cv2.waitKey() == 113:
                # press 'q' to exit
                break

        frame_id += 1

    assert frame_id == i_GT
    assert len(annotations) == len(detections)

    # Create GIFs
    if args['make_gifs']:
        create_gif(args, subfolder='segm', extension='.bmp')
        create_gif(args, subfolder='segm_bbox', extension='.png')

    recall, precision, F1, AP, IoU = voc_eval(detections, annotations, ovthresh=0.5)

    return recall, precision, F1, AP, IoU


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
    for t in tqdm(range(N_train), desc='Fitting Gaussian background model... (computing mean and std)'):
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
        count += 1
        dev_avg = frame - avg
        avg += dev_avg / count
        dev_frame = frame - avg
        SS += dev_avg * dev_frame

    std = np.sqrt(SS / count)

    print("Mean and std have been calculated!")

    if args['store_results']:
        save_image(avg, 'mean_train', args, subfolder=None, extension='.png')
        save_image(std, 'std_train', args, subfolder=None, extension='.png')

    return avg, std
