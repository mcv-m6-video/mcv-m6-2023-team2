
from tqdm import tqdm
from utils import (
    load_predictions,
    load_annotations, 
    group_annotations_by_frame, 
    create_fake_track_predictions
)
from metrics import voc_eval


def task1_1():
    # TODO: Reformat and make it prettier
    annotations = load_annotations('data/AICity_data/train/S03/c010/ai_challenge_s03_c010-full_annotation.xml')
    grouped_annotations = group_annotations_by_frame(annotations)
    annotations_with_noise = create_fake_track_predictions(annotations, noise=0.1, prob_generate=0.1, prob_delete=0.1)
    grouped_annotations_with_noise = group_annotations_by_frame(annotations_with_noise)

    n_frames = len(grouped_annotations)

    print(f'Number of frames: {n_frames}')
    print(f'Number of annotations: {len(annotations)}')
    print(f'Number of annotations with noise: {len(annotations_with_noise)}')

    rec, prec, ap = voc_eval(grouped_annotations_with_noise, grouped_annotations)

    print(f'AP: {ap}')
    print(f'Precision: {prec}')
    print(f'Recall: {rec}')


def task1_2():
    annotations = load_annotations('data/AICity_data/train/S03/c010/ai_challenge_s03_c010-full_annotation.xml')
    grouped_annotations = group_annotations_by_frame(annotations)
    predictions = load_predictions('data/AICity_data/train/S03/c010/det/det_mask_rcnn.txt')
    grouped_predictions = group_annotations_by_frame(predictions)

    n_frames = len(grouped_annotations)

    print(f'Number of frames: {n_frames}')
    print(f'Number of annotations: {len(annotations)}')
    print(f'Number of predictions: {len(predictions)}')

    rec, prec, ap = voc_eval(grouped_predictions, grouped_annotations)

    print(f'AP: {ap}')
    print(f'Precision: {prec}')
    print(f'Recall: {rec}')
        