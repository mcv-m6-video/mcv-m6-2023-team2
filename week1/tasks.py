
from tqdm import tqdm
from utils import load_annotations, group_annotations_by_frame, create_fake_track_predictions


def task1_1():
    annotations = load_annotations('data/ai_challenge_s03_c010-full_annotation.xml')
    grouped_annotations = group_annotations_by_frame(annotations)
    annotations_with_noise = create_fake_track_predictions(annotations, noise=0.1, prob_generate=0.1, prob_delete=0.1)
    grouped_annotations_with_noise = group_annotations_by_frame(annotations_with_noise)

    n_frames = len(grouped_annotations)

    print(f'Number of frames: {n_frames}')
    print(f'Number of annotations: {len(annotations)}')
    print(f'Number of annotations with noise: {len(annotations_with_noise)}')

    for frame in tqdm(range(n_frames)):
        frame_annotations = grouped_annotations[frame]
        frame_annotations_with_noise = grouped_annotations_with_noise[frame]

        