import argparse
import os
import torch
import cv2
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

from tracking.tracking_utils import store_trackers_list, viz_tracking

from metric_learning.models.resnet import ResNetWithEmbedder
from metric_learning.models.vgg import VGG19

from utils import (
    load_config,
    load_predictions,
    group_annotations_by_track,
    group_annotations_by_frame,
    filter_annotations,
    non_maxima_suppression,
    load_optical_flow,
    return_image_full_range,
)

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Multi-Object Multi-Camera Tracking ID reassignment'
    )
    # Dataset configuration
    parser.add_argument('--dataset', type=str, default='aic19',
                        help='Dataset to use. Options: aic19.')
    parser.add_argument('--dataset_path', type=str, default='../data/aic19/train',
                        help='Path to the dataset.')
    parser.add_argument('--metric_dataset_path', type=str, default='./metric_learning_dataset',
                        help='Path to the metric learning dataset.')
    parser.add_argument('--detections_root', type=str, default='./detections/yolo',
                        help='Path to the detections root directory.')
    # Tracking data
    parser.add_argument('--path_tracking_data', type=str,
                        default="./data/trackers/mot_challenge/parabellum-train",
                        help='The path to the directory where the results will be stored.')
    parser.add_argument('--experiment', type=str, default='overlap_filtAreaTrue_filtParkedTrue',
                        help='Path to the results of single camera tracking.')
    parser.add_argument('--output_path', type=str, default='./results_metric_learning',
                        help='Path to the results of metric learning ID reassignment.')
    parser.add_argument('--sequence_name', type=str, default='S04',
                        help='Sequence name: S01, S03 or S04.')
    parser.add_argument('--threshold', type=float, default=0.90,
                        help='Max distance threshold for two tracks to be considered the same.')
    # Metric learning config
    parser.add_argument('--model', type=str, default='resnet_18',
                        help='Model to use. Options: resnet_X, vgg.')
    parser.add_argument('--embedding_size', type=int, default=256,
                        help='Size of the embedding vector.')
    parser.add_argument('--model_weights_path', type=str,
                        default="./model_final.pth",
                        help='Path to the model weights.')
    parser.add_argument('--n_neighbors', type=int, default=20,
                        help='Number of nearest neighbors to retrieve.')

    args = parser.parse_args()
    return args


def load_tracks(args: argparse.Namespace):
    track_files = os.listdir(os.path.join(args.path_tracking_data, args.experiment, 'data'))
    filtered_track_files = [track_file for track_file in track_files if args.sequence_name.lower() in track_file]
    # Crate a list of trackers
    trackers_dict = dict()
    track_idx = 1
    for track_file in filtered_track_files:
        # Get camera name from track file name
        camera_name = track_file.split('.')[0].split('_')[1]
        # Read the track file
        track_file_path = os.path.join(args.path_tracking_data, args.experiment, 'data', track_file)
        trackers = load_predictions(track_file_path)
        group_annotations = group_annotations_by_track(trackers)

        # Reassign track ids
        group_annotations = dict(enumerate(group_annotations.values(), start=track_idx))
        track_idx += len(group_annotations.keys())

        # Convert group annotations a list of lists
        new_list = []
        for key in group_annotations.keys():
            new_list.append(group_annotations[key])

        trackers_dict[track_file] = group_annotations

    return trackers_dict


def load_model(args: argparse.Namespace):
    # Model loading
    if args.model.split("_")[0] == 'resnet':
        model = ResNetWithEmbedder(resnet=args.model.split("_")[1], embed_size=args.embedding_size)
    elif args.model == 'vgg':
        model = VGG19()
    else:
        raise ValueError(f'Unknown model: {args.model}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(args.model_weights_path, map_location=torch.device(device)))
    model.eval()
    torch.set_grad_enabled(False)

    return model


def pre_compute_embeddings(trackers_dict: dict, args: argparse.Namespace):
    save_embeddings_path = os.path.join(args.output_path, "embeddings.pkl")
    # Check if embeddings have already been computed
    if os.path.exists(save_embeddings_path):
        # load embeddings from pkl file
        with open(save_embeddings_path, "rb") as input_file:
            embeddings_dict = pickle.load(input_file)
        print("Embeddings loaded from file.")
        return embeddings_dict

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224), antialias=True),
            # transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(
                mean=[0.4850, 0.4560, 0.4060],
                std=[0.2290, 0.2240, 0.2250]),
            # transforms.ToTensor(),
        ])

    model = load_model(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Compute and store the embeddings of every cam and track
    # Save it into a dict: key: cam, value: dict: key: track, value: embedding
    embeddings_dict = dict()
    for cam in tqdm(trackers_dict.keys()):
        for track in trackers_dict[cam].keys():
            ann = trackers_dict[cam][track][0]

            xtl, ytl, w, h = ann.coordinates_dim
            xtl, ytl, w, h = int(xtl), int(ytl), int(w), int(h)

            cam_name = cam.split('.')[0].split('_')[1]
            seq_name = cam.split('.')[0].split('_')[0]
            # cam_name = 'c010'  # TODO: remove this line

            # Open the video at the current frame
            video_path = os.path.join(args.dataset_path, seq_name.upper(), cam_name, 'vdo.avi')
            video = cv2.VideoCapture(video_path)
            video.set(cv2.CAP_PROP_POS_FRAMES, ann.frame)
            ret, frame = video.read()
            if not ret:
                raise ValueError(f'Could not read frame {ann.frame} from video {video_path}')

            # Crop the frame
            cropped_frame = frame[ytl:ytl + h, xtl:xtl + w]

            # Get embedding of the cropped frame by using the 'model'
            embedding = model(transform(
                torch.tensor(cropped_frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device))).squeeze()

            # Store the embedding
            if cam not in embeddings_dict.keys():
                embeddings_dict[cam] = dict()
            embeddings_dict[cam][track] = embedding

    # store the embeddings dict into a pkl file
    os.makedirs(os.path.dirname(save_embeddings_path), exist_ok=True)
    with open(save_embeddings_path, 'wb') as f:
        pickle.dump(embeddings_dict, f)

    return embeddings_dict


def store_trackers_dict(args, trackers_dict):
    # for each video, convert tracks to list of lists, and store them
    for file_name in trackers_dict.keys():
        # from dict to list of lists
        tracks_list = [trackers_dict[file_name][track] for track in trackers_dict[file_name].keys()]
        seq_name, cam_name = file_name.split('.')[0].split('_')

        save_tracking_path_data = "./data/trackers/mot_challenge/parabellum"+f"-{seq_name.lower()}"+"-train"
        # save_tracking_path = os.path.join(args.path_tracking_data+f"-{seq_name.lower()}", "metric_learning", "data", f"{seq_name.lower()}_{cam_name}" + ".txt")
        save_tracking_path = os.path.join(save_tracking_path_data, "metric_learning", "data", f"{seq_name.lower()}_{cam_name}" + ".txt")
        os.makedirs(os.path.dirname(save_tracking_path), exist_ok=True)

        store_trackers_list(tracks_list, save_tracking_path, file_mode="w")


def main(args: argparse.Namespace):

    trackers_dict = load_tracks(args)

    embeddings_dict = pre_compute_embeddings(trackers_dict, args)

    distances = []
    seen_cameras, seen_tracks = set(), set()
    for cam in trackers_dict.keys():
        for track in trackers_dict[cam].keys():
            for other_cam in trackers_dict.keys():
                if other_cam != cam and other_cam not in seen_cameras:
                    new_other_tracks = dict()
                    for other_track in trackers_dict[other_cam].keys():
                        if other_track not in seen_tracks:
                            # Get the embedding of the reference track representative
                            ref_embedding = embeddings_dict[cam][track]
                            # Get the embedding of the other track representative
                            other_embedding = embeddings_dict[other_cam][other_track]

                            # Compute the distance between the two embeddings
                            distance = torch.nn.CosineSimilarity(dim=0, eps=1e-08)(ref_embedding, other_embedding)
                            distances.append(distance)
                            # print(f"Distance between track {track} and track {other_track} is {distance}.")

                            # If the distance is smaller than a threshold, then the two tracks are assigned the same id
                            if distance < args.threshold:
                                new_other_tracks[track] = trackers_dict[other_cam][other_track]
                                for bbox in new_other_tracks[track]:
                                    bbox.track_id = track
                                embeddings_dict[other_cam][track] = embeddings_dict[other_cam][other_track]
                                print(
                                    f"Reassigned track id for ref. cam {cam}, ref id {track}; other cam {other_cam}, other id {other_track}!"
                                )
                                # # Add the other track to the ref track
                                # trackers_dict[other_cam][track] = trackers_dict[other_cam][other_track]
                                # del trackers_dict[other_cam][other_track]
                                # for bbox in trackers_dict[other_cam][track]:
                                #     bbox.track_id = track
                            else:
                                new_other_tracks[other_track] = trackers_dict[other_cam][other_track]
                        else:
                            new_other_tracks[other_track] = trackers_dict[other_cam][other_track]
                    trackers_dict[other_cam] = new_other_tracks

            seen_tracks.add(track)

        seen_cameras.add(cam)

    # store results
    store_trackers_dict(args, trackers_dict)

    # generate distances histogram

    # An "interface" to matplotlib.axes.Axes.hist() method
    n, bins, patches = plt.hist(x=distances, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.title('Distribution of distances between embeddings')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    # plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.savefig(os.path.join(args.output_path, "distances_histogram.png"))

    print("Done reassigning IDs !")


if __name__ == "__main__":
    args = __parse_args()
    main(args)
