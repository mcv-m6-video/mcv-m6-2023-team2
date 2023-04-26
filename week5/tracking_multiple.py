
import argparse
import os
import torch
import cv2
from tqdm import tqdm

from tracking.tracking_utils import store_trackers_list, viz_tracking

from metric_learning.models.resnet import ResNetWithEmbedder
from metric_learning.models.vgg import VGG19

from methods.annoyers import Annoyer

from utils import (
    load_config,
    load_predictions,
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
    parser.add_argument('--dataset_path', type=str, default='../data/AICity_data/train',
                        help='Path to the dataset.')
    parser.add_argument('--metric_dataset_path', type=str, default='./metric_learning_dataset',
                        help='Path to the metric learning dataset.')
    parser.add_argument('--detections_root', type=str, default='./detections/yolo',
                        help='Path to the detections root directory.')
    # Tracking data
    parser.add_argument('--path_tracking_data', type=str,
                        default="./data/trackers/mot_challenge/parabellum-train",
                        help='The path to the directory where the results will be stored.')
    parser.add_argument('--experiment', type=str, default='kalman_filtAreaFalse_filtParkedFalse',
                        help='Path to the results of single camera tracking.')
    # Metric learning config
    parser.add_argument('--model', type=str, default='resnet_18',
                        help='Model to use. Options: resnet_X, vgg.')
    parser.add_argument('--embedding_size', type=int, default=256,
                        help='Size of the embedding vector.')
    parser.add_argument('--model_weights_path', type=str,
                        default="./outputs_metric_learning/models/resnet_18_aic19_loss_triplet_miner_TripletMargin_distance_cosine/model_final.pth",
                        help='Path to the model weights.')
    parser.add_argument('--n_neighbors', type=int, default=50,
                        help='Number of nearest neighbors to retrieve.')

    args = parser.parse_args()
    return args


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, img_list, augmentations):
        super(MyDataset, self).__init__()
        self.img_list = img_list
        self.augmentations = augmentations

    def load_image(self, idx):
        img = cv2.imread(self.img_list[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1)
        img = img / 255.0
        return img

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = self.load_image(idx)
        return self.augmentations(img), self.img_list[idx]


def main(args: argparse.Namespace):

    # Group images by frame
    images_dict = dict()  # key: frame number, value: list of images
    for instance in os.listdir(args.metric_dataset_path):
        for frame in os.listdir(os.path.join(args.metric_dataset_path, instance)):
            if frame.endswith('.jpg'):
                # Get the frame number from the file name 'frame_c037_0064.jpg'
                frame_number = int(frame.split('_')[-1].split('.')[0])
                if frame_number not in images_dict:
                    images_dict[frame_number] = []
                images_dict[frame_number].append(os.path.join(args.metric_dataset_path, instance, frame))
    print(f"Created {len(images_dict)} groups of images by frame")

    # Scan all sequences in directory 'detections_root'
    # Group detections by frame
    all_seqs_detections = dict(dict())  # key: frame number, value: key: camera name, value: list of detections
    for seq in os.listdir(args.detections_root):
        seq_name = os.path.basename(seq)
        # Scan all cameras in directory cfg["detections_dir"]/seq
        for camera in os.listdir(os.path.join(args.detections_root, seq)):
            camera_name = os.path.basename(camera)
            # Load detections

            detections_path = f"{os.path.join(args.detections_root, seq, camera)}/detections.txt"

            # Load and process detections
            confidence_threshold = 0.6
            detections = load_predictions(detections_path)
            detections = filter_annotations(detections, confidence_thr=confidence_threshold)
            detections = group_annotations_by_frame(detections)
            detections = non_maxima_suppression(detections)
            # detections_type = args.detections_root.split("/")[-1]

            seq_cam_name = f'{seq_name.lower()}_{camera_name}'

            # For every frame in the sequence, add a dictionary {seq_cam_name: detections} for each frame
            for idx in range(len(detections)):
                if idx not in all_seqs_detections:
                    all_seqs_detections[idx] = dict()
                if seq_cam_name not in all_seqs_detections[idx]:
                    all_seqs_detections[idx][seq_cam_name] = []
                some_detections = detections[idx]
                if some_detections:
                    all_seqs_detections[idx][seq_cam_name].append(some_detections[0])
    print(f"Created {len(all_seqs_detections)} groups of detections by frame")

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

    # Iter over all frames
    for idx_frame in tqdm(range(len(all_seqs_detections))):
        # Get the detections for the current frame
        detections = all_seqs_detections[idx_frame]

        # Get the images for the current frame
        try:
            images = images_dict[idx_frame]
        except KeyError:
            continue

        # Build custom dataset for the current frame and dataloader
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224), antialias=True),
                # transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(
                    mean=[0.4850, 0.4560, 0.4060],
                    std=[0.2290, 0.2240, 0.2250]),
                # transforms.ToTensor(),
            ])
        if isinstance(images, str):
            pass
        dataset = MyDataset(images, augmentations=transform)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        with torch.no_grad():
            model = model.to(device)

            # Annoyer
            annoy = Annoyer(model, dataloader, emb_size=args.embedding_size,
                            device=device, distance='angular', experiment_name=f"frame_{idx_frame}")
            try:
                annoy.load()
            except:
                annoy.state_variables['built'] = False
                annoy.fit()

            # Iterate over all detections
            for cam, cam_detections in detections.items():
                if cam_detections:

                    for detection in cam_detections:
                        xtl, ytl, w, h = detection.coordinates_dim
                        xtl, ytl, w, h = int(xtl), int(ytl), int(w), int(h)

                        # Open the video at the current frame
                        video_path = os.path.join(args.dataset_path, cam.split("_")[0].upper(), cam.split("_")[1], "vdo.avi")
                        video = cv2.VideoCapture(video_path)
                        video.set(cv2.CAP_PROP_POS_FRAMES, idx_frame)
                        ret, frame = video.read()
                        if not ret:
                            raise ValueError(f'Could not read frame {idx_frame} from video {video_path}')

                        # Crop the frame
                        cropped_frame = frame[ytl:ytl + h, xtl:xtl + w]

                        # Get embedding of the cropped frame by using the 'model'
                        embedding = model(transform(torch.tensor(cropped_frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device))).squeeze()

                        nns, distances = annoy.retrieve_by_vector(
                            embedding, n=1, include_distances=True)

                        # Get the path of the image that is the nearest neighbor
                        try:
                            _, nn_path = dataset[nns[0]]
                        except:
                            pass

                        # Get the seq and cam name of the nearest neighbor
                        seq_with_id = os.path.basename(os.path.dirname(nn_path))
                        seq, id = seq_with_id.split("_")
                        cam = os.path.basename(nn_path).split("_")[1]

                        # Assign ID to the detection
                        detection.track_id = id

                    save_tracking_path = os.path.join(args.path_tracking_data, "metric_learning", "data",
                                                      f"{seq.lower()}_{cam}" + ".txt")
                    os.makedirs(os.path.dirname(save_tracking_path), exist_ok=True)
                    store_trackers_list([cam_detections], save_tracking_path)


if __name__ == "__main__":
    args = __parse_args()
    main(args)
