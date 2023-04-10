from collections import defaultdict, OrderedDict
from copy import deepcopy
import math
from typing import List, Dict
import cv2
import xmltodict
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from class_utils import BoundingBox
from metrics import voc_eval
import os

import imageio
import optuna


def iou_over_time(
        video_path: str,
        annotations: List[BoundingBox],
        predictions: List[BoundingBox],
        show_video: bool = False,
        max_frames: int = 9999,
        frame_sampling_each: int = 4,
        save_plots: bool = True,
        save_path: str = "week1/results/",
) -> float:
    """
    Shows the given annotations and predictions in the given video and returns the mean IoU.

    :param video_path: Path to the video.
    :param annotations: List of annotations.
    :param predictions: List of predictions.
    :param show_video: If True, the video will be shown.
    :param max_frames: Maximum number of frames to show and process.
    :param frame_sampling_each: Process every n-th frame.
    :param save_plots: If True, the plots will be saved.
    :param save_path: Path to save the plots, and both plot and video GIFs.
    """
    grouped_annotations = group_annotations_by_frame(annotations)
    grouped_predictions = group_annotations_by_frame(predictions)

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    miou = []
    frames = []
    max_frames = min(max_frames, total_frames)
    steps = []

    for idx_frame in range(0, max_frames, frame_sampling_each):
        video.set(cv2.CAP_PROP_POS_FRAMES, idx_frame)
        ret, frame = video.read()
        if not ret:
            break

        for box in annotations:
            if box.frame == idx_frame:
                cv2.rectangle(frame, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), (0, 255, 0), 2)

        for box in predictions:
            if box.frame == idx_frame:
                cv2.rectangle(frame, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), (0, 0, 255), 2)

        # progress bar
        cv2.rectangle(frame, (0, height - 25), (int(idx_frame * (width / max_frames)), width), (0, 255, 0), -1)

        if show_video:
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (int(width/10), int(height/10)))
        frames.append(frame)
        steps.append(idx_frame)

        _, _, _, iou = voc_eval([grouped_annotations[idx_frame]], [grouped_predictions[idx_frame]])
        miou.append(iou)

        if save_plots:

            run_name = "MY_RUN"
            plots_folder = os.path.join(save_path, run_name, str(max_frames))
            os.makedirs(plots_folder, exist_ok=True)
            os.makedirs(plots_folder+"/static", exist_ok=True)

            plt.plot(steps, miou, c="red")
            plt.xlabel('Frame number')
            plt.ylabel('Mean IoU')
            plt.locator_params(axis='y', nbins=11)
            plt.xlim([0, max_frames])
            plt.ylim([0, 1])
            plt.grid(visible=True)
            plt.title("Mean IoU over time")
            plt.savefig(os.path.join(plots_folder, f"miou_plot_{idx_frame}.png"))

            if idx_frame == max_frames-1:
                plt.savefig(os.path.join(plots_folder, f"static/miou_plot_{idx_frame}.png"))

    video.release()
    cv2.destroyAllWindows()

    plot_files = [x for x in os.listdir(plots_folder) if x.endswith(".png")]
    plot_images = []
    for filename in sorted(plot_files, key=lambda x: int(x.split('_')[-1].split('.')[0])):
        full_filename = os.path.join(plots_folder, filename)
        plot_images.append(cv2.imread(full_filename))
        os.remove(full_filename)
    imageio.mimsave(os.path.join(plots_folder, "miou.gif"), plot_images, duration=0.05)

    print(f"Saving GIF...")
    imageio.mimsave(os.path.join(plots_folder, f"video_small_{max_frames}.gif"), frames, duration=0.05)
    print(f"GIF saved at {save_path}")

    _, _, _, mean_miou = voc_eval(grouped_annotations, grouped_predictions)
    with open(os.path.join(plots_folder, 'mean_iou.txt'), 'w') as f:
        f.write(str(mean_miou))

    return mean_miou


def group_by_frame(boxes):
    grouped = defaultdict(list)
    for box in boxes:
        grouped[box.frame].append(box)
    return OrderedDict(sorted(grouped.items()))


def group_annotations_by_frame(annotations: List[BoundingBox]) -> List[List[BoundingBox]]:
    """
    Groups the given list of annotations by frame.

    Parameters:
    annotations (list): List of annotations to group by frame.

    Returns:
    A list of lists of annotations grouped by frame.
    """
    grouped_annotations = []

    for box in annotations:
        if len(grouped_annotations) <= box.frame:
            for _ in range(box.frame - len(grouped_annotations) + 1):
                grouped_annotations.append([])

        grouped_annotations[box.frame].append(box)

    return grouped_annotations


def load_annotations(
    xml_file_path: str,
    select_label_types: List = None,
    grouped: bool = False,
    use_parked: bool = False,
    ) -> List[BoundingBox]:
    """
    Loads the annotations from the given XML file.
    """
    with open(xml_file_path) as f:
        annotations = xmltodict.parse(f.read())

    tracks = annotations['annotations']['track']
    bboxes = []
    for track in tracks:
        if select_label_types and track['@label'] not in select_label_types:
            continue

        for box in track['box']:
            is_parked = 'attribute' in box and box['attribute']['#text'] == 'true'
            
            if not use_parked and is_parked:
                continue

            bboxes.append(BoundingBox(
                x1=float(box['@xtl']),
                y1=float(box['@ytl']),
                x2=float(box['@xbr']),
                y2=float(box['@ybr']),
                frame=int(box['@frame']),
                track_id=int(track['@id']),
                label=track['@label'],
                parked=is_parked,
            ))

    if grouped:
        return group_annotations_by_frame(bboxes)

    return bboxes


def load_predictions(csv_file_path: str, grouped: bool = False) -> List[BoundingBox]:
    """
    Loads the predictions from the given CSV file.

    Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    We checked the format in https://github.com/mcv-m6-video/mcv-m6-2021-team4/blob/main/W1/aicity_reader.py
    Also, solved the frame-1 issue :)
    """
    with open(csv_file_path) as f:
        lines = f.readlines()

    bboxes = []

    for line in lines:
        frame, track_id, xtl, ytl, width, height, confidence, _, _, _ = line.split(',')
        xbr = float(xtl) + float(width)
        ybr = float(ytl) + float(height)
        bboxes.append(BoundingBox(
            x1=float(xtl),
            y1=float(ytl),
            x2=xbr,
            y2=ybr,
            frame=int(frame)-1,
            track_id=int(track_id),
            label='car',
            parked=False,
            confidence=float(confidence),
        ))

    if grouped:
        return group_annotations_by_frame(bboxes)

    return bboxes


def filter_annotations(annotations: List[BoundingBox], confidence_thr: float = 0.0) -> List[BoundingBox]:
    return [x for x in annotations if x.confidence >= confidence_thr]


def convert_optical_flow_to_image(flow: np.ndarray) -> np.ndarray:
    # The 3-channel uint16 PNG images that comprise optical flow maps contain information
    # on the u-component in the first channel, the v-component in the second channel,
    # and whether a valid ground truth optical flow value exists for a given pixel in the third channel.
    # A value of 1 in the third channel indicates the existence of a valid optical flow value
    # while a value of 0 indicates otherwise. To convert the u- and v-flow values from
    # their original uint16 format to floating point values, one can do so by subtracting 2^15 from the value,
    # converting it to float, and then dividing the result by 64.

    img_u = (flow[:, :, 2] - 2 ** 15) / 64
    img_v = (flow[:, :, 1] - 2 ** 15) / 64

    img_available = flow[:, :, 0]  # whether a valid GT optical flow value is available
    img_available[img_available > 1] = 1

    img_u[img_available == 0] = 0
    img_v[img_available == 0] = 0

    optical_flow = np.dstack((img_u, img_v, img_available))
    return optical_flow


def convert_image_to_optical_flow(flow: np.ndarray) -> np.ndarray:
    img_u = flow[:, :, 0]
    img_v = flow[:, :, 1]
    img_available = flow[:, :, 2]

    img_u = (img_u * 64 + 2 ** 15).astype(np.uint16)
    img_v = (img_v * 64 + 2 ** 15).astype(np.uint16)
    img_available = img_available.astype(np.uint16)

    optical_flow = np.dstack((img_available, img_v, img_u))
    return optical_flow


def load_optical_flow(file_path: str):
    # channels arranged as BGR
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED).astype(np.double)
    return convert_optical_flow_to_image(img)


def read_flow_unimatch(fn):
    """
    Read .flo file in Middlebury format
    """
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
            # Reshape testdata into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))


def histogram_error_distribution(error, GT):
    max_range = int(math.ceil(np.amax(error)))

    plt.title('Mean square error distribution')
    plt.ylabel('Density')
    plt.xlabel('Mean square error')
    plt.hist(error[GT[2] == 1].ravel(), bins=30, range=(0.0, max_range))


def get_experiment_path(args):
    rho_str = f"_rho_{args['rho']}" if args['bg_model'] == 'adaptive' else ''
    experiment_path = os.path.join(
        args['path_results'],
        f"bg_{args['bg_model']}",
        f"vote_{args['voting']}",
        f"color_{args['color_space']}",
        f"alpha_{args['alpha']}{rho_str}",
    )
    return experiment_path


def save_metrics(args, recall, precision, F1, AP, IoU, filename="metrics.txt"):
    experiment_path = os.path.join(get_experiment_path(args), filename)
    with open(experiment_path, 'w') as f:
        f.write(f"Recall: {recall[-1]}\nPrecision: {precision[-1]}\nF1: {F1[-1]}\nAP: {AP}\nIoU: {IoU}")


def create_gif(args, subfolder, extension, duration=0.05, remove_images=False):
    experiment_path = os.path.join(get_experiment_path(args), subfolder)
    if not os.path.exists(experiment_path):
        return
    files = [x for x in os.listdir(experiment_path) if x.endswith(extension)]
    images = []
    for filename in sorted(files, key=lambda x: int(os.path.basename(x).replace(extension, ''))):
        full_filename = os.path.join(experiment_path, filename)
        images.append(cv2.cvtColor(cv2.imread(full_filename), cv2.COLOR_BGR2RGB))
        if remove_images:
            os.remove(full_filename)
    imageio.mimsave(os.path.join(experiment_path, "video.gif"), images, duration=duration)


def save_image(img, name, args, subfolder=None, extension='.bmp'):
    experiment_path = get_experiment_path(args)
    if subfolder is not None:
        experiment_path = os.path.join(experiment_path, subfolder)
    os.makedirs(experiment_path, exist_ok=True)
    filename = f"{str(name)}{extension}"
    cv2.imwrite(os.path.join(experiment_path, filename), img)


def draw_legend(image, labels=None, colors=None, linewidth=2):
    list_colors = {
        'r': (0, 0, 255),
        'g': (0, 255, 0),
        'b': (255, 0, 0),
        'w': (255, 255, 255),
    }
    cv2.putText(image, labels[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, list_colors[colors[0]], linewidth)
    cv2.putText(image, labels[1], (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, list_colors[colors[1]], linewidth)
    return image


def draw_boxes(image, boxes, tracker=None, color='g', linewidth=2, det=False, boxIds=False, old=False):
    colors = {
        'r': (0, 0, 255),
        'g': (0, 255, 0),
        'b': (255, 0, 0),
        'w': (255, 255, 255),
    }
    color_ids = {}

    rgb = colors[color]
    for box in boxes:
        # print('box.track_id: ', box.track_id)
        if boxIds:
            if box.track_id in list(color_ids.keys()):
                pass
            else:
                color_ids[box.track_id] = np.random.uniform(0, 256, size=3)
            if old:
                cv2.putText(image, str(box.track_id), (int(box.x1), int(box.y1) + 120), cv2.FONT_ITALIC, 0.6,
                            color_ids[box.track_id], linewidth)
            else:
                cv2.putText(image, str(box.track_id), (int(box.x1), int(box.y1) + 20), cv2.FONT_ITALIC, 0.6,
                            color_ids[box.track_id], linewidth)

            if tracker is not None:
                if box.track_id in tracker:
                    if len(tracker[box.track_id]) > 2:
                        image = cv2.polylines(image, [np.array(tracker[box.track_id])], False, color_ids[box.track_id], linewidth)

            # if len(kalman_predictions[box.id])>2:
            #     image =cv2.polylines(image,[np.array(kalman_predictions[box.id])],False,color_ids[box.id],linewidth)

            image = cv2.rectangle(image, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), color_ids[box.track_id],
                                  linewidth)
        else:
            image = cv2.rectangle(image, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), rgb, linewidth)

        if det:
            cv2.putText(image, str(box.confidence), (int(box.x1), int(box.y1) - 5), cv2.FONT_ITALIC, 0.6, rgb,
                        linewidth)

    return image


def plot_3d_surface(args, study: optuna.study.Study, metric: str = 'F1', interactive=False):
    mapping = {
        'recall': 'values_0',
        'precision': 'values_1',
        'F1': 'values_2',
        'AP': 'values_3',
        'IoU': 'values_4',
    }
    df = study.trials_dataframe()
    x = df['params_alpha'].values
    y = df['params_rho'].values
    z = df[mapping[metric]].values

    # prepare the interpolator
    triang = tri.Triangulation(x, y)
    interpolator = tri.LinearTriInterpolator(triang, z)

    X, Y = np.meshgrid(x, y)
    Z = interpolator(X, Y)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    plt.xlabel('alpha')
    plt.ylabel('rho')
    plt.title('recall')
    plt.legend()
    plt.savefig(os.path.join(args.path_results, f'optuna-{args.optuna_study_name}-{metric}.png'))

    if interactive:
        mpl.use('macosx')
        plt.show()


def generate_test_subset(data, N_frames=2141, test_p=0.75):
    init_frame = int((1-test_p) * N_frames)
    test_data = []
    for annotation in data:
        if annotation.frame >= init_frame:
            test_data.append(annotation)

    return test_data


def non_maxima_suppression(bboxes_per_frame: List[List[BoundingBox]], iou_threshold: float = 0.7) -> List[BoundingBox]:
    """
    Perform Non Maxima Suppression (NMS) on a list of bounding boxes.

    :param bboxes: a list of BoundingBox objects per frame.
    :param iou_threshold: the IoU threshold for overlapping bounding boxes.
    :return: a list of selected BoundingBox objects after NMS.
    """
    new_bboxes_per_frame = []

    for bboxes in bboxes_per_frame:
        # Sort the bounding boxes by decreasing confidence scores.
        bboxes_sorted = sorted(bboxes, key=lambda bbox: bbox.confidence or 0, reverse=True)

        selected_bboxes = []

        while bboxes_sorted:
            # Select the bounding box with the highest confidence score.
            bbox = bboxes_sorted[0]
            selected_bboxes.append(bbox)

            # Remove the selected bounding box from the list.
            bboxes_sorted = bboxes_sorted[1:]

            # Compute the IoU between the selected bounding box and the remaining bounding boxes.
            ious = [bbox.IoU(other) for other in bboxes_sorted]

            # Remove the bounding boxes with IoU > threshold.
            bboxes_sorted = [b for i, b in enumerate(bboxes_sorted) if ious[i] <= iou_threshold]

        new_bboxes_per_frame.append(selected_bboxes)

    return new_bboxes_per_frame


def resize_image_keep_aspect_ratio(image, max_size, interpolation=cv2.INTER_LINEAR):
    """
    Resize an image to a maximum size while keeping the aspect ratio.

    :param image: the image to resize.
    :param max_size: the maximum size of the image.
    :param interpolation: the interpolation method.
    :return: the resized image.
    """
    h, w = image.shape[:2]
    
    if h > w:
        new_h, new_w = max_size, int(max_size * w / h)
    else:
        new_h, new_w = int(max_size * h / w), max_size

    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)