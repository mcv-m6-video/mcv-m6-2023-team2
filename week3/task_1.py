import os
import cv2
import numpy as np
from PIL import Image
import requests
import matplotlib.pyplot as plt

import torch
torch.set_grad_enabled(False)
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T

from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from tqdm import tqdm


COCO_CAR_ID = 3  # COCO class id for car -1.
VALID_IDS = [COCO_CAR_ID-1]

VALID_IDS_DETR = [COCO_CAR_ID]


def run_inference_detectron(args):

    MODELS = {
        'retina': 'retinanet_R_50_FPN_3x',
        'faster': 'faster_rcnn_R_50_FPN_3x',
    }

    model_path = 'COCO-Detection/' + MODELS[args.model] + '.yaml'
    print(model_path)

    # Run a pre-trained detectron2 model
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(model_path))
    if args.model == 'retina':
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.RETINANET.NMS_THRESH_TEST = 0.4
    elif args.model == 'faster':
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_path)
    cfg.OUTPUT_DIR = os.path.join(args.path_results, args.model)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    predictor = DefaultPredictor(cfg)

    res_path = os.path.join(cfg.OUTPUT_DIR, 'detections.txt')
    if os.path.exists(res_path):
        os.remove(res_path)

    cv2_vid = cv2.VideoCapture(args.path_video)
    num_frames = int(cv2_vid.get(cv2.CAP_PROP_FRAME_COUNT))
    num_frames = 10

    # keep track of time (s/img) to run inferece
    timestamps = []
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    f = open(res_path, 'a')
    for frame_id in tqdm(range(num_frames)):
        _, frame = cv2_vid.read()

        # record inference time
        begin.record()
        model_preds = predictor(frame)
        end.record()

        torch.cuda.synchronize()
        timestamps.append(begin.elapsed_time(end))

        instances = model_preds["instances"]

        filtered_instances = instances[instances.pred_classes == COCO_CAR_ID] # or instances.pred_classes == 7
        bboxes = filtered_instances.pred_boxes.to("cpu")
        confs = filtered_instances.scores.to("cpu")
        classes = filtered_instances.pred_classes.to("cpu")
        # confident_detections = instances[instances.scores > 0.9]

        # discard predictions corresponding to classes not in VALID_IDS
        for i, prediction in enumerate(classes):
            # TODO: also allow predicting trucks (because pick-up trucks are also cars, but in COCO they are considered trucks)
            box = bboxes[i].tensor.numpy()[0]

            # Store in AI City Format:
            # <frame> <id> <bb_left> <bb_top> <bb_width> <bb_height> <conf> <x> <y> <z>
            det = str(frame_id+1)+',-1,'+str(box[0])+','+str(box[1])+','+str(box[2]-box[0])+','+str(box[3]-box[1])+','+str(confs[i].item())+',-1,-1,-1\n'
            f.write(det)

        if args.store_results:
            output_path = os.path.join(cfg.OUTPUT_DIR, 'det_frame_' + str(frame_id) + '.png')
            v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
            out = v.draw_instance_predictions(filtered_instances.to("cpu"))
            cv2.imwrite(output_path, out.get_image()[:, :, ::-1])

    f.close()

    print('Inference time (s/img): ', np.mean(timestamps)/1000)

    return res_path


# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.ToPILImage(),
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def plot_results(pil_img, prob, boxes, output_path):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes, colors):
        print("xmin: ", xmin, "ymin: ", ymin, "xmax: ", xmax, "ymax: ", ymax)
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.savefig(output_path)


def run_inference_detr(args):

    res_dir = os.path.join(args.path_results, args.model)
    os.makedirs(res_dir, exist_ok=True)

    res_path = os.path.join(res_dir, 'detections.txt')
    if os.path.exists(res_path):
        os.remove(res_path)

    model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
    model.eval()

    cv2_vid = cv2.VideoCapture(args.path_video)
    num_frames = min( int(cv2_vid.get(cv2.CAP_PROP_FRAME_COUNT)), args.num_frames )

    # keep track of time (s/img) to run inferece
    timestamps = []
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    f = open(res_path, 'a')
    for frame_id in tqdm(range(num_frames)):
        _, frame_orig = cv2_vid.read()
        # frame_orig = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB)
        url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
        frame_orig = Image.open(requests.get(url, stream=True).raw)
        # print("Before transform: ", frame_orig.min(), frame_orig.max(), frame_orig.mean(), frame_orig.std(), frame_orig.shape)
        frame = transform(frame_orig).unsqueeze(0)
        # print("After transform: ", frame.min(), frame.max(), frame.mean(), frame.std(), frame.shape)

        # record inference time
        begin.record()
        model_preds = model(frame)
        end.record()

        torch.cuda.synchronize()
        timestamps.append(begin.elapsed_time(end))

        confs = model_preds['pred_logits'].softmax(-1)[0, :, :-1]
        # TODO: when visualizing results, keep only predictions with 0.7+ confidence
        # keep = confs.max(-1).values > 0.9

        # convert boxes from [0; 1] to image scales

        bboxes = rescale_bboxes(model_preds['pred_boxes'][0, ...], frame_orig.size)#shape[:2])
        # bboxes_scaled = rescale_bboxes(model_preds['pred_boxes'][0, keep], frame.size)

        # classes_idxs = []
        # confs_filt, bboxes_filt = [], []
        # for i, p, in enumerate(confs):
        #     cl = p.argmax()
        #     if cl in VALID_IDS_DETR:
        #         print(f'{CLASSES[cl]}: {p[cl]:0.2f}')
        #         classes_idxs.append((i, cl))

        #         # TODO: also allow predicting trucks (because pick-up trucks are also cars, but in COCO they are considered trucks)
        #         box = bboxes.numpy()[i]

        #         bboxes_filt.append(box)
        #         confs_filt.append(confs[i])

        #         # Store in AI City Format:
        #         # <frame> <id> <bb_left> <bb_top> <bb_width> <bb_height> <conf> <x> <y> <z>
        #         det = str(frame_id+1)+',-1,'+str(box[0])+','+str(box[1])+','+str(box[2]-box[0])+','+str(box[3]-box[1])+','+str(confs[i][cl].item())+',-1,-1,-1\n'
        #         f.write(det)

        if args.store_results:
            output_path = os.path.join(res_dir, 'det_frame_' + str(frame_id) + '.png')
            # plot_results(frame.squeeze().permute(1, 2, 0)[..., (2,1,0)], confs_filt, bboxes_filt, output_path)
            plot_results(frame_orig, confs, bboxes, output_path)

    f.close()

    print('Inference time (s/img): ', np.mean(timestamps)/1000)

    return res_path


def viz_detr_att(args, model, img,):
    # use lists to store the outputs via up-values
    conv_features, enc_attn_weights, dec_attn_weights = [], [], []

    hooks = [
        model.backbone[-2].register_forward_hook(
            lambda self, input, output: conv_features.append(output)
        ),
        model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
            lambda self, input, output: enc_attn_weights.append(output[1])
        ),
        model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
            lambda self, input, output: dec_attn_weights.append(output[1])
        ),
    ]

    # propagate through the model
    outputs = model(img)

    for hook in hooks:
        hook.remove()

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.9

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

    # don't need the list anymore
    conv_features = conv_features[0]
    enc_attn_weights = enc_attn_weights[0]
    dec_attn_weights = dec_attn_weights[0]

    # get the feature map shape
    h, w = conv_features['0'].tensors.shape[-2:]

    fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=2, figsize=(22, 7))
    colors = COLORS * 100
    for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T, bboxes_scaled):
        ax = ax_i[0]
        ax.imshow(dec_attn_weights[0, idx].view(h, w))
        ax.axis('off')
        ax.set_title(f'query id: {idx.item()}')
        ax = ax_i[1]
        ax.imshow(im)
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                fill=False, color='blue', linewidth=3))
        ax.axis('off')
        ax.set_title(CLASSES[probas[idx].argmax()])
    fig.tight_layout()
    # store figure in output directory
    plt.savefig(os.path.join(args.output_dir, 'detr_att.png'))


def task_1_1(args):

    if args.model in ['retina', 'faster']:
        res_path = run_inference_detectron(args)
    elif args.model.lower() == 'detr':
        res_path = run_inference_detr(args)

    print("Results saved in: ", res_path)
