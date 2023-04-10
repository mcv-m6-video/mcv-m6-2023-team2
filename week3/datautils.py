from pathlib import Path
from typing import Dict, Tuple
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import os
import json
from tqdm import tqdm

import json
from detectron2.data import MetadataCatalog, DatasetCatalog
import random

CocoDict = Dict
Height, Width, Channels = int, int, int

def generate_gt_from_txt(in_path):
    pass

def generate_gt_from_xml(in_path: Path, ignore_parked: bool = False, ignore_classes: bool = False) -> CocoDict:

    ## Thanks for the parser dear pau torras 
    ### Your effort will be remembered through de eons 
    ### https://github.com/mcv-m6-video/mcv-m6-2022-team1/blob/main/w2/data.py


    dataset = ET.parse(str(in_path)).getroot()

    # Build coco-compliant dataset in JSON format
    if ignore_classes:
        labels = {
            "moving": 1,
        }
        last_label = 1
    else:
        labels = {}
        last_label = -1

    # FIXME: Hardcoded, but necessary to ensure all images appear on the gt
    frames = set([ii for ii in range(1, 2142)])
    ann_id = 0

    # Create the annotations field
    annotations = []
    for track in dataset.findall("track"):
        if ignore_classes:
            obj_label = 1
        else:
            if track.attrib["label"] not in labels:
                last_label += 1
                labels[track.attrib["label"]] = last_label
            obj_label = labels[track.attrib["label"]]

        for num, box in enumerate(track.findall("box")):
            if ignore_parked and track.attrib["label"] == "car":
                continue

            # Keep track of images with annotations
            frame = int(box.attrib["frame"]) + 1
            frames.add(frame)

            # Generate a bounding box
            bbox = [
                float(box.attrib["xtl"]),
                float(box.attrib["ytl"]),
                float(box.attrib["xbr"]) - float(box.attrib["xtl"]),
                float(box.attrib["ybr"]) - float(box.attrib["ytl"]),
            ]

            annotations.append({
                "id": ann_id,
                "image_id": frame,
                "category_id": obj_label,
                "bbox": bbox,
                "segmentation": [],
                "keypoints": [],
                "num_keypoints": 0,
                "score": 1,
                "area": bbox[-2] * bbox[-1],
                "iscrowd": 0
            })
            ann_id += 1

    # Create the images field
    images = []
    for ii in frames:
        images.append({
            "id": ii,
            "license": 1,
            "file_name": f"frames/{ii:05}.jpg",
            "height": 1080,
            "width": 1920,
            "date_captured": None,
        })

    # Create the categories field
    categories = []
    for name, cat_id in labels.items():
        categories.append({
            "id": cat_id,
            "name": name,
            "supercategory": "vehicle",
            "keypoints": [],
            "skeleton": [],
        })
    licenses = {
        "id": 1,
        "name": "Unknown",
        "url": "Unknown",
    }
    info = {
        "year": 2022,
        "version": "0.0",
        "description": "Hopefully I did not screw it up this time",
        "contributor": "Nobody",
        "url": "None",
    }

    coco_dict = {
        "info": info,
        "images": images,
        "categories": categories,
        "annotations": annotations,
        "licenses": licenses
    }
    return coco_dict

def unvideo_video(video, frames_folder = './frames/'):
    os.makedirs(frames_folder, exist_ok=True)

    cv2_vid = cv2.VideoCapture(video)
    num_frames = int(cv2_vid.get(cv2.CAP_PROP_FRAME_COUNT))
    print(num_frames)

    for frame_id in tqdm(range(num_frames)):
        _, frame = cv2_vid.read()
        cv2.imwrite(os.path.join(frames_folder, f"{(frame_id +1):05}.jpg"), frame)

    return True

def load_jsons(coco_dictionary):

    #### first 25% training #####
    start_validation_idx = int(len(coco_dictionary['images']) * .25)
    train, val = list(), list()
    using = train
    everything = []

    for idx, image in enumerate(coco_dictionary['images']):

        gt = [{**x, 'bbox_mode': 1} for x in coco_dictionary['annotations'] if image['id'] == x['image_id']]
        
        if idx > start_validation_idx: using = val
        using.append({**image, 'image_id': image['id'], 'annotations': gt})
        everything.append({**image, 'image_id': image['id'], 'annotations': gt})
    open('train_first.json', 'w').write(json.dumps(train))
    open('everything.json', 'w').write(json.dumps(everything))
    open('val_first.json', 'w').write(json.dumps(val))


    train, val = list(), list()
    using = train

    for idx, image in enumerate(coco_dictionary['images']):

        gt = [{**x, 'bbox_mode': 1} for x in coco_dictionary['annotations'] if image['id'] == x['image_id']]
        
        if random.random() > .25: using = val
        else: using = train
        using.append({**image, 'image_id': image['id'], 'annotations': gt})
    
    random.shuffle(train)
    random.shuffle(val)
    open('train_random.json', 'w').write(json.dumps(train))
    open('val_random.json', 'w').write(json.dumps(val))

    ### CROSS VALIDATION ####
    fold_1, fold_2, fold_3 = list(), list(), list()
    start_fold_2 = int(0.33 * len(coco_dictionary['images']))
    start_fold_3 = int(0.66 * len(coco_dictionary['images']))
    using = fold_1

    for idx, image in enumerate(coco_dictionary['images']):

        gt = [{**x, 'bbox_mode': 1} for x in coco_dictionary['annotations'] if image['id'] == x['image_id']]
        
        if idx > start_fold_3: using = fold_3
        elif idx > start_fold_2: using = fold_2
        else: using = fold_1

        using.append({**image, 'image_id': image['id'], 'annotations': gt})

    open('fold1.json', 'w').write(json.dumps(fold_1))
    open('fold2.json', 'w').write(json.dumps(fold_2))
    open('fold3.json', 'w').write(json.dumps(fold_3))

    open('fold1+3.json', 'w').write(json.dumps(fold_1 + fold_3))
    open('fold1+2.json', 'w').write(json.dumps(fold_1 + fold_2))
    open('fold3+2.json', 'w').write(json.dumps(fold_3 + fold_2))

def load_first_data(t="train"):
    if t == "train":
        with open("train_first.json", 'r') as file:
            train = json.load(file)
        return train
    elif t == "val":
      with open("val_first.json", 'r') as file:
          val = json.load(file)
    return val

def load_random_data(t="train"):
    if t == "train":
        with open("train_random.json", 'r') as file:
            train = json.load(file)
        return train
    elif t == "val":
      with open("val_random.json", 'r') as file:
          val = json.load(file)
    return val

def load_xox(t="train"):
    if t == "train":
        with open("./datafolds/fold2.json", 'r') as file:
            train = json.load(file)
        return train
    elif t == "val":
      with open("./datafolds/fold1+3.json", 'r') as file:
          val = json.load(file)
    return val

def load_xxo(t="train"):
    if t == "train":
        with open("./datafolds/fold3.json", 'r') as file:
            train = json.load(file)
        return train
    elif t == "val":
      with open("./datafolds/fold1+2.json", 'r') as file:
          val = json.load(file)
    return val

def load_oxx(t="train"):
    if t == "train":
        with open("./datafolds/fold1.json", 'r') as file:
            train = json.load(file)
        return train
    elif t == "val":
      with open("./datafolds/fold3+2.json", 'r') as file:
          val = json.load(file)
    return val

if __name__ == '__main__':
    #unvideo_video('/home/adria/Desktop/mcv-m6-2023-team2/data/AICity_S03_c010/vdo.avi')
    a = (generate_gt_from_xml('../data/AICity_S03_c010/ai_challenge_s03_c010-full_annotation.xml'))
    load_jsons(a)
    print(a['categories'])