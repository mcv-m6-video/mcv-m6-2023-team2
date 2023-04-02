# Week 3


## Task 1: Object Detection
+ Inference with off-the-shelf methods
   + ...
+ Data annotation. Annotations can be found in `week3/task1_2_annotations/`. We include `Pascal VOC` and `KITTI` format annotations.
+ Fine-tune pretrained models with our own data
+ K-Fold Cross-Validation
   + Strategy A - First 25% for training and second 75% for test.
   + Strategy B - K=4 fold cross-validation. Fixed folds. Split sequence into 4 folds (25% each). One fold is the same as strategy A.
   + Strategy C - K-Fold cross-validation (use K=4). Random 25% Train - rest for Test


## Task 2: Object Tracking
+ Tracking by Maximum Overlap
+ Tracking with a Kalman Filter
+ IDF1, HOTA scores


## Task 3: CVPR 2021 AI City Challenge
+ Asses quality of best solution from previous tasks on the CVPR 2022 AI City Challenge.


## Execution
 
To execute each task, simply run with:

```bash
python main.py -h
```

For task 1.1, run:

```bash
python main.py -t1_1
```

Note: Add AIcity folder into data/ to use it. Clone [this repo](https://github.com/abewley/sort) in ./week3 to use the SORT algorithm. Clone [this repo](https://github.com/matterport/Mask_RCNN) in ./week3/mask_rcnn to use Mask R-CNN (Keras), and download the official YOLOv3 weights, coco.classes and coco.names from [here](https://pjreddie.com/darknet/yolo/) and include them in ./week3/yolo.
