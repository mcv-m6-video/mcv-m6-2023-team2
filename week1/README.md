# Week 1

In this week, we carry out the following tasks:

* **Learn about the data to be used later in the project**
    * AICityChallenge
    * KITTI

* **Become familiar with the main evaluation metrics**
    * Mean Intersection over Union (mIoU)
    * Mean Average Precision (mAP)
    * Mean Square Error in Non-occluded areas (MSEN)
    * Percentage of Erroneous Pixels in Non-occluded areas (PEPN)

* **Evaluate and Analyze:**
    * Effect of additive Gaussian noise
    * Tempral evolution of mIoU
    * Optical Flow

## To run the experiments

From directory `week1/`, execute the command
 ```bash
python main.py -h

usage: main.py [-h] [--t1] [--t2] [--t3] [--t4]

Road Traffic Monitoring Analysis for Video Surveillance. MCV-M6-Project. Team
2

optional arguments:
  -h, --help  show this help message and exit
  --t1        Task1 - produce noisy bounding boxes from GT annotations and
              calculate mIoU/AP
  --t2        Task2 - calculate mIoU/AP over time
  --t3        Task 3 - calculate MSEN, PEPN, and visualize errors in the
              estimated optical flow
  --t4        Task 4 - visualize optical flow
```
Note: You must add the `AIcity` directory into `data` to use it
along the annotations file, `ai_challenge_s03_c010_full_annotation.xml`, in path `AICity_data/train/S03/c010`.
