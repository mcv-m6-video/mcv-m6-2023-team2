# Week 2

The goals of this week are:
* **Perform background estimation**
  * Model the background pixels of a video with a simple statistical model that classifies pixels into background / foreground
    + Single per-pixel Gaussian
    + Adaptive / Non-adaptive modeling

* **Compare simple models with more complex ones**

### Task 1: Implement and evaluate per-pixel Gaussian distribution
+ A Gaussian distribution models the background at each pixel
  + First 25% of the video sequence to extract statistics (mean and variance of pixels)
  + Second 75% to segment the foreground and evaluate
+ Evaluate results

### Task 2: Implement and evaluate Adaptive Gaussian modeling
+ Recursive formulation as moving average
  + First 25% of frames for training
  + Second 75% of frames for background adaptation
+ Best pair of values (𝛼, ⍴) to maximize mAP
  + Best 𝛼 for non-recursive case, best ⍴ for recursive case
  + Joint grid/random search over (𝛼, ⍴)
+ Compare the adaptive and non-adaptive versions qualitatively and quantitatively

### Task 3: Compare with SOTA and evaluate one SOTA method
+ Choose one SOTA method, implement it or find an existing implementation (chosen methods):
    + Z. Zivkovic et.al. Efficient adaptive density estimation per image pixel for the task of background subtraction, Pattern Recognition Letters, 2005. Implementation: BackgroundSubtractorMOG2 (OpenCV)
    + L. Guo, et.al. Background subtraction using local svd binary pattern. CVPRW, 2016. Implementation: BackgroundSubtractorLSBP (OpenCV)
    + St-Charles, Pierre-Luc, and Guillaume-Alexandre Bilodeau. Improving Background Subtraction using Local Binary Similarity Patterns. Applications of Computer Vision (WACV), 2014.

### Task 4: Update simple Gaussian model to support color sequences
+ Use multiple gaussians in different color spaces


## Execution

From directory `week2/`, execute the command
 ```bash
python main.py -h

usage: main.py [-h] [--t1] [--t2] [--t3] [--t4] [--path_video PATH_VIDEO]
               [--path_roi PATH_ROI] [--path_GT PATH_GT]
               [--path_results PATH_RESULTS] [--bg_model BG_MODEL]
               [--alpha ALPHA] [--rho RHO] [--viz_bboxes]
               [--color_space COLOR_SPACE] [--voting VOTING] [--store_results]
               [--make_gifs]

Road Traffic Monitoring Analysis for Video Surveillance. MCV-M6-Project. Team
2

optional arguments:
  -h, --help            show this help message and exit
  --t1                  Task1 - background estimation with non-adaptive
                        Gaussian model
  --t2                  Task2 - background estimation with adaptive Gaussian
                        model
  --t3                  Task 3 - explore and evaluate a SOTA method
  --t4                  Task 4 - background estimation with non-adaptive,
                        color-aware, multidimensional Gaussian model
  --path_video PATH_VIDEO
                        The path to the video file to be processed.
  --path_roi PATH_ROI   The path to the ROI file corresponding to the video to
                        be processed.
  --path_GT PATH_GT     The path to the ground truth file corresponding to the
                        video to be processed.
  --path_results PATH_RESULTS
                        The path to the directory where the results will be
                        stored.
  --bg_model BG_MODEL   Model to be used for background estimation.
  --alpha ALPHA         alpha parameter
  --rho RHO             rho parameter
  --viz_bboxes          Whether to visualize the bounding boxes.
  --color_space COLOR_SPACE
                        Color space to be used for background estimation.
  --voting VOTING       Voting scheme to be used for background estimation.
  --store_results       Whether to store the intermediate results.
  --make_gifs           Whether to store make GIFs of the intermediate
                        results.
                        
  --sus                 KNN, MOG2 or LBSP background substraction algorithm.
  --make_video          Bool, if True the script will save an output video with
                        task3 process
```
