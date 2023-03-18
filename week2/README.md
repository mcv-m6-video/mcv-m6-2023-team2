# Week 2

The goal of this project is to:
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
  + Bes 𝛼 for non-recursive case, best ⍴ for recursive case
  + Joint grid/random search over (𝛼, ⍴)
+ Compare the adaptive and non-adaptive versions qualitatively and quantitatively

### Task 3: Compare with SOTA and evaluate one SOTA method
+ Choose one SOTA method, implement it or find an existing implementation:
    + P. KaewTraKulPong et.al. An improved adaptive background mixture model for real-time tracking with shadow detection. In Video-Based Surveillance Systems, 2002. Implementation: BackgroundSubtractorMOG (OpenCV)
    + Z. Zivkovic et.al. Efficient adaptive density estimation per image pixel for the task of background subtraction, Pattern Recognition Letters, 2005. Implementation: BackgroundSubtractorMOG2 (OpenCV)
    + L. Guo, et.al. Background subtraction using local svd binary pattern. CVPRW, 2016. Implementation: BackgroundSubtractorLSBP (OpenCV)
    + St-Charles, Pierre-Luc, and Guillaume-Alexandre Bilodeau. Improving Background Subtraction using Local Binary Similarity Patterns. Applications of Computer Vision (WACV), 2014.
    + M. Braham et al. Deep background subtraction with scene-specific convolutional neural networks. In International Conference on Systems, Signals and Image Processing, 2016.
+ Qualitatively and quantiatively compare with the simple statistical baseline

### Task 4: Update simple Gaussian model to support color sequences
+ Use multiple gaussians in different color spaces


## Execution

# TODO