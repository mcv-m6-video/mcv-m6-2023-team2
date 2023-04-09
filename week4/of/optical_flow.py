import numpy as np
import cv2
import pyflow

from tqdm import tqdm
from typing import Tuple, Optional

from utils import resize_image_keep_aspect_ratio
from of.of_utils import mse, mae


class OpticalFlowEstimator:
    def estimate_optical_flow(self, image_prev, image_next):
        pass


class BlockMatching(OpticalFlowEstimator):
    
    def __init__(self,
                 estimation_type: str = "forward",
                 search_window_size: int = 76,
                 block_size: int = 24,
                 error_function="nccorr"
                 ):
        self.estimation_type = estimation_type
        self.search_window_half_size = search_window_size // 2
        self.block_size = block_size

        if error_function == "mse":
            self.error_function = mse
            self.matching_function = self.match_blocks
        elif error_function == "mae":
            self.error_function = mae
            self.matching_function = self.match_blocks
        elif error_function == "nccorr":
            self.error_function = cv2.TM_CCORR_NORMED
            self.matching_function = self.match_templates
        elif error_function == "nccoeff":
            # For more info, check https://stackoverflow.com/questions/55469431/what-does-the-tm-ccorr-and-tm-ccoeff-in-opencv-mean
            self.error_function = cv2.TM_CCOEFF_NORMED
            self.matching_function = self.match_templates
        else:
            raise ValueError(f"Error function '{error_function}' not found.")

    def match_templates(self, block_prev, image_next, y_min, y_max, x_min, x_max):
        search_window = image_next[y_min:y_max, x_min:x_max]
        res = cv2.matchTemplate(search_window, block_prev, self.error_function)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        return max_loc[0] + x_min, max_loc[1] + y_min

    def match_blocks(self, block_prev, image_next, y_min, y_max, x_min, x_max):
        block_prev = block_prev.reshape(-1)
        min_error = np.inf
        min_x_next = min_y_next = 0

        for y_next in range(y_min, y_max - self.block_size):
            for x_next in range(x_min, x_max - self.block_size):
                block_next = image_next[y_next:y_next +
                                        self.block_size, x_next:x_next + self.block_size]
                block_next = block_next.reshape(-1)

                error = self.error_function(block_prev, block_next)

                if error < min_error:
                    min_error = error
                    min_x_next = x_next
                    min_y_next = y_next

        return min_x_next, min_y_next

    def estimate_optical_flow(self, image_prev, image_next, leave_tqdm=True):
        """
        Estimate optical flow using block matching.

        Args:
            image_prev: Previous grayscale image.
            image_next: Next grayscale image.

        Returns:
             3-channel float32 image with optical flow vectors. h x w x (u, v, 1)
        """
        optical_flow_image = np.zeros(
            (image_prev.shape[0], image_prev.shape[1], 3), dtype=np.float32)

        if self.estimation_type == "backward":
            image_prev, image_next = image_next, image_prev

        for y_prev in tqdm(range(0, image_prev.shape[0]-self.block_size, self.block_size),
                           desc="Estimating optical flow using block matching... (rows to process)",
                           leave=leave_tqdm,
                           ):
            for x_prev in range(0, image_prev.shape[1]-self.block_size, self.block_size):
                y_min = max(0, y_prev - self.search_window_half_size)
                y_max = min(
                    image_prev.shape[0], y_prev + self.search_window_half_size + self.block_size)

                x_min = max(0, x_prev - self.search_window_half_size)
                x_max = min(
                    image_prev.shape[1], x_prev + self.search_window_half_size + self.block_size)

                block_prev = image_prev[y_prev:y_prev +
                                        self.block_size, x_prev:x_prev + self.block_size]

                min_x_next, min_y_next = self.matching_function(
                    block_prev, image_next, y_min, y_max, x_min, x_max)

                optical_flow_image[y_prev:y_prev + self.block_size,
                                   x_prev:x_prev + self.block_size, 0] = min_x_next - x_prev
                optical_flow_image[y_prev:y_prev + self.block_size,
                                   x_prev:x_prev + self.block_size, 1] = min_y_next - y_prev

        optical_flow_image[:, :, 2] = 1

        if self.estimation_type == "backward":
            optical_flow_image[:, :, 0] = -optical_flow_image[:, :, 0]
            optical_flow_image[:, :, 1] = -optical_flow_image[:, :, 1]

        return optical_flow_image
    
    def postprocess(self, optical_flow_image, variance_thr: float = 100, color_diff_thr: float = 8, window_size: int = 3):
        new_flow = optical_flow_image.copy()
        # Interpolate the optical flow in border pixels from neighborhood
        # For instance, for each pixel of the left border, we calculate 
        # the optical flow using the pixels of (i, block_size), (i-1, block_size), (i+1, block_size)
        # by interpolation
        for i in range(0, optical_flow_image.shape[0], self.block_size):
            new_flow[i:i+self.block_size, :self.block_size, :] = (optical_flow_image[i-1, self.block_size, :] + optical_flow_image[i, self.block_size, :] + optical_flow_image[i+1, self.block_size, :]) / 3
            new_flow[i:i+self.block_size, -self.block_size*2:, :] = (optical_flow_image[i-1, -self.block_size*2-1, :] + optical_flow_image[i, -self.block_size*2-1, :] + optical_flow_image[i+1, -self.block_size*2-1, :]) / 3

        for j in range(0, optical_flow_image.shape[1], self.block_size):
            new_flow[:self.block_size, j:j+self.block_size, :] = (optical_flow_image[self.block_size, j-1, :] + optical_flow_image[self.block_size, j, :] + optical_flow_image[self.block_size, j+1, :]) / 3
            new_flow[-self.block_size:, j:j+self.block_size, :] = (optical_flow_image[-self.block_size-1, j-1, :] + optical_flow_image[-self.block_size-1, j, :] + optical_flow_image[-self.block_size-1, j+1, :]) / 3

        # Special corner cases 
        new_flow[:self.block_size, :self.block_size, :] = (new_flow[self.block_size, self.block_size, :] + new_flow[self.block_size, 0, :] + new_flow[0, self.block_size, :]) / 3
        new_flow[:self.block_size, -self.block_size:, :] = (new_flow[self.block_size, -self.block_size-1, :] + new_flow[self.block_size, -self.block_size*2-1, :] + new_flow[0, -self.block_size-1, :]) / 3
        new_flow[-self.block_size:, :self.block_size, :] = (new_flow[-self.block_size-1, self.block_size, :] + new_flow[-self.block_size-1, 0, :] + new_flow[-self.block_size*2-1, self.block_size, :]) / 3
        new_flow[-self.block_size:, -self.block_size:, :] = (new_flow[-self.block_size-1, -self.block_size-1, :] + new_flow[-self.block_size-1, -self.block_size*2-1, :] + new_flow[-self.block_size*2-1, -self.block_size-1, :]) / 3
        optical_flow_image = new_flow
        new_flow = optical_flow_image.copy()
        hsv = cv2.cvtColor(optical_flow_image, cv2.COLOR_BGR2HSV)
        hue = hsv[:,:,2]
        # save the hue image
        cv2.imwrite("hue.png", hue)

        # Define the window size and threshold for computing the mask
        window_size = self.block_size*window_size
        mask = np.zeros_like(new_flow[:,:,0], dtype=np.uint8)

        # Compute the var of the flow in each window
        variance = np.zeros_like(new_flow[:,:,0])
        for i in range(0, new_flow.shape[0], self.block_size):
            for j in range(0, new_flow.shape[1], self.block_size):
                x_min = max(0, j - window_size)
                x_max = min(new_flow.shape[1], j + window_size)
                y_min = max(0, i - window_size)
                y_max = min(new_flow.shape[0], i + window_size)
                
                # # Compute the variance of the flow in the window only in the masked area
                window = hue[y_min:y_max, x_min:x_max]
                variance[i:i+self.block_size,j:j+self.block_size] = np.var(window)

                current_color = np.mean(optical_flow_image[i:i+self.block_size, j:j+self.block_size, :], axis=(0, 1))
                
                # Sum all the colors in the neighborhood and divide by 8 to get the mean color
                neighbors_color = []
                for k in range(i-self.block_size, i+self.block_size+1, self.block_size):
                    for l in range(j-self.block_size, j+self.block_size+1, self.block_size):
                        if k == i and l == j:
                            continue
                        k = max(0, k)
                        k = min(optical_flow_image.shape[0]-1, k)
                        l = max(0, l)
                        l = min(optical_flow_image.shape[1]-1, l)
                        neighbors_color.append(np.mean(optical_flow_image[k:k+self.block_size, l:l+self.block_size, :], axis=(0, 1)))

                neighbors_color_mean = np.mean(neighbors_color, axis=0)

                # If the average color is different from the mean color, we interpolate the optical flow
                if np.linalg.norm(current_color - neighbors_color_mean) > color_diff_thr:
                    # new_flow[i:i+self.block_size, j:j+self.block_size, :] = neighbors_color_mean
                    mask[i:i+self.block_size, j:j+self.block_size] = 1

        # Create a binary mask based on the var threshold
        var_mask = (variance > variance_thr).astype(np.uint8)
        var_mask = cv2.dilate(var_mask, np.ones((self.block_size*3,self.block_size*3), np.uint8), iterations=1)
        mask += var_mask

        cv2.imwrite("mask.png", mask*255)
        # Inpaint the masked areas using Navier-Stokes based method
        new_flow[:,:,0] = cv2.inpaint(new_flow[:,:,0], mask, 3, cv2.INPAINT_NS)
        new_flow[:,:,1] = cv2.inpaint(new_flow[:,:,1], mask, 3, cv2.INPAINT_NS)

        optical_flow_image = new_flow

        # Apply a median filter to the optical flow
        optical_flow_image = cv2.medianBlur(optical_flow_image, 5)

        # Apply a Gaussian filter to the optical flow
        optical_flow_image = cv2.GaussianBlur(optical_flow_image, (55, 55), 0)
        optical_flow_image = cv2.GaussianBlur(optical_flow_image, (27, 27), 0)
        optical_flow_image = cv2.GaussianBlur(optical_flow_image, (13, 13), 0)
        optical_flow_image[:,:,2] = 1 # Just in case
        return optical_flow_image


class Coarse2FineFlow(OpticalFlowEstimator):
    """
    Coarse2FineFlow optical flow estimator.

    Args (default values in the library):
        alpha (0.012): the regularization weight
        ratio (0.75): the downsample ratio
        minWidth (20): the width of the coarsest level
        nOuterFPIterations (7): the number of outer fixed point iterations
        nInnerFPIterations (1): the number of inner fixed point iterations
        nSORIterations (30): the number of SOR iterations
        colType (1): the color type
    """

    def __init__(self,
                 alpha: float = 0.00794736062813633,
                 ratio: float = 0.6980577394671629,
                 minWidth: int = 12,
                 nOuterFPIterations: int = 6,
                 nInnerFPIterations: int = 7,
                 nSORIterations: int = 28,
                 colType: int = 1,
                 resize_to: Optional[int] = None,
                 ):
        self.alpha = alpha
        self.ratio = ratio
        self.minWidth = minWidth
        self.nOuterFPIterations = nOuterFPIterations
        self.nInnerFPIterations = nInnerFPIterations
        self.nSORIterations = nSORIterations
        self.colType = colType
        self.resize_to = resize_to

    def estimate_optical_flow(self, image_prev, image_next):
        """
        Estimate optical flow using Coarse2FineFlow algorithm.

        Args:
            image_prev: Previous grayscale image.
            image_next: Next grayscale image.

        Returns:
             3-channel float32 image with optical flow vectors. h x w x (u, v, 1)
        """
        if self.resize_to is not None:
            image_prev = resize_image_keep_aspect_ratio(image_prev, self.resize_to)
            image_next = resize_image_keep_aspect_ratio(image_next, self.resize_to)

        image_prev = image_prev.astype(float) / 255
        image_next = image_next.astype(float) / 255

        u, v, im2W = pyflow.coarse2fine_flow(
            image_prev, image_next, self.alpha, 
            self.ratio, self.minWidth, self.nOuterFPIterations, 
            self.nInnerFPIterations, self.nSORIterations, self.colType)
        
        flow = np.stack((u, v, np.ones_like(u)), axis=2)
        return flow


class FarneBack(OpticalFlowEstimator):
    def __init__(self, random_params=False):
        pass


class HornSchunk(OpticalFlowEstimator):
    # Source: https://github.com/scivision/pyoptflow
    def __init__(self, random_params=False):
        pass


def get_optical_flow_model(method_name, random_params=False):
    if method_name == "bm":  # Block Matching
        return BlockMatching()
    elif method_name == "lk":  # Lucas-Kanade
        pass
    elif method_name == "fb":  # FarneBack
        return FarneBack(random_params=random_params)
    elif method_name == "hs":  # Horn-Schunk
        return HornSchunk(random_params=random_params)
    else:
        raise ValueError(f"Method '{method_name}' not found.")
