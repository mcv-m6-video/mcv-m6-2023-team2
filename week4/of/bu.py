import time
import numpy as np

from sklearn.feature_extraction.image import extract_patches_2d
from tqdm import tqdm

from utils import mse, mae


class OpticalFlowEstimator:
    def estimate_optical_flow(self, image_prev, image_next):
        pass


class BlockMatching(OpticalFlowEstimator):
    def __init__(self, 
                 estimation_type: str = "backward", 
                 search_area_radius: int = 15, 
                 block_size: int = 7, 
                 error_function="mse"
                 ):
        self.estimation_type = estimation_type
        self.search_area_radius = search_area_radius
        self.block_size = block_size
        self.stride = stride

        if error_function == "mse":
            self.error_function = mse
        elif error_function == "mae":
            self.error_function = mae

    def estimate_optical_flow(self, image_prev, image_next):
        """
        Estimate optical flow using block matching.

        Args:
            image_prev: Previous image.
            image_next: Next image.

        Returns:
             3-channel uint16 image with optical flow vectors. h x w x (u, v, 1)
        """
        start = time.time()
        optical_flow_image = np.zeros((image_prev.shape[0], image_prev.shape[1], 3), dtype=np.uint16)

        if self.estimation_type == "backward":
            image_prev, image_next = image_next, image_prev

        for 

        # patches_prev = extract_patches_2d(image_prev, (self.block_size, self.block_size))
        # patches_next = extract_patches_2d(image_next, (self.block_size, self.block_size))

        # for index_prev, patch_prev in tqdm(enumerate(patches_prev), total=len(patches_prev)):
            # min_error = np.inf
            # min_index_next = 0

            # for index_next, patch_next in enumerate(patches_next):
            #     if self.search_area_radius > 0:
            #         x_next, y_next = np.unravel_index(index_next, image_next.shape)
            #         x_prev, y_prev = np.unravel_index(index_prev, image_prev.shape)
            #         distance = np.sqrt((x_next - x_prev) ** 2 + (y_next - y_prev) ** 2)
            #         if distance > self.search_area_radius:
            #             continue
                    
            #     if self.stride > 1:
            #         x_next, y_next = np.unravel_index(index_next, image_next.shape)
            #         if x_next % self.stride != 0 or y_next % self.stride != 0:
            #             continue

            #     error = self.error_function(patch_prev, patch_next)

            #     if error < min_error:
            #         min_error = error
            #         min_index_next = index_next

            # distances = self.error_function(patch_prev, patches_next)

            # # Search only in the search area
            # if self.search_area_radius > 0:
            #     x_prev, y_prev = np.unravel_index(index_prev, image_prev.shape)
            #     x_next, y_next = np.unravel_index(np.arange(len(distances)), image_next.shape)
            #     distances[np.sqrt((x_next - x_prev) ** 2 + (y_next - y_prev) ** 2) > self.search_area_radius] = np.inf
                
            # min_index_next = np.argmin(distances)

            # x_prev, y_prev = np.unravel_index(index_prev, image_prev.shape)
            # x_next, y_next = np.unravel_index(min_index_next, image_next.shape)

            # optical_flow_image[x_prev, y_prev, 0] = x_next - x_prev
            # optical_flow_image[x_prev, y_prev, 1] = y_next - y_prev
            # optical_flow_image[x_prev, y_prev, 2] = 1

        print(f"Block matching took {time.time() - start} seconds.")
        return optical_flow_image
            

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
