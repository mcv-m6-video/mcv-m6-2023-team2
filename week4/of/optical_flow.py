import numpy as np

from tqdm import tqdm

from of.of_utils import mse, mae


class OpticalFlowEstimator:
    def estimate_optical_flow(self, image_prev, image_next):
        pass


class BlockMatching(OpticalFlowEstimator):
    def __init__(self, 
                 estimation_type: str = "backward", 
                 search_window_size: int = 32, 
                 block_size: int = 32, 
                 error_function="mse"
                 ):
        self.estimation_type = estimation_type
        self.search_window_half_size = search_window_size // 2
        self.block_size = block_size

        if error_function == "mse":
            self.error_function = mse
        elif error_function == "mae":
            self.error_function = mae

    def estimate_optical_flow(self, image_prev, image_next):
        """
        Estimate optical flow using block matching.

        Args:
            image_prev: Previous grayscale image.
            image_next: Next grayscale image.

        Returns:
             3-channel float32 image with optical flow vectors. h x w x (u, v, 1)
        """
        optical_flow_image = np.zeros((image_prev.shape[0], image_prev.shape[1], 3), dtype=np.float32)

        if self.estimation_type == "backward":
            image_prev, image_next = image_next, image_prev

        for y_prev in tqdm(range(0, image_prev.shape[0]-self.block_size, self.block_size)):
            y_min = max(0, y_prev - self.search_window_half_size)
            y_max = min(image_prev.shape[0] - self.block_size, y_prev + self.search_window_half_size + self.block_size)

            for x_prev in range(0, image_prev.shape[1]-self.block_size, self.block_size):
                x_min = max(0, x_prev - self.search_window_half_size)
                x_max = min(image_prev.shape[1] - self.block_size, x_prev + self.search_window_half_size + self.block_size)

                block_prev = image_prev[y_prev:y_prev + self.block_size, x_prev:x_prev + self.block_size]
                block_prev = block_prev.reshape(-1)

                min_error = np.inf
                min_x_next = min_y_next = 0

                for y_next in range(y_min, y_max):
                    for x_next in range(x_min, x_max):
                        block_next = image_next[y_next:y_next + self.block_size, x_next:x_next + self.block_size]
                        block_next = block_next.reshape(-1)

                        error = self.error_function(block_prev, block_next)
                        
                        if error < min_error:
                            min_error = error
                            min_x_next = x_next
                            min_y_next = y_next

                optical_flow_image[y_prev:y_prev + self.block_size, x_prev:x_prev + self.block_size, 0] = min_x_next - x_prev
                optical_flow_image[y_prev:y_prev + self.block_size, x_prev:x_prev + self.block_size, 1] = min_y_next - y_prev

        optical_flow_image[:, :, 2] = 1
        
        if self.estimation_type == "backward":
            optical_flow_image[:, :, 0] = -optical_flow_image[:, :, 0]
            optical_flow_image[:, :, 1] = -optical_flow_image[:, :, 1]

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
