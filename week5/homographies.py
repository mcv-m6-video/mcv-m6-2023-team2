import numpy as np
import numpy.linalg as LA

from typing import Tuple
from scipy.ndimage import map_coordinates


def forward_warping(p: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Forward warp a point with a given Homography H.
    """
    x1, x2, x3 = H @ p.T
    return x1/x3, x2/x3


def backward_warping(p: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Backward warp a point with a given Homography H.
    """
    x1, x2, x3 = LA.inv(H) @ np.array(p)
    return x1/x3, x2/x3
    

def find_max_size(m: int, n: int, H: np.ndarray) -> Tuple[int, int, int, int]:
    corners = np.array([[0, 0, 1], [n, 0, 1], [0, m, 1], [n, m, 1]])
    corners = np.array(forward_warping(corners, H))

    min_x = np.ceil(corners[0].min())
    max_x = np.floor(corners[0].max())
    min_y = np.ceil(corners[1].min())
    max_y = np.floor(corners[1].max())

    return max_x, min_x, max_y, min_y


def apply_H(I: np.ndarray, H: np.ndarray, min_x, min_y, max_x, max_y) -> Tuple[np.uint, tuple]:
    """
    Applies a homography to an image.

    Args:
        I (np.array): Image to be transformed.
        H (np.array): Homography matrix. The homography is defined as
            H = [[h11, h12, h13],
                [h21, h22, h23],
                [h31, h32, h33]]

    Returns:
        np.array: Transformed image.
    """
    m, n, C = I.shape
    # max_x, min_x, max_y, min_y = find_max_size(m, n, H)

    # Compute size of output image
    # width_canvas, height_canvas = max_x - min_x, max_y - min_y
    width_canvas, height_canvas = 1920, 1080

    # Create grid in the output space such that it is in [min_x, max_x]*1920 and [min_y, max_y]*1080
    # X, Y = np.meshgrid(np.arange(0, 1920), np.arange(0, 1080))
    X, Y = np.meshgrid(np.linspace(min_x, max_x, 1920), np.linspace(min_y, max_y, 1080))
    X_flat, Y_flat = X.flatten(), Y.flatten()

    # Generate matrix with output points in homogenous coordinates
    dest_points = np.array([X_flat, Y_flat, np.ones_like(X_flat)])

    # Backward warp output points to their source points
    source_x, source_y = backward_warping(dest_points, H)

    # Get src_x and src_y in meshgrid-like coordinates
    source_x = np.reshape(source_x, X.shape) 
    source_y = np.reshape(source_y, Y.shape)

    # Set up output image.
    out = np.zeros((int(np.ceil(height_canvas)), int(np.ceil(width_canvas)), 3))

    # Map source coordinates to their corresponding value.
    # Interpolation is needed as coordinates may be real numbers.
    for i in range(C):
        out[:,:,i] = map_coordinates(I[:,:,i], [source_y, source_x])

    return np.uint8(out), (min_x, min_y)