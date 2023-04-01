class OpticalFlowEstimator:
    def estimate_optical_flow(self, image_prev, image_next):
        pass


class FarneBack(OpticalFlowEstimator):
    def __init__(self, random_params=False):
        pass


class HornSchunk(OpticalFlowEstimator):
    # Source: https://github.com/scivision/pyoptflow
    def __init__(self, random_params=False):
        pass


def get_optical_flow_model(method_name, random_params=False):
    if method_name == "lk":  # Lucas-Kanade
        pass
    elif method_name == "fb":  # FarneBack
        return FarneBack(random_params=random_params)
    elif method_name == "hs":  # Horn-Schunk
        return HornSchunk(random_params=random_params)
    else:
        raise ValueError(f"Method '{method_name}' not found.")
