import numpy as np


class AdverseDetector:
    def __init__(self, config: dict):
        self.config = config
        self.det_scores = None

    def build_detector(self, x_train: np.ndarray, pred_train: np.ndarray, label_train: np.ndarray):
        raise Exception('Not Implemented.')

    def detect(self, x):
        pass

    def batched_detect(self, x):
        pass
