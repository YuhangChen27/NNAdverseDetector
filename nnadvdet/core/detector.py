import numpy as np


class AdverseDetector:
    def __init__(self, config: dict):
        self.config = config
        self.det_scores = None

    def build_detector(self, feat_train: list, pred_train: np.ndarray, label_train: np.ndarray):
        """
        Acquire prior knowledge from training data.
        :param feat_train: List of layer features(depend on corresponding algorithm) extracted from NN model using training samples.
        :param label_train: Ground truth of training samples.
        :param pred_train: Predicted label of training samples.
        """
        raise Exception('Not Implemented.')

    def detect(self, x: list):
        """
        Acquire prior knowledge from training data.
        :param x: Features from a single test input.
        :return Abnormal score for this test input.
        """
        raise Exception('Not Implemented.')

    def batched_detect(self, x, batch_size):
        raise Exception('Not Implemented.')

    def stats(self, y):
        raise Exception('Not Implemented.')
