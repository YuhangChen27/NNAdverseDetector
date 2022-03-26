import numpy as np


class AdverseDetector:
    def __init__(self, config: dict):
        self.config = config
        self.det_scores = None

    def build_detector(self, feat_train: list, pred_train: np.ndarray, label_train: np.ndarray):
        """
        Acquire prior knowledge from training data.
        :param feat_train: List of layer features (depend on corresponding algorithm) extracted from NN model using training samples.
        :param label_train: Ground truth of training samples, np.ndarray(num_samples, ).
        :param pred_train: Predicted label of training samples, np.ndarray(num_samples, ).
        """
        raise Exception('Not Implemented.')

    def detect(self, x: list, **kwargs):
        """
        Detect abnormal intensity for a single input.
        :param x: Features (depend on corresponding algorithm) from a single test input, in the form
                  [np.ndarray(num_neurons_layer1), ...].
        :return Abnormal score for this test input.
        """
        raise Exception('Not Implemented.')

    def batched_detect(self, x: list, **kwargs):
        """
        Detect abnormal intensities for batch of inputs.
        :param x: List of features (depend on corresponding algorithm) from test inputs, in the form
                  [np.ndarray((num_samples, num_neurons_layer1)), ...].
        :return Abnormal scores for all test inputs.
        """
        raise Exception('Not Implemented.')

    def stats(self, y: np.ndarray):
        """
        Calculates detector performance using different metrics (e.g. AUROC).
        :param y: Ground truth labels for the latest test inputs.
        """
        raise Exception('Not Implemented.')
