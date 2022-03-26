import numpy as np
from nnadvdet.core.detector import AdverseDetector
from loguru import logger


class DeepGini(AdverseDetector):
    def __init__(self, config: dict):
        AdverseDetector.__init__(self, config)

    def build_detector(self, x_train: list, pred_train: np.ndarray, label_train: np.ndarray):
        assert len(x_train) == 1 and x_train[0].ndim == 2 and \
            'DeepGini only accept features from logits layer.'
        logger.info('DeepGini build finished.')

    def detect(self, x: np.ndarray, **kwargs):
        assert len(x) == 1 and x[0].ndim == 2 and \
            'DeepGini only accept features from logits layer.'
        x = x[0]
        exp = np.exp(x - np.max(x))
        conf = exp / sum(exp)
        return np.sum(np.power(conf, 2))

    def batched_detect(self, x: list, **kwargs):
        assert len(x) == 1 and x[0].ndim == 2 and \
            'DeepGini only accept features from logits layer.'
        x = x[0]
        max = np.max(x, axis=-1, keepdims=True)
        exp = np.exp(x - max)
        conf = exp / np.sum(exp, axis=-1, keepdims=True)
        return np.sum(np.power(conf, 2), axis=-1)
