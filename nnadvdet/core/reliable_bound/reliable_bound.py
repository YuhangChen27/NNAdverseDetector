import numpy as np
from nnadvdet.core.detector import AdverseDetector
from loguru import logger
from .ocsvm import OCSVM
from .sgd import SGDDetector


class ReliableBound(AdverseDetector):
    built_ins = {
        'ocsvm': OCSVM,
        'sgd': SGDDetector
    }

    def __init__(self, config: dict):
        AdverseDetector.__init__(self, config)
        self.occ = None
        name = self.config['ReliableBound']['occ_name']
        if name in self.built_ins:
            self.occ = self.built_ins[name](self.config['ReliableBound']['occ_config'], self.config['num_classes'])
        else:
            raise Exception(f'Do not support {name} as occ.')

    def build_detector(self, x_train: list, pred_train: np.ndarray, label_train: np.ndarray, **kwargs):
        """
        Build detector from both original and transformed samples with
        :param x_train: [np.ndarray(k*n, logits)]
        :param pred_train: np.ndarray(k*n,)
        :param label_train: np.ndarray(k*n,)
        :return:
        """
        assert len(x_train) == 1 and x_train[0].ndim == 2, 'ReliableBound only accept features from single layer.'
        if kwargs:
            self.occ.train(x_train, pred_train, label_train, weights=kwargs['weights'])
        else:
            self.occ.train(x_train, pred_train, label_train)

        logger.info('')

    def detect(self, x: np.ndarray, **kwargs):
        assert x.shape[0] == 1 and 'x_pred' in kwargs
        return self.occ.predict(x, kwargs['x_pred'])

    def batched_detect(self, x: np.ndarray, **kwargs):
        pass

