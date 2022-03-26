from tqdm import tqdm
import numpy as np
from loguru import logger
from sklearn.linear_model import SGDClassifier, SGDRegressor
from .one_class import OneClassDetector


class SGDDetector(OneClassDetector):
    def __init__(self, config, num_classes):
        OneClassDetector.__init__(self, config)
        self.classifiers = None
        self.bounds = None
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.scale = config['scale']
        self.num_classes = num_classes

    def train(self, x_train, pred_train, label_train, weights=None):
        self.classifiers = []
        self.bounds = []
        x_train = x_train[0]
        correct = np.where(pred_train == label_train)[0]
        x_train_corr = x_train[correct]
        pred_train_corr = pred_train[correct]
        if weights:
            weights = weights[correct]

        logger.info('Build start......')
        self.train_classifiers(x_train_corr, pred_train_corr, x_train, pred_train, label_train, weights)
        logger.info('Calculating boundaries for each class......')
        self.boundary(x_train_corr, pred_train_corr, x_train, pred_train, label_train)
        logger.info('Build finished......')

    def predict(self, x, x_pred):
        assert x_pred < len(self.classifiers)
        if self.scale:
            lower = min(self.bounds[x_pred])
            upper = max(self.bounds[x_pred])
            scaled_score = (self.classifiers[x_pred].predict(x) - lower) / (upper - lower)
            if scaled_score < 0:
                scaled_score = np.array([0], dtype=np.float32)
            if scaled_score > 1:
                scaled_score = np.array([1], dtype=np.float32)
            return scaled_score[0]
        else:
            return self.classifiers[x_pred].predict(x)[0]

    def train_classifiers(self, x_corr, pred_corr, x, pred, label, weights):
        pbar = tqdm(total=self.num_classes)
        for cls in range(self.num_classes):
            x_corr_cls = x_corr[pred_corr == cls]
            x_corr_other = x_corr[pred_corr != cls]
            x_out = x[pred != label]

            x_train = np.concatenate((x_corr_cls, x_corr_other, x_out), axis=0)
            y_train = np.zeros((x.shape[0]), dtype=int)
            y_train[:x_corr_cls.shape[0]] = 1
            y_train[y_train != 1] = 1 / self.num_classes

            clf = SGDRegressor(loss=self.config['loss'], max_iter=10000)
            if weights:
                clf.fit(x_train, y_train, sample_weight=weights)
            else:
                clf.fit(x_train, y_train)

            # num_batches = x_train.shape[0] // self.batch_size
            # for e in range(self.epochs):
            #     for b in range(num_batches):
            #         start = b * self.batch_size
            #         end = (b + 1) * self.batch_size
            #         if end >= x_corr_cls.shape[0]:
            #             end = x_corr_cls.shape[0]
            #         if weights:
            #             clf.partial_fit(x_corr_cls[start:end], sample_weight=weights[train_corr == cls][start:end])
            #         else:
            #             clf.partial_fit(x_train[start:end], y_train[start:end])
            self.classifiers.append(clf)
            pbar.update(1)
        pbar.close()

    def boundary(self, x_corr, pred_corr, x, pred, label):
        for cls in range(self.num_classes):
            x_corr_cls = x_corr[pred_corr == cls]
            x_other = x_corr[pred_corr != cls]
            x_out = x[pred != label]

            score_in = self.classifiers[cls].predict(x_corr_cls)
            score_other = self.classifiers[cls].predict(x_other)
            score_out = self.classifiers[cls].predict(x_out)

            self.bounds.append([score_in.min(), score_in.max(), score_out.min(), score_out.max(),
                                score_other.min(), score_other.max()])

