from tqdm import tqdm
import numpy as np
from loguru import logger
from sklearn.linear_model import SGDOneClassSVM
from sklearn.kernel_approximation import Nystroem
from .one_class import OneClassDetector


class OCSVM(OneClassDetector):
    def __init__(self, config, num_classes):
        OneClassDetector.__init__(self, config)
        self.classifiers = None
        self.bounds = None
        self.kernel_approximator = None
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.scale = config['scale']
        self.num_classes = num_classes

    def train(self, x_train, pred_train, label_train, weights=None):
        logger.info('Build start......')
        self.classifiers = []
        self.bounds = []
        x_train = x_train[0]

        # logger.info('Non-linear transform......')
        # x_train = self.fit_approximator(x_train)
        correct = np.where(pred_train == label_train)[0]
        x_train_corr = x_train[correct]
        pred_train_corr = pred_train[correct]
        if weights:
            weights = weights[correct]

        logger.info('Train classifiers......')
        self.train_classifiers(x_train_corr, pred_train_corr, weights)
        logger.info('Calculating boundaries for each class......')
        self.boundary(x_train_corr, pred_train_corr, x_train, pred_train, label_train)
        logger.info('Build finished......')

    def predict(self, x, x_pred):
        assert x_pred < len(self.classifiers)
        # x = self.kernel_approximator.transform(x)
        if self.scale:
            lower = min(self.bounds[x_pred])
            upper = max(self.bounds[x_pred])
            scaled_score = (self.classifiers[x_pred].score_samples(x) - lower) / (upper - lower)
            if scaled_score < 0:
                scaled_score = np.array([0], dtype=np.float32)
            if scaled_score > 1:
                scaled_score = np.array([1], dtype=np.float32)
            return scaled_score[0]
        else:
            return self.classifiers[x_pred].score_samples(x)[0]

    def train_classifiers(self, train_corr, pred_corr, weights):
        pbar = tqdm(total=self.num_classes)
        for cls in range(self.num_classes):
            x_corr_cls = train_corr[pred_corr == cls]
            clf = SGDOneClassSVM(nu=self.config['nu'])
            if weights is not None:
                clf.fit(x_corr_cls, weights)
            else:
                clf.fit(x_corr_cls)
            self.classifiers.append(clf)
            pbar.update(1)
        pbar.close()

    def boundary(self, x_corr, pred_corr, x, pred, label):
        for cls in range(self.num_classes):
            x_corr_cls = x_corr[pred_corr == cls]
            x_other = x_corr[pred_corr != cls]
            x_out = x[pred != label]

            score_in = self.classifiers[cls].score_samples(x_corr_cls)
            score_other = self.classifiers[cls].score_samples(x_other)
            score_out = self.classifiers[cls].score_samples(x_out)

            self.bounds.append([score_in.min(), score_in.max(), score_out.min(), score_out.max(),
                                score_other.min(), score_other.max()])

    def fit_approximator(self, x, y=None):
        self.kernel_approximator = Nystroem(random_state=42, n_components=300, n_jobs=-1, gamma=0.0001)
        if y is not None:
            x_transform = self.kernel_approximator.fit_transform(x, y)
        else:
            x_transform = self.kernel_approximator.fit_transform(x)
        return x_transform
