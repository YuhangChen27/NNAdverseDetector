import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.covariance import EmpiricalCovariance
from nnadvdet.core.detector import AdverseDetector
from loguru import logger
from tqdm import tqdm


class MahalanobisDetector(AdverseDetector):
    def __init__(self, config: dict):
        AdverseDetector.__init__(self, config)
        self.num_classes = config['num_classes']
        self.mean = None
        self.precision = None

    @staticmethod
    def sample_estimator(features, pred, target, num_classes=10, require_tp=False):
        group_lasso = EmpiricalCovariance(assume_centered=False)
        num_features = len(features)
        features_mean_class = []
        precision = []

        pbar = tqdm(total=num_features*num_classes)
        for f in range(num_features):
            feat = features[f]
            diff = np.zeros((feat.shape[0], feat.shape[-1]), dtype=float)
            # feature dim: [num_samples, w, h, channels]
            if feat.ndim > 2:
                feat = feat.reshape(feat.shape[0], -1, feat.shape[-1])
                feat = np.mean(feat, axis=1)

            feat_mean_class = []
            for i in range(num_classes):
                if require_tp:
                    tp = np.where(pred == target)[0]
                    class_ind = tp[np.where(pred[tp] == i)[0]]
                else:
                    class_ind = np.where(target == i)[0]
                mean = np.mean(feat[class_ind], axis=0)
                diff[class_ind] = feat[class_ind] - mean
                feat_mean_class.append(mean)
                pbar.update(1)
            group_lasso.fit(diff)
            precision_temp = group_lasso.precision_
            precision.append(precision_temp)
            features_mean_class.append(feat_mean_class)

        pbar.close()
        return features_mean_class, precision

    @staticmethod
    def get_mahalanobis_score(features, sample_mean, precision, num_classes=10):
        num_features = len(features)
        mahalanobis = []
        for f in range(num_features):
            feat = features[f]
            if feat.ndim > 2:
                feat = feat.reshape(feat.shape[0], -1, feat.shape[-1])
                feat = np.mean(feat, axis=1)
            gaussian_score = []
            for i in range(num_classes):
                diff = feat - sample_mean[f][i]
                term_gau = -0.5 * np.diag(np.matmul(np.matmul(diff, precision[f]), diff.T))
                gaussian_score.append(term_gau)
            gaussian_score = np.array(gaussian_score)
            mahalanobis.append(np.max(gaussian_score, axis=0))
        return mahalanobis

    @staticmethod
    def regression(x_test, x_out, train_percent=0.1):
        num_test = x_test[0].shape[0]
        num_out = x_out[0].shape[0]
        num_feat = len(x_test)
        sc_test = []
        sc_out = []

        for i in range(num_test):
            test_tmp = []
            for j in range(num_feat):
                test_tmp.append(x_test[j][i])
            sc_test.append(test_tmp)

        for i in range(num_out):
            out_tmp = []
            for j in range(num_feat):
                out_tmp.append(x_out[j][i])
            sc_out.append(out_tmp)

        sc_test = np.array(sc_test)
        sc_out = np.array(sc_out)
        X_train = np.concatenate((sc_test[:int(num_test*train_percent)], sc_out[:int(num_test*train_percent)]), axis=0)
        Y_train = np.concatenate((np.ones(int(num_test*train_percent)), np.zeros(int(num_test*train_percent))), axis=0)
        lr = LogisticRegressionCV(n_jobs=-1).fit(X_train, Y_train)
        X_test = np.concatenate((sc_test, sc_out), axis=0)
        score = lr.predict_proba(X_test)[:, 1]
        return score

    def build_detector(self, x_train: list, pred_train: np.ndarray, label_train: np.ndarray):
        logger.info('Mahalanobis distance build start......')
        self.mean, self.precision = self.sample_estimator(x_train, pred_train, label_train, num_classes=self.num_classes)
        logger.info('Mahalanobis distance build finished......')

    def detect(self, x: np.ndarray, **kwargs):
        score = self.get_mahalanobis_score([x], self.mean, self.precision, self.num_classes)
        return score[-1]

    def batched_detect(self, x: list, **kwargs):
        score = self.get_mahalanobis_score(x, self.mean, self.precision, self.num_classes)
        return score[-1]
