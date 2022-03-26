from sklearn.metrics import auc, roc_curve, precision_recall_curve
from scipy.interpolate import interp1d


class Metrics:
    def __init__(self, config):
        self.config = config
        self.fprs = None
        self.tprs = None
        self.precisions = None
        self.recalls = None

    @staticmethod
    def auroc(scores, labels):
        fpr, tpr, thresholds = roc_curve(scores, labels)
        area = auc(fpr, tpr)
        print('AUROC:', area)
        return area

    @staticmethod
    def aupr(scores, labels):
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        area = auc(precision, recall)
        print('AUPR:', area)
        return area

    def fpr95(self):
        if self.fprs is None:
            raise Exception('Please calculate auroc first.')
        fpr95 = float(interp1d(self.tprs, self.fprs)(0.95))
        print('FPR95:', fpr95)
        return fpr95
