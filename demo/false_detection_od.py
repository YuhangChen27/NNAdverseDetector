import numpy as np
from nnadvdet.core.mahalanobis import MahalanobisDetector
from nnadvdet.core.odin import Odin
from nnadvdet.core.ginicoef import DeepGini
from nnadvdet.core.openmax import OpenmaxLibmr
from nnadvdet.core.reliable_bound import ReliableBound
from copy import deepcopy
import torch
from tqdm import tqdm
from sklearn.metrics import auc, roc_curve


config = {
    'num_classes': 8,
    'Odin': {
        'temper': 1000,
        # 'noiseMagnitude': 0.0014,
    },
    'ReliableBound': {
        'occ_name': 'sgd',
        'occ_config': {
            'epochs': 1000,
            'batch_size': 100000,
            'nu': 0.01,
            'scale': True,
            'loss': 'epsilon_insensitive'
        }
    }
}


def eval_ood(det, x_test, preds_test):
    result = []
    pbar = tqdm(total=x_test.shape[0])
    for i in range(x_test.shape[0]):
        result.append(det.detect(x_test[i].reshape(1, -1), x_pred=preds_test[i]))
        pbar.update(1)
    pbar.close()
    return result


if __name__ == '__main__':
    fcs_train = np.load('/home/kengo/Documents/GitHub/mmdetection-intermediate/demo/extract/train/fcs_city_train.npy')
    logits_train = np.load('/home/kengo/Documents/GitHub/mmdetection-intermediate/demo/extract/train/logits_city_train.npy')
    preds_train = np.load('/home/kengo/Documents/GitHub/mmdetection-intermediate/demo/extract/train/preds_city_train.npy')
    labels_train = np.load('/home/kengo/Documents/GitHub/mmdetection-intermediate/demo/extract/train/labels_city_train.npy')
    flags_train = np.load('/home/kengo/Documents/GitHub/mmdetection-intermediate/demo/extract/train/flags_city_train.npy')

    print(flags_train.sum())
    print(np.where(labels_train == 8)[0].shape)
    adv_det0 = DeepGini(config)
    adv_det0.build_detector([logits_train], preds_train, labels_train)
    adv_det = ReliableBound(config)
    adv_det.build_detector([logits_train], preds_train, labels_train)
    adv_det2 = OpenmaxLibmr(config)
    adv_det2.build_detector([deepcopy(logits_train[:, :config["num_classes"]])], preds_train, labels_train)
    adv_det3 = MahalanobisDetector(config)
    adv_det3.build_detector([fcs_train], preds_train, labels_train)
    adv_det4 = Odin(config)
    adv_det4.build_detector([logits_train], preds_train, labels_train)

    fcs_test = np.load('/home/kengo/Documents/GitHub/mmdetection-intermediate/demo/extract/train/fcs_city_val.npy')
    logits_test = np.load('/home/kengo/Documents/GitHub/mmdetection-intermediate/demo/extract/train/logits_city_val.npy')
    preds_test = np.load('/home/kengo/Documents/GitHub/mmdetection-intermediate/demo/extract/train/preds_city_val.npy')
    flags_test = np.load('/home/kengo/Documents/GitHub/mmdetection-intermediate/demo/extract/train/flags_city_val.npy')
    softmax_test = np.load('/home/kengo/Documents/GitHub/mmdetection-intermediate/demo/extract/train/softmax_city_val.npy')

    fpr, tpr, thres = roc_curve(flags_test, softmax_test)
    print('softmax', auc(fpr, tpr))

    result0 = adv_det0.batched_detect([logits_test])
    fpr0, tpr0, thres0 = roc_curve(flags_test, result0)
    print('deepgini', auc(fpr0, tpr0))

    result = eval_ood(adv_det, logits_test, preds_test)
    fpr2, tpr2, thres2 = roc_curve(flags_test, result)
    print('proposed', auc(fpr2, tpr2))

    result2 = eval_ood(adv_det2, deepcopy(logits_test[:, :config["num_classes"]]), preds_test)
    fpr3, tpr3, thres3 = roc_curve(flags_test, result2)
    print('openmax', auc(fpr3, tpr3))

    result3 = adv_det3.batched_detect([fcs_test])
    fpr4, tpr4, thres4 = roc_curve(flags_test, result3)
    print('mahal', auc(fpr4, tpr4))

    result4 = adv_det4.batched_detect([logits_test])
    fpr5, tpr5, thres5 = roc_curve(flags_test, result4)
    print('odin', auc(fpr5, tpr5))
    print('\n\n')

    fcs_test = np.load('/home/kengo/Documents/GitHub/mmdetection-intermediate/demo/extract/foggy/fcs_city_foggy_train.npy')
    logits_test = np.load('/home/kengo/Documents/GitHub/mmdetection-intermediate/demo/extract/foggy/logits_city_foggy_train.npy')
    preds_test = np.load('/home/kengo/Documents/GitHub/mmdetection-intermediate/demo/extract/foggy/preds_city_foggy_train.npy')
    flags_test = np.load('/home/kengo/Documents/GitHub/mmdetection-intermediate/demo/extract/foggy/flags_city_foggy_train.npy')
    softmax_test = np.load('/home/kengo/Documents/GitHub/mmdetection-intermediate/demo/extract/foggy/softmax_city_foggy_train.npy')

    fpr, tpr, thres = roc_curve(flags_test, softmax_test)
    print('softmax', auc(fpr, tpr))

    result0 = adv_det0.batched_detect([logits_test])
    fpr0, tpr0, thres0 = roc_curve(flags_test, result0)
    print('deepgini', auc(fpr0, tpr0))

    result = eval_ood(adv_det, logits_test, preds_test)
    fpr2, tpr2, thres2 = roc_curve(flags_test, result)
    print('proposed', auc(fpr2, tpr2))

    result2 = eval_ood(adv_det2, deepcopy(logits_test[:, :config["num_classes"]]), preds_test)
    fpr3, tpr3, thres3 = roc_curve(flags_test, result2)
    print('openmax', auc(fpr3, tpr3))

    result3 = adv_det3.batched_detect([fcs_test])
    fpr4, tpr4, thres4 = roc_curve(flags_test, result3)
    print('mahal', auc(fpr4, tpr4))

    result4 = adv_det4.batched_detect([logits_test])
    fpr5, tpr5, thres5 = roc_curve(flags_test, result4)
    print('odin', auc(fpr5, tpr5))
    print('\n\n')

    fcs_test = np.load('/home/kengo/Documents/GitHub/mmdetection-intermediate/demo/extract/foggy/fcs_city_foggy_val.npy')
    logits_test = np.load('/home/kengo/Documents/GitHub/mmdetection-intermediate/demo/extract/foggy/logits_city_foggy_val.npy')
    preds_test = np.load('/home/kengo/Documents/GitHub/mmdetection-intermediate/demo/extract/foggy/preds_city_foggy_val.npy')
    flags_test = np.load('/home/kengo/Documents/GitHub/mmdetection-intermediate/demo/extract/foggy/flags_city_foggy_val.npy')
    softmax_test = np.load('/home/kengo/Documents/GitHub/mmdetection-intermediate/demo/extract/foggy/softmax_city_foggy_val.npy')

    fpr, tpr, thres = roc_curve(flags_test, softmax_test)
    print('softmax', auc(fpr, tpr))

    result0 = adv_det0.batched_detect([logits_test])
    fpr0, tpr0, thres0 = roc_curve(flags_test, result0)
    print('deepgini', auc(fpr0, tpr0))

    result = eval_ood(adv_det, logits_test, preds_test)
    fpr2, tpr2, thres2 = roc_curve(flags_test, result)
    print('proposed', auc(fpr2, tpr2))

    result2 = eval_ood(adv_det2, deepcopy(logits_test[:, :config["num_classes"]]), preds_test)
    fpr3, tpr3, thres3 = roc_curve(flags_test, result2)
    print('openmax', auc(fpr3, tpr3))

    result3 = adv_det3.batched_detect([fcs_test])
    fpr4, tpr4, thres4 = roc_curve(flags_test, result3)
    print('mahal', auc(fpr4, tpr4))

    result4 = adv_det4.batched_detect([logits_test])
    fpr5, tpr5, thres5 = roc_curve(flags_test, result4)
    print('odin', auc(fpr5, tpr5))
    print('\n\n')

    fcs_test = np.load(
        '/home/kengo/Documents/GitHub/mmdetection-intermediate/demo/extract/bdd100k/val/fcs_bdd100k_val.npy')
    logits_test = np.load(
        '/home/kengo/Documents/GitHub/mmdetection-intermediate/demo/extract/bdd100k/val/logits_bdd100k_val.npy')
    preds_test = np.load(
        '/home/kengo/Documents/GitHub/mmdetection-intermediate/demo/extract/bdd100k/val/preds_bdd100k_val.npy')
    flags_test = np.load(
        '/home/kengo/Documents/GitHub/mmdetection-intermediate/demo/extract/bdd100k/val/flags_bdd100k_val.npy')
    softmax_test = np.load(
        '/home/kengo/Documents/GitHub/mmdetection-intermediate/demo/extract/bdd100k/val/softmax_bdd100k_val.npy')

    fpr, tpr, thres = roc_curve(flags_test, softmax_test)
    print('softmax', auc(fpr, tpr))

    result0 = adv_det0.batched_detect([logits_test])
    fpr0, tpr0, thres0 = roc_curve(flags_test, result0)
    print('deepgini', auc(fpr0, tpr0))

    result = eval_ood(adv_det, logits_test, preds_test)
    fpr2, tpr2, thres2 = roc_curve(flags_test, result)
    print('proposed', auc(fpr2, tpr2))

    result2 = eval_ood(adv_det2, deepcopy(logits_test[:, :config["num_classes"]]), preds_test)
    fpr3, tpr3, thres3 = roc_curve(flags_test, result2)
    print('openmax', auc(fpr3, tpr3))

    result3 = adv_det3.batched_detect([fcs_test])
    fpr4, tpr4, thres4 = roc_curve(flags_test, result3)
    print('mahal', auc(fpr4, tpr4))

    result4 = adv_det4.batched_detect([logits_test])
    fpr5, tpr5, thres5 = roc_curve(flags_test, result4)
    print('odin', auc(fpr5, tpr5))
    print('\n\n')
