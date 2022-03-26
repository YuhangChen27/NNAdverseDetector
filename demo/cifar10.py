import gc

import numpy as np

from nnadvdet.core.maxprob.maxprob import MaxProb
from nnadvdet.core.ginicoef.ginicoef import DeepGini
from nnadvdet.core.reliable_bound import ReliableBound
from torchvision import transforms
from util.custom_transform import Translation2D
from copy import deepcopy
from torch.utils.data import DataLoader
from util.load_data import SelectDataset
import torch
from tqdm import tqdm
from sklearn.metrics import auc, roc_curve


config = {
    'num_classes': 10,
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


def build_translate_loader(input_size, factor, dataset='MNIST', batch_size=1000, step=1):
    translation_range = int(input_size * factor)
    num_move = translation_range // step
    loaders = {}
    for i in range(-num_move, num_move+1):
        for j in range(-num_move, num_move+1):
            trans_t = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                transforms.Resize(input_size),
                Translation2D(i*step, j*step)
            ])
            data_t = SelectDataset(dataset, None, train=True, transform=trans_t)
            loaders[f'{i}{j}'] = DataLoader(data_t.dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return loaders


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_name = 'LeNet5'
    num_classes = 10
    input_size = 224
    model = torch.load('../checkpoints/cifar10_vgg16.pth')
    model.to(device)
    model.eval()
    # loader_set = build_translate_loader(input_size, 0.05, dataset='CIFAR10', batch_size=32, step=4)
    #
    # logits = []
    # preds = []
    # label_collect = []
    #
    # pbar = tqdm(total=len(loader_set)*len(loader_set['00']))
    # for idx, loader in enumerate(loader_set):
    #     for inputs, labels in loader_set[loader]:
    #         inputs = inputs.to(device)
    #         labels = labels.to(device)
    #
    #         logit_tmp = model(inputs)
    #         _, pred_tmp = torch.max(logit_tmp, 1)
    #         label_tmp = labels.data
    #
    #         logits.extend(deepcopy(logit_tmp.detach().cpu().numpy().tolist()))
    #         preds.extend(deepcopy(pred_tmp.cpu().numpy().tolist()))
    #         label_collect.extend(deepcopy(label_tmp.cpu().numpy().tolist()))
    #         pbar.update(1)
    # pbar.close()
    # del loader_set
    # gc.collect()
    #
    # logits = np.asarray(logits, dtype=np.float32)
    # preds = np.asarray(preds, dtype=int)
    # label_collect = np.asarray(label_collect, dtype=int)
    #
    # np.save('cifar10_logits.npy', logits)
    # np.save('cifar10_preds.npy', preds)
    # np.save('cifar10_labels.npy', label_collect)
    logits = np.load('cifar10_logits.npy')
    preds = np.load('cifar10_preds.npy')
    label_collect = np.load('cifar10_labels.npy')

    adv_det0 = MaxProb(config)
    adv_det1 = DeepGini(config)
    adv_det2 = ReliableBound(config)

    adv_det0.build_detector([logits], preds, label_collect)
    adv_det1.build_detector([logits], preds, label_collect)
    adv_det2.build_detector([logits], preds, label_collect)


    trans_t2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.Resize(input_size)
    ])
    data_t2 = SelectDataset('CIFAR10', None, transform=trans_t2, train=False)
    loader_test = DataLoader(data_t2.dataset, batch_size=32, shuffle=False, num_workers=1)
    logits_test = []
    preds_test = []
    label_test = []
    pbar = tqdm(total=len(loader_test))
    for inputs, labels in loader_test:
        inputs = inputs.to(device)
        labels = labels.to(device)

        logit_tmp = model(inputs)
        _, pred_tmp = torch.max(logit_tmp, 1)
        label_tmp = labels.data

        logits_test.extend(deepcopy(logit_tmp.detach().cpu().numpy().tolist()))
        preds_test.extend(deepcopy(pred_tmp.cpu().numpy().tolist()))
        label_test.extend(deepcopy(label_tmp.cpu().numpy().tolist()))
        pbar.update(1)
    pbar.close()
    logits_test = np.asarray(logits_test, dtype=np.float32)
    preds_test = np.asarray(preds_test, dtype=int)
    label_test = np.asarray(label_test, dtype=int)
    flag_test = np.zeros(logits_test.shape[0], dtype=int)
    flag_test[preds_test == label_test] = 1
    print(flag_test.sum())

    result0 = adv_det0.batched_detect([logits_test])
    result1 = adv_det1.batched_detect([logits_test])
    fpr0, tpr0, thres0 = roc_curve(flag_test, result0)
    fpr1, tpr1, thres1 = roc_curve(flag_test, result1)
    print(auc(fpr0, tpr0))
    print(auc(fpr1, tpr1))

    result2 = []
    pbar = tqdm(total=logits_test.shape[0])
    for i in range(logits_test.shape[0]):
        result2.append(adv_det2.detect(logits_test[i].reshape(1, -1), x_pred=preds_test[i]))
        pbar.update(1)
    pbar.close()
    result2 = np.array(result2)
    print()
    fpr2, tpr2, thres2 = roc_curve(flag_test, result2)
    print(auc(fpr2, tpr2))

    data_ood = SelectDataset('CIFAR100', None, transform=trans_t2, train=False)
    loader_ood = DataLoader(data_ood.dataset, batch_size=32, shuffle=False, num_workers=1)
    logits_ood = []
    preds_ood = []
    pbar = tqdm(total=len(loader_ood))
    for inputs, labels in loader_ood:
        inputs = inputs.to(device)
        labels = labels.to(device)

        logit_tmp = model(inputs)
        _, pred_tmp = torch.max(logit_tmp, 1)
        label_tmp = labels.data

        logits_ood.extend(deepcopy(logit_tmp.detach().cpu().numpy().tolist()))
        preds_ood.extend(deepcopy(pred_tmp.cpu().numpy().tolist()))
        pbar.update(1)
    pbar.close()
    logits_ood = np.asarray(logits_ood, dtype=np.float32)
    preds_ood = np.asarray(preds_ood, dtype=int)
    logits_ood = logits_ood[:2000]
    preds_ood = preds_ood[:2000]

    mix_logits = np.concatenate((logits_test, logits_ood), axis=0)
    mix_pred = np.concatenate((preds_test, preds_ood), axis=0)
    mix_flag = np.concatenate((flag_test, np.zeros(logits_ood.shape[0], dtype=int)), axis=0)
    result3 = adv_det0.batched_detect([mix_logits])
    result4 = adv_det1.batched_detect([mix_logits])
    fpr3, tpr3, thres3 = roc_curve(mix_flag, result3)
    fpr4, tpr4, thres4 = roc_curve(mix_flag, result4)
    print(auc(fpr3, tpr3))
    print(auc(fpr4, tpr4))

    result5 = []
    pbar = tqdm(total=mix_logits.shape[0])
    for i in range(mix_logits.shape[0]):
        result5.append(adv_det2.detect(mix_logits[i].reshape(1, -1), x_pred=mix_pred[i]))
        pbar.update(1)
    pbar.close()
    result5 = np.array(result5)
    fpr5, tpr5, thres5 = roc_curve(mix_flag, result5)
    print(auc(fpr5, tpr5))
