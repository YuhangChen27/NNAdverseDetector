import numpy as np

from nnadvdet.core.maxprob import MaxProb
from nnadvdet.core.ginicoef import DeepGini
from nnadvdet.core.reliable_bound import ReliableBound
from nnadvdet.core.openmax import OpenmaxLibmr
from torchvision import transforms
from util.custom_transform import Translation2D
from copy import deepcopy
from torch.utils.data import DataLoader
from util.load_data import SelectDataset
from nnadvdet.frontend.hook import LinearHook
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


def build_translate_loader(input_size, factor, dataset='MNIST', batch_size=1000):
    translation_range = int(input_size * factor)
    loaders = {}
    for i in range(-translation_range, translation_range+1):
        for j in range(-translation_range, translation_range+1):
            trans_t = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(input_size),
                Translation2D(i, j)
            ])
            data_t = SelectDataset(dataset, None, train=True, transform=trans_t)
            loaders[f'{i}{j}'] = DataLoader(data_t.dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return loaders


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_name = 'LeNet5'
num_classes = 10
input_size = 32
model = torch.load('../checkpoints/mnist_lenet5.pth')
model.to(device)
model.eval()
# loader_set = build_translate_loader(input_size, 0.1, batch_size=1000)
# # hook_fc = LinearHook(module=model.fc2)
#
# # fcs = []
# logits = []
# preds = []
# label_collect = []
#
#
# pbar = tqdm(total=len(loader_set)*len(loader_set['00']))
# for idx, loader in enumerate(loader_set):
#     for inputs, labels in loader_set[loader]:
#         inputs = inputs.to(device)
#         labels = labels.to(device)
#
#         logit_tmp = model(inputs)
#         # hook_fc.calc()
#         # fc = hook_fc.noact_output.cpu().numpy().tolist()
#         _, pred_tmp = torch.max(logit_tmp, 1)
#         label_tmp = labels.data
#
#         # fcs.extend(fc)
#         logits.extend(deepcopy(logit_tmp.detach().cpu().numpy().tolist()))
#         preds.extend(deepcopy(pred_tmp.cpu().numpy().tolist()))
#         label_collect.extend(deepcopy(label_tmp.cpu().numpy().tolist()))
#         pbar.update(1)
# pbar.close()

# fcs = np.asarray(fcs, dtype=np.float32)
# logits = np.asarray(logits, dtype=np.float32)
# preds = np.asarray(preds, dtype=int)
# label_collect = np.asarray(label_collect, dtype=int)
#
# np.save('mnist_logits.npy', logits)
# np.save('mnist_preds.npy', preds)
# np.save('mnist_labels.npy', label_collect)

logits = np.load('mnist_logits.npy')
preds = np.load('mnist_preds.npy')
label_collect = np.load('mnist_labels.npy')


adv_det0 = MaxProb(config)
adv_det1 = DeepGini(config)
adv_det2 = ReliableBound(config)

adv_det0.build_detector([logits], preds, label_collect)
adv_det1.build_detector([logits], preds, label_collect)
adv_det2.build_detector([logits], preds, label_collect)


trans_t2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(input_size)
])
data_t2 = SelectDataset('MNIST', None, train=False, transform=trans_t2)
loader_test = DataLoader(data_t2.dataset, batch_size=10000, shuffle=False, num_workers=1)
# fcs_test = []
logits_test = []
preds_test = []
label_test = []
pbar2 = tqdm(total=len(loader_test))
for inputs, labels in loader_test:
    inputs = inputs.to(device)
    labels = labels.to(device)

    logit_tmp = model(inputs)
    # hook_fc.calc()
    # fc = hook_fc.noact_output.cpu().numpy().tolist()
    _, pred_tmp = torch.max(logit_tmp, 1)
    label_tmp = labels.data

    # fcs_test.extend(fc)
    logits_test.extend(deepcopy(logit_tmp.detach().cpu().numpy().tolist()))
    preds_test.extend(deepcopy(pred_tmp.cpu().numpy().tolist()))
    label_test.extend(deepcopy(label_tmp.cpu().numpy().tolist()))
    pbar2.update(1)
pbar2.close()
# fcs_test = np.asarray(fcs_test, dtype=np.float32)
logits_test = np.asarray(logits_test, dtype=np.float32)
preds_test = np.asarray(preds_test, dtype=int)
label_test = np.asarray(label_test, dtype=int)
flag_test = np.zeros(logits_test.shape[0], dtype=int)
flag_test[preds_test == label_test] = 1

result2 = []
pbar3 = tqdm(total=logits_test.shape[0])
for i in range(logits_test.shape[0]):
    result2.append(adv_det2.detect(logits_test[i].reshape(1, -1), x_pred=preds_test[i]))
    pbar3.update(1)
pbar3.close()
result2 = np.asarray(result2)
print()
fpr2, tpr2, thres2 = roc_curve(flag_test, result2)
print(auc(fpr2, tpr2))
