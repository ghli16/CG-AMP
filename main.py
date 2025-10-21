import os
from random import random

import torch
import torch.optim as optim
import numpy as np
from torch import nn

from model import newModel, ContrastiveLoss, PolyLoss
from mydataset import MyDataset, collate_fn1, collate_fn2
from test import test
from utils import *
from get_data import data
from tqdm import tqdm
import warnings
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

seed = 126  # 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
# torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

HIDDEN_UNITS = 10
NUM_HEADS = 6
NUM_LAYERS = 1
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
protein_in_dim = 1280
protein_out_dim = 128
k = 0.96

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


train_set, valid_set, test_set = data('dataset/AMPlify/AMPlify.fasta', 'dataset/AMPlify/amplify_esm2.npy')
# train_set, valid_set, test_set = data('dataset/AMPScanner/AMPScanner.fasta', 'dataset/AMPScanner/ampscanner_esm2.npy')

train_dataset = MyDataset(train_set)
valid_dataset = MyDataset(valid_set)
test_dataset = MyDataset(test_set)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn1)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn2)

model = newModel().to(device)
optimizer = optim.AdamW(model.parameters(), LEARNING_RATE)
contrastive_loss_fn = ContrastiveLoss()
cross_entropy_loss_fn = nn.BCELoss(reduction='none')
# cross_entropy_loss_fn = PolyLoss()
best_mcc = 0
for epoch in tqdm(range(NUM_EPOCHS)):
    loss_all = 0
    y = []
    real = []
    for features1, features2, labels in train_loader:
        model.train()
        optimizer.zero_grad()

        # y_pred = model.trainmodel(features1, features2)
        _, duibi1, _ = model(features1['esm1'], features2['data1'])
        _, duibi2, _ = model(features1['esm2'], features2['data2'])

        y_pred = model.trainmodel(features1['esm'], features2['data'])
        y_pred = y_pred.squeeze(1)
        contrastive_loss = contrastive_loss_fn(duibi1, duibi2, labels['label_'])
        # _loss = cross_entropy_loss_fn(y_pred, labels).mean()
        _loss = cross_entropy_loss_fn(y_pred, labels['labels']).mean()

        loss = _loss + k * contrastive_loss # 0.97
        loss.backward()
        optimizer.step()
        loss_all += loss

    train_loss = loss_all / len(train_loader)
    val_loss = 0
    y_pred_all = []
    real_all = []
    _outputs = []
    model.eval()
    with torch.no_grad():
        for feature_val1, feature_val2, labels_val in valid_loader:
            y_pred_val = model.trainmodel(feature_val1, feature_val2)
            y_pred_all.append(y_pred_val)
            real_all.append(labels_val)
        y_pred_all = torch.cat(y_pred_all, dim=0).cpu().reshape(-1).numpy()
        real_all = torch.cat(real_all, dim=0).cpu().reshape(-1).numpy()
        metric_tmp = get_metrics(real_all, y_pred_all)

        # if (epoch + 1) % 10 == 0 or epoch == 0:
        print('epoch  {}: train_Loss: {}'.format(epoch + 1, train_loss))
        print(
            'vaildset: f1:{} acc:{} recall:{} precision:{} mcc:{} roc_auc:{}'.format(metric_tmp[0], metric_tmp[1],
                                                                          metric_tmp[2],
                                                                          metric_tmp[3], metric_tmp[
                                                                              4], metric_tmp[5]))
        mcc = float(metric_tmp[4])
        # 保存最佳模型参数
        if mcc > best_mcc:
            best_mcc = mcc
            torch.save(model.state_dict(), "model.pth")

test(test_loader)





