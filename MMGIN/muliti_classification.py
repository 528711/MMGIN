#!/usr/bin/env python
# # Import
from chemprop.features import morgan_binary_features_generator
# In[7]:


from numpy.random import seed
import sqlite3
import time
import numpy as np
import random
import pandas as pd
from pandas import DataFrame
import math
import matplotlib.pyplot as plt
from rdkit import Chem

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc, average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize

import warnings
import os
import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric.data import Data
from tqdm import tqdm
from torch_geometric.nn import global_mean_pool as gap,global_max_pool as gmp

from pubchemfp import GetPubChemFPs
from utils import *
seed = 1
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 使用GPU编号，可以根据需要更改


file_path = "./"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


learn_rating = 1e-3
weight_decay_rate = 1e-3
patience = 7
delta = 0
label_num_multi = 5
label_num_binary = 1


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        # print("val_loss={}".format(val_loss))
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")
        torch.save(model.state_dict(), path + "/" + "model_checkpoint.pth")
        self.val_loss_min = val_loss


# https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
from torch.optim.optimizer import Optimizer


class Ranger(Optimizer):

    def __init__(self, params, lr=1e-3,  # lr
                 alpha=0.5, k=6, N_sma_threshhold=5,  # Ranger options
                 betas=(.95, 0.999), eps=1e-5, weight_decay=0,  # Adam options
                 # Gradient centralization on or off, applied to conv layers only or conv + fc layers
                 use_gc=True, gc_conv_only=False
                 ):

        # parameter checks
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        if not lr > 0:
            raise ValueError(f'Invalid Learning Rate: {lr}')
        if not eps > 0:
            raise ValueError(f'Invalid eps: {eps}')

        # prep defaults and init torch.optim base
        defaults = dict(lr=lr, alpha=alpha, k=k, step_counter=0, betas=betas,
                        N_sma_threshhold=N_sma_threshhold, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # adjustable threshold
        self.N_sma_threshhold = N_sma_threshhold

        # look ahead params

        self.alpha = alpha
        self.k = k

        # radam buffer for state
        self.radam_buffer = [[None, None, None] for ind in range(10)]

        # gc on or off
        self.use_gc = use_gc

        # level of gradient centralization
        self.gc_gradient_threshold = 3 if gc_conv_only else 1

        print(
            f"Ranger optimizer loaded. \nGradient Centralization usage = {self.use_gc}")
        if (self.use_gc and self.gc_gradient_threshold == 1):
            print(f"GC applied to both conv and fc layers")
        elif (self.use_gc and self.gc_gradient_threshold == 3):
            print(f"GC applied to conv layers only")

    def __setstate__(self, state):
        print("set state called")
        super(Ranger, self).__setstate__(state)

    def step(self, closure=None):
        loss = None

        # Evaluate averages and grad, update param tensors
        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()

                if grad.is_sparse:
                    raise RuntimeError(
                        'Ranger optimizer does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]  # get state dict for this param

                if len(state) == 0:

                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)

                    # look ahead weight storage now in state dict
                    state['slow_buffer'] = torch.empty_like(p.data)
                    state['slow_buffer'].copy_(p.data)

                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(
                        p_data_fp32)

                # begin computations
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # GC operation for Conv layers and FC layers
                if grad.dim() > self.gc_gradient_threshold:
                    grad.add_(-grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True))

                state['step'] += 1

                # compute variance mov avg
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                # compute mean moving avg
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                buffered = self.radam_buffer[int(state['step'] % 10)]

                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma
                    if N_sma > self.N_sma_threshhold:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (
                                N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay']
                                     * group['lr'], p_data_fp32)

                # apply lr
                if N_sma > self.N_sma_threshhold:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size *
                                         group['lr'], exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)

                p.data.copy_(p_data_fp32)

                # integrated look ahead...
                # we do it at the param level instead of group level
                if state['step'] % group['k'] == 0:
                    # get access to slow param tensor
                    slow_p = state['slow_buffer']
                    # (fast weights - slow weights) * alpha
                    slow_p.add_(self.alpha, p.data - slow_p)
                    # copy interpolated weights to RAdam param tensor
                    p.data.copy_(slow_p)

        return loss

class DDIDataset(Dataset):
    def __init__(self, x, y_multi,y_binary):
        self.len = len(x)
        self.x_data=x
        self.y_multi_data=y_multi
        self.y_binary_data=y_binary
    def __getitem__(self, index):
        return self.x_data[index], self.y_multi_data[index], self.y_binary_data[index]

    def __len__(self):
        return self.len


from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv, GCNConv, GINConv, global_add_pool


class MultiTaskModel(nn.Module):
    def __init__(self, num_classes_multi, num_classes_binary):
        super(MultiTaskModel, self).__init__()
        num_features_xd = 78
        dropout = 0.2
        dim = 256
        dim1 = 512
        dim2 = 256
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        # convolution layers
        nn1 = Sequential(Linear(num_features_xd, dim), ReLU(), Linear(dim, dim1))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim1)

        nn2 = Sequential(Linear(dim1, dim1), ReLU(), Linear(dim1, dim1))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim1)

        nn3 = Sequential(Linear(dim1, dim1), ReLU(), Linear(dim1, dim1))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim1)

        self.fc_1 = Linear(dim1, dim2)

        self.fp_1_dim = 1024
        self.fp_2_dim = 128
        self.fp_3_dim = 256

        self.fp_type = 'mixed'

        if self.fp_type == 'mixed':
            self.fp_dim = 1489
        else:
            self.fp_dim = 1024

        self.fc1 = nn.Linear(self.fp_dim, self.fp_2_dim)
        self.fc2 = nn.Linear(self.fp_2_dim, self.fp_3_dim)
        self.act_func = nn.ReLU()

        # 多分类任务的输出层
        self.fc_multi = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes_multi)
        )

    def forward(self, data):
        x, finger, edge_index, batch = data.x, data.finger, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)

        x = global_add_pool(x, batch)
        x = F.relu(self.fc_1(x))
        x = F.dropout(x, p=0.2, training=self.training)

        fp_list = []
        for i, one in enumerate(finger):
            fp = []
            mol = Chem.MolFromSmiles(one)
            if self.fp_type == 'mixed':
                fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)
                fp_phaErGfp = AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath=21, minPath=1)
                fp_pubcfp = GetPubChemFPs(mol)
                fp.extend(fp_maccs)
                fp.extend(fp_phaErGfp)
                fp.extend(fp_pubcfp)
            else:
                fp_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                fp.extend(fp_morgan)
            fp_list.append(fp)
        fp_list = torch.Tensor(fp_list).to(device)
        fpn_out = self.fc1(fp_list)
        fpn_out = self.act_func(fpn_out)
        fpn_out = self.fc2(fpn_out)
        # fpn_out = self.dropout(fpn_out)
        fpn_out = self.act_func(fpn_out)

        # concat
        xc = torch.cat((x, fpn_out), dim=1)

        # 多分类任务的预测
        y_multi = self.fc_multi(xc)

        return y_multi


class MultiClassLoss(nn.Module):
    def __init__(self, gamma=2):
        super(MultiClassLoss, self).__init__()
        self.gamma = gamma

    def forward(self, preds, labels):
        labels = labels.view(-1, 1).type(torch.int64)  # [B * S, 1]
        preds = preds.view(-1, preds.size(-1))  # [B * S, C]

        preds_logsoft = F.log_softmax(preds, dim=1)  # 先softmax, 然后取log
        preds_softmax = torch.exp(preds_logsoft)  # softmax
        preds_softmax = preds_softmax.gather(1, labels)  # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1, labels)

        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = loss.mean()

        return loss
#
# class BinaryClassLoss(nn.Module):
#     def __init__(self):
#         super(BinaryClassLoss, self).__init__()
#
#     def forward(self, pred, target):
#
#         target = target.view(-1, 1)
#         # pred是模型的输出，通常是经过Sigmoid函数的结果
#         # target是二分类任务的真实标签，通常是0或1
#         criterion = nn.BCELoss()
#         loss = criterion(pred, target.float())  # 将target转换为浮点数
#         return loss

# # Training

# In[ ]:


from torch.optim.lr_scheduler import CosineAnnealingLR
train_epochs_loss = []
valid_epochs_loss = []
def val():
    TRAIN_BATCH_SIZE = 128
    TEST_BATCH_SIZE = 128
    epochs=700
    model=MultiTaskModel(label_num_multi,label_num_binary)
    optimizer = Ranger(model.parameters(), lr=learn_rating, weight_decay=weight_decay_rate, betas=(0.95, 0.999),eps=1e-6)
    train_data = TestbedDataset(root='./feng/train/', path='train_graph_dataset.csv')
    test_data = TestbedDataset(root='./feng/test/', path='test_graph_dataset.csv')
    train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

    result_all,result_eve=train_fn(model,optimizer,train_loader,test_loader,epochs)
    return result_all,result_eve


def train_fn(model,optimizer,train_loader,test_loader,epochs):

    model = model.to(device)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-3)
    my_loss_multi = MultiClassLoss()
    earlystop = EarlyStopping(patience=patience, delta=delta)
    running_losses=[]
    y_true_multi=np.array([])
    y_score_multi=np.zeros((0,label_num_multi),dtype=float)
    y_pred_multi = np.array([])
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for i,batch_data in enumerate(train_loader):
            batch_data = batch_data.to(device)
            outputs_multi_train = model(batch_data)
            label_multi_train=batch_data.y_multi
            # 计算多分类任务的损失
            loss_multi = my_loss_multi(outputs_multi_train, label_multi_train)
            optimizer.zero_grad()
            loss = loss_multi
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        testing_loss = 0.0
        with torch.no_grad():
            for i,batch_data in enumerate(test_loader):
                batch_data = batch_data.to(device)
                outputs_multi_test= model(batch_data)
                label_multi_test = batch_data.y_multi
                loss_multi = my_loss_multi(outputs_multi_test, label_multi_test)
                loss = loss_multi
                testing_loss += loss.item()
                scheduler.step(loss)

        print("epoch [%d] loss: %.6f testing_loss: %.6f "% (epoch + 1, running_loss / len(train_loader.dataset), testing_loss / len(test_loader.dataset)))
        running_losses.append(running_loss)
        train_epochs_loss.append(running_loss / len(train_loader.dataset))
        valid_epochs_loss.append(testing_loss / len(test_loader.dataset))
        earlystop(valid_epochs_loss[-1], model, file_path)
        if earlystop.early_stop:
            print("Early stopping\n")
            break


    pre_score_multi = np.zeros((0, label_num_multi), dtype=float)

    model.eval()
    with torch.no_grad():
        for i, batch_data in enumerate(test_loader):
            batch_data = batch_data.to(device)
            outputs_multi = model(batch_data)
            # print(data.y_multi.view(-1,1))
            label_multi_test=batch_data.y_multi
            pre_score_multi = np.vstack((pre_score_multi, F.softmax(outputs_multi).cpu().numpy()))
            y_true_multi = np.hstack((y_true_multi, label_multi_test.cpu().numpy()))

        # 多分类任务的评价
        pred_type_multi = np.argmax(pre_score_multi, axis=1)  # pred_type_multi预测标签
        y_pred_multi = np.hstack((y_pred_multi, pred_type_multi))
        y_score_multi = np.row_stack((y_score_multi, pre_score_multi))
        result_all, result_eve = evaluate(y_pred_multi, y_score_multi, y_true_multi, label_num_multi)



    return result_all,result_eve


# ## start training
def roc_aupr_score(y_true, y_score, average="macro"):
    def _binary_roc_aupr_score(y_true, y_score):
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        return auc(recall, precision)

    def _average_binary_score(
            binary_metric, y_true, y_score, average
    ):  # y_true= y_one_hot
        if average == "binary":
            return binary_metric(y_true, y_score)
        if average == "micro":
            y_true = y_true.ravel()
            y_score = y_score.ravel()
        if y_true.ndim == 1:
            y_true = y_true.reshape((-1, 1))
        if y_score.ndim == 1:
            y_score = y_score.reshape((-1, 1))
        n_classes = y_score.shape[1]
        score = np.zeros((n_classes,))
        for c in range(n_classes):
            y_true_c = y_true.take([c], axis=1).ravel()
            y_score_c = y_score.take([c], axis=1).ravel()
            score[c] = binary_metric(y_true_c, y_score_c)
        return np.average(score)

    return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)
# In[ ]:
def evaluate(y_pred, y_score, y_true, label_num):
    all_eval_type = 11
    result_all = np.zeros((all_eval_type, 1), dtype=float)
    each_eval_type = 6
    result_eve = np.zeros((label_num, each_eval_type), dtype=float)
    y_one_hot = label_binarize(y_true, classes=range(label_num))
    pred_one_hot = label_binarize(y_pred, classes=range(label_num))
    result_all[0] = accuracy_score(y_true, y_pred)
    print('acc',result_all[0])
    result_all[1] = roc_aupr_score(y_one_hot, y_score, average="micro")
    print('aupr_micro', result_all[1])
    result_all[2] = roc_aupr_score(y_one_hot, y_score, average="macro")
    print('aupr_macro', result_all[2])
    result_all[3] = roc_auc_score(y_one_hot, y_score, average="micro")
    print('auc_micro', result_all[3])
    result_all[4] = roc_auc_score(y_one_hot, y_score, average="macro")
    print('auc_macro', result_all[4])
    result_all[5] = f1_score(y_true, y_pred, average="micro")
    print('f1_micro', result_all[5])
    result_all[6] = f1_score(y_true, y_pred, average="macro")
    print('f1_macro', result_all[6])
    result_all[7] = precision_score(y_true, y_pred, average="micro")
    print('precision_micro', result_all[7])
    result_all[8] = precision_score(y_true, y_pred, average="macro")
    print('precision_macro', result_all[8])
    result_all[9] = recall_score(y_true, y_pred, average="micro")
    print('recall_micro', result_all[9])
    result_all[10] = recall_score(y_true, y_pred, average="macro")
    print('recall_macro', result_all[10])
    for i in range(label_num_multi):
        result_eve[i, 0] = accuracy_score(
            y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel()
        )
        result_eve[i, 1] = roc_aupr_score(
            y_one_hot.take([i], axis=1).ravel(),
            pred_one_hot.take([i], axis=1).ravel(),
            average=None,
        )
        result_eve[i, 2] = roc_auc_score(
            y_one_hot.take([i], axis=1).ravel(),
            pred_one_hot.take([i], axis=1).ravel(),
            average=None,
        )
        result_eve[i, 3] = f1_score(
            y_one_hot.take([i], axis=1).ravel(),
            pred_one_hot.take([i], axis=1).ravel(),
            average="binary",
        )
        result_eve[i, 4] = precision_score(
            y_one_hot.take([i], axis=1).ravel(),
            pred_one_hot.take([i], axis=1).ravel(),
            average="binary",
        )
        result_eve[i, 5] = recall_score(
            y_one_hot.take([i], axis=1).ravel(),
            pred_one_hot.take([i], axis=1).ravel(),
            average="binary",
        )
    return [result_all, result_eve]

def save_result(result_type, result):
    index = ['accuracy', 'aupr_micro', 'aupr_macro', 'auc_micro', 'auc_macro', 'f1_micro', 'f1_macro',
             'precision_micro', 'precision_macro', 'recall_micro', 'recall_macro']

    if result_type == 'all':
        all_ = pd.DataFrame(result, index=index)
        all_.to_csv('./results_all_multi_multi_classfication.csv')
    else:
        each = pd.DataFrame(result)
        each.to_csv('./results_each_multi_multi_classfication.csv', index=False)



if __name__ == '__main__':

    result_all,result_eve=val()
    save_result('all',result_all)
    save_result('each',result_eve)
    index = ['accuracy', 'aupr_micro', 'aupr_macro', 'auc_micro', 'auc_macro', 'f1_micro', 'f1_macro',
             'precision_micro',
             'precision_macro', 'recall_micro', 'recall_macro']
    all_ = pd.DataFrame(result_all, index=index)
    each = pd.DataFrame(result_eve)
    each.columns = ['accuracy', 'aupr', 'auc', 'f1', 'precision', 'recall']
