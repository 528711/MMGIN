#!/usr/bin/env python
# # Import
from chemprop.features import morgan_binary_features_generator


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
from sklearn import metrics

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc, average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.neighbors import KNeighborsClassifier
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
from torch_geometric.nn import GATConv
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
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 


file_path = "./"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label_num_multi = 5
label_num_binary = 1

from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score,recall_score,f1_score,average_precision_score

train_epochs_loss = []
valid_epochs_loss = []
def val():
    TRAIN_BATCH_SIZE = 128
    TEST_BATCH_SIZE = 128

    train_data = TestbedDataset(root='./feng/train/', path='train_graph_dataset.csv')
    test_data = TestbedDataset(root='./feng/test/', path='test_graph_dataset.csv')
    train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

    for i,batch_data in enumerate(train_loader):
        batch_data = batch_data.to(device)
        # print('batch_data',batch_data)
        # outputs_multi_train, outputs_bin_train = model(batch_data)
        # print(data.y_multi.view(-1,1))
        smiles_train=batch_data.finger
        fp_list_train = []
        for i, one in enumerate(smiles_train):
            fp = []
            mol = Chem.MolFromSmiles(one)
            fp_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            fp.extend(fp_morgan)
            fp_list_train.append(fp)
        fp_list_train = torch.Tensor(fp_list_train)
        fp_list_train=np.array(fp_list_train)

        label_multi_train=batch_data.y_multi
        label_multi_train=label_multi_train.cpu()
        label_multi_train=label_multi_train.numpy()
        label_bin_train = batch_data.y_bin
        label_bin_train = label_bin_train.cpu()
        label_bin_train = label_bin_train.numpy()
        
        LR_multi = LogisticRegression(multi_class='multinomial',solver='lbfgs')
        LR_multi.fit(fp_list_train, label_multi_train)

        LR_bin = LogisticRegression()
        LR_bin.fit(fp_list_train, label_bin_train)

    y_true_multi = np.array([])
    y_true_bin = np.array([])
    y_score_multi = np.zeros((0, label_num_multi), dtype=float)
    y_score_bin = np.zeros((0, label_num_binary), dtype=float)
    y_pred_multi = np.array([])
    y_pred_bin = np.array([])

    for i, batch_data in enumerate(test_loader):
        batch_data = batch_data.to(device)
        smiles_test = batch_data.finger
        fp_list_test = []
        for i, one in enumerate(smiles_test):
            fp = []
            mol = Chem.MolFromSmiles(one)
            fp_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            fp.extend(fp_morgan)
            fp_list_test.append(fp)
        fp_list_test = torch.Tensor(fp_list_test)
        fp_list_test = np.array(fp_list_test)
        label_multi_test = batch_data.y_multi
        label_multi_test = label_multi_test.cpu()
        label_multi_test = label_multi_test.numpy()

        label_bin_test = batch_data.y_bin
        label_bin_test = label_bin_test.cpu()
        label_bin_test = label_bin_test.numpy()

    y_score_multi = np.zeros((0, label_num_multi), dtype=float)

    pre_score_multi = LR_multi.predict_proba(fp_list_test)
    y_pred_multi = np.argmax(pre_score_multi, axis=1) 
    y_score_multi = np.row_stack((y_score_multi, pre_score_multi))
    result_all, result_eve = evaluate(y_pred_multi, y_score_multi, label_multi_test, label_num_multi)

    pred_bin = LR_bin.predict(fp_list_test)
    accuracy_bin = metrics.accuracy_score(label_bin_test, pred_bin)
    f1_bin_micro = f1_score(label_bin_test, pred_bin, average='micro')
    f1_bin_macro = f1_score(label_bin_test, pred_bin, average='macro')
    precision_bin_micro = precision_score(label_bin_test, pred_bin, average='micro')
    precision_bin_macro = precision_score(label_bin_test, pred_bin, average='macro')
    recall_bin_micro = recall_score(label_bin_test, pred_bin, average='micro')
    recall_bin_macro = recall_score(label_bin_test, pred_bin, average='macro')
    print('二分类')
    print('输出结果', metrics.classification_report(label_bin_test, pred_bin))
    print('acc', accuracy_bin)
    print('f1_bin_micro', f1_bin_micro)
    print('f1_bin_macro', f1_bin_macro)
    print('precision_bin_micro', precision_bin_micro)
    print('precision_bin_macro', precision_bin_macro)
    print('recall_bin_micro', recall_bin_micro)
    print('recall_bin_macro', recall_bin_macro)
    return result_all, result_eve
   


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
    print('precision_micro', result_all[6])
    result_all[8] = precision_score(y_true, y_pred, average="macro")
    print('precision_macro', result_all[7])
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
        all_.to_csv('./results_all_multi_LR.csv')
    else:
        each = pd.DataFrame(result)
        each.to_csv('./results_each_multi_LR.csv', index=False)



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
