import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('..')
from config import params
from models import *
from utils import utils, torch_utils

from sklearn.metrics import average_precision_score
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score
# ----------------------------------------------------------------------- # 

class BCLoss(nn.Module):
    # binary classification loss
    def __init__(self, args, pr, device):
        super(BCLoss, self).__init__()
        self.pr = pr
        # self.class_dist = pr.class_dist
        self.class_info = pr.class_info
        self.device = device
        self.pos_weight = torch.tensor(pr.data_weight).to(device)
        self.criterion = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=self.pos_weight)

    def forward(self, pred, target):
        # import pdb; pdb.set_trace()
        loss = self.criterion(pred, target.float())
        return loss

    def evaluate(self, pred, target):
        loss = self.criterion(pred, target.float())
        pred = torch.sigmoid(pred)
        pred = pred.data.cpu().numpy()
        target = target.data.cpu().numpy()
        ap = average_precision_score(target, pred)
        acc = torch_utils.binary_acc(pred, target, thred=0.5)
        res = {
            'Loss': loss.item(),
            'AP': ap,
            'Acc': acc
        }
        return res

    def _generate_class_weight(self):
        # import pdb; pdb.set_trace()
        class_dist = torch.from_numpy(self.class_dist).float()
        pos_weights = torch.tensor([class_dist[0] / class_dist[1]])
        return pos_weights


# ----------------------------------------------------------------------- # 

class MCLoss(nn.Module):
    # multi-classification loss
    def __init__(self, args, pr, device):
        super(MCLoss, self).__init__()
        self.pr = pr
        # self.class_dist = pr.class_dist
        self.class_info = pr.class_info
        self.device = device
        self.class_weight = torch.tensor(pr.data_weight).to(device)
        self.criterion = nn.CrossEntropyLoss(reduction='mean', weight=self.class_weight.float())
    
    def forward(self, pred, target):
        # import pdb; pdb.set_trace()
        loss = self.criterion(pred, target.long())
        return loss

    def evaluate(self, pred, target):
        loss = self.criterion(pred, target.long())
        pred = F.softmax(pred, dim=-1)
        top_1_acc = torch_utils.calc_acc(pred, target, k=1)
        top_5_acc = torch_utils.calc_acc(pred, target, k=5)
        micro_f1 = self.calc_f1(pred, target)
        avg_distance = self.calc_avgdis(pred, target)
        res =  {
            'Top-1 acc': top_1_acc.item(),
            'Top-5 acc': top_5_acc.item(),
            'micro_f1': micro_f1,
            'Avg Dis.': avg_distance
        } 
        return res
    
    def calc_f1(self, pred, target):
        pred = torch.argmax(pred, dim=-1) 
        pred = pred.data.cpu().numpy()
        target = target.data.cpu().numpy()
        f1 = f1_score(target, pred, average='micro')
        return f1

    def calc_avgdis(self, pred, target, unit=1):
        pred = torch.argmax(pred, dim=-1) 
        pred = pred.data.cpu().numpy()
        target = target.data.cpu().numpy()
        avg_dis = np.abs(pred - target) * unit
        avg_dis = np.mean(avg_dis)
        return avg_dis


# ----------------------------------------------------------------------- # 


if __name__ == '__main__':
    pass


