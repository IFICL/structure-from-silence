import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import torchaudio
import librosa

import sys
sys.path.append('..')
from utils import sourcesep
from config import params
import models


def conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=None):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01),
        nn.LeakyReLU(inplace=True)
    )
# -------------------------------------------------------------------------------------- # 


class VisionEncoder(nn.Module):
    # base encoder
    def __init__(self, args, pr, device=None):
        super(VisionEncoder, self).__init__()
        self.pr = pr
        self.num_classes = pr.num_classes
    

class VisionADNet(VisionEncoder):
    # Audio Relative Depth Net
    def __init__(self, args, pr, output_dim=None, device=None, backbone=None):
        super(VisionADNet, self).__init__(args, pr, device)
        if not backbone:
            backbone = args.backbone
        dim_out = output_dim if output_dim else pr.num_classes
        network = getattr(models, backbone)
        if backbone.find('resnet') != -1:
            in_channels = 3
            self.net = network(pretrained=args.pretrained, input_channel=in_channels)
            self.net.fc = nn.Linear(self.net.fc.in_features, dim_out)
    
    def forward(self, img, return_permute=False, return_feat=False):
        # import pdb; pdb.set_trace()
        ''' 
            img_1: (N, C, H, W)

        '''
        output = self.net(img).squeeze(1)
        return output

# -------------------------------------------------------------------------------------- # 

class VisionRDNet(VisionEncoder):
    # Audio Relative Depth Net
    def __init__(self, args, pr, output_dim=None, device=None, backbone=None):
        super(VisionRDNet, self).__init__(args, pr, device)
        if not backbone:
            backbone = args.backbone
        network = getattr(models, backbone)
        if backbone.find('resnet') != -1:
            in_channels = 3
            self.net = network(pretrained=args.pretrained, input_channel=in_channels)
            self.net.fc = nn.Linear(self.net.fc.in_features, pr.feat_dim)
        
        dim_out = output_dim if output_dim else pr.num_classes
        self.fusion = nn.Sequential(
            nn.Linear(2 * pr.feat_dim, pr.feat_dim),
            nn.ReLU(True),
            nn.Linear(pr.feat_dim, dim_out)
        )
    
    def forward(self, img_1, img_2, return_feat=False):
        # import pdb; pdb.set_trace()
        ''' 
            img_1: (N, C, H, W)
            img_2: (N, C, H, W)

        '''
        img = torch.cat([img_1, img_2], dim=0)
        output = self.net(img).squeeze()
        output = output.contiguous().view(2, -1, output.size(-1))
        if return_feat:
            return output[0], output[1]
        output = torch.cat([output[0], output[1]], dim=-1)
        output = self.fusion(output).squeeze()
        return output
