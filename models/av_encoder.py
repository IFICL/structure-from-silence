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
from models import *


def conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=None):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01),
        nn.LeakyReLU(inplace=True)
    )


# -------------------------------------------------------------------------------------- # 

class AVEncoder(nn.Module):
    # base encoder
    def __init__(self, args, pr, device=None):
        super(AVEncoder, self).__init__()
        self.pr = pr
        self.num_classes = pr.num_classes
        
        network = getattr(models, args.backbone)

        self.visionnet = models.__dict__[pr.visionnet](args, pr, output_dim=pr.feat_dim, device=device, backbone=pr.vision_backbone)
        
        self.audionet = models.__dict__[pr.audionet](args, pr, output_dim=pr.feat_dim, device=device, backbone=pr.audio_backbone)

    def forward(self, audio_1, audio_2, img_1, img_2, augment=False):
        # import pdb; pdb.set_trace()
        ''' 
            audio_1: (N, K)
            audio_2: (N, K)
            img_1: (N, C, H, W)
            img_2: (N, C, H, W)
        '''
        feat_aud_1 = self.audionet(audio_1, audio_2, augment=augment, return_feat=True)
        feat_aud_2 = self.audionet(audio_2, audio_1, augment=augment, return_feat=True)

        feat_img_1 = self.visionnet(img_1, img_2, return_feat=True)
        feat_img_2 = self.visionnet(img_2, img_1, return_feat=True)

        feat_aud = torch.cat([feat_aud_1.unsqueeze(1), feat_aud_2.unsqueeze(1)], dim=1)
        feat_img = torch.cat([feat_img_1.unsqueeze(1), feat_img_2.unsqueeze(1)], dim=1)

        return {
            'aud': feat_aud,
            'img': feat_img
        }


class AVEncoderBinary(AVEncoder):
    # Audio Relative Depth Net
    def __init__(self, args, pr, device=None):
        super(AVEncoderBinary, self).__init__(args, pr, device)

        self.fusion = nn.Sequential(
            nn.Linear(2 * pr.feat_dim, pr.feat_dim),
            nn.ReLU(True),
            nn.Linear(pr.feat_dim, 1)
        )
    
    def forward(self, audio_1, audio_2, img_1, img_2, augment=False):
        # import pdb; pdb.set_trace()
        ''' 
            audio_1: (N, K)
            audio_2: (N, K)
            img_1: (N, C, H, W)
            img_2: (N, C, H, W)
        '''
        feat_aud_1 = self.audionet(audio_1, audio_2, augment=augment, return_feat=False)
        feat_aud_2 = self.audionet(audio_2, audio_1, augment=augment, return_feat=False)

        feat_img_1 = self.visionnet(img_1, img_2, return_feat=False)
        feat_img_2 = self.visionnet(img_2, img_1, return_feat=False)


        out_1 = self.fusion(torch.cat([feat_img_1, feat_aud_1], dim=-1))
        out_2 = self.fusion(torch.cat([feat_img_1, feat_aud_2], dim=-1))
        out_3 = self.fusion(torch.cat([feat_img_2, feat_aud_1], dim=-1))
        out_4 = self.fusion(torch.cat([feat_img_2, feat_aud_2], dim=-1))
        out = torch.cat([out_1, out_2, out_3, out_4], dim=-1)
        target = torch.tensor([1, 0, 0, 1]).repeat(out.size(0), 1).to(out.device)
        out = out.contiguous().view(-1)
        target = target.contiguous().view(-1)

        return {
            'pred': out,
            'target': target
        }

class AVEncoderBinaryV2(AVEncoder):
    # Audio Relative Depth Net
    def __init__(self, args, pr, device=None):
        super(AVEncoderBinaryV2, self).__init__(args, pr, device)

        self.fusion = nn.Sequential(
            nn.Linear(pr.feat_dim, pr.feat_dim),
            nn.ReLU(True),
            nn.Linear(pr.feat_dim, 1)
        )
    
    def forward(self, audio_1, audio_2, img_1, img_2, augment=False):
        # import pdb; pdb.set_trace()
        ''' 
            audio_1: (N, K)
            audio_2: (N, K)
            img_1: (N, C, H, W)
            img_2: (N, C, H, W)
        '''
        feat_aud_1 = self.audionet(audio_1, audio_2, augment=augment, return_feat=False)
        feat_aud_2 = self.audionet(audio_2, audio_1, augment=augment, return_feat=False)

        feat_img_1 = self.visionnet(img_1, img_2, return_feat=False)
        feat_img_2 = self.visionnet(img_2, img_1, return_feat=False)


        out_1 = self.fusion(feat_img_1 + feat_aud_1)
        out_2 = self.fusion(feat_img_1 + feat_aud_2)
        out_3 = self.fusion(feat_img_2 + feat_aud_1)
        out_4 = self.fusion(feat_img_2 + feat_aud_2)
        out = torch.cat([out_1, out_2, out_3, out_4], dim=-1)
        target = torch.tensor([1, 0, 0, 1]).repeat(out.size(0), 1).to(out.device)
        out = out.contiguous().view(-1)
        target = target.contiguous().view(-1)

        return {
            'pred': out,
            'target': target
        }


# --------------- Linear Classification model -------------------------- #

class AVPairLinearProbe(AudioEncoder):
    # Audio Absolute Depth Net
    def __init__(self, args, pr, model=None, device=None):
        super(AVPairLinearProbe, self).__init__(args, pr, device)
        self.input_type = args.input
        self.freeze = args.freeze
        self.features, feat_dim = self.get_trained_feature(args, pr, model)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2 * feat_dim, pr.num_classes)
        # init the fc layer
        self.fc.weight.data.normal_(mean=0.0, std=0.01)
        self.fc.bias.data.zero_()
        # import pdb; pdb.set_trace()

        if args.freeze:
            for param in self.features.parameters():
                param.requires_grad = False
        else:
            self.fc = nn.Sequential(
                nn.Linear(2 * pr.feat_dim, pr.feat_dim),
                nn.ReLU(True),
                nn.Linear(pr.feat_dim, pr.num_classes)
            )

    def forward(self, input_1, input_2):
        if self.input_type == 'audio' and self.pr.format in ['mel']:
            input_1 = self.audio_transform(input_1)
            input_2 = self.audio_transform(input_2)
        
        feat_1 = self.features(input_1)
        feat_2 = self.features(input_2)
        if self.freeze:
            feat_1 = self.avgpool(feat_1).view(feat_1.size(0), -1)
            feat_2 = self.avgpool(feat_2).view(feat_2.size(0), -1)
        feat = torch.cat([feat_1, feat_2], dim=-1)
        output = self.fc(feat).squeeze()
        return output

    
class AVSingleLinearProbe(AudioEncoder):
    # Audio Absolute Depth Net
    def __init__(self, args, pr, model=None, device=None):
        super(AVSingleLinearProbe, self).__init__(args, pr, device=device)
        self.input_type = args.input
        self.freeze = args.freeze
        self.features, feat_dim = self.get_trained_feature(args, pr, model)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(feat_dim, pr.num_classes)
        # init the fc layer
        self.fc.weight.data.normal_(mean=0.0, std=0.01)
        self.fc.bias.data.zero_()
        # import pdb; pdb.set_trace()

        if args.freeze:
            for param in self.features.parameters():
                param.requires_grad = False
        else:
            self.fc = nn.Sequential(
                nn.ReLU(True),
                nn.Linear(pr.feat_dim, pr.num_classes)
            )

    def forward(self, input_1):
        # import pdb; pdb.set_trace()
        if self.input_type == 'audio' and self.pr.format in ['mel']:
            input_1 = self.audio_transform(input_1)
        
        feat_1 = self.features(input_1)
        if self.freeze:
            feat_1 = self.avgpool(feat_1).view(feat_1.size(0), -1)
        output = self.fc(feat_1).squeeze()
        return output



class AVPairLinearProbewithBoth(AudioEncoder):
    # Audio Absolute Depth Net
    def __init__(self, args, pr, model=None, device=None):
        super(AVPairLinearProbewithBoth, self).__init__(args, pr, device=device)
        # AudioEncoder.__init__(args, pr, device)

        # self.input_type = args.input
        self.audio_features, audio_feat_dim = self.get_trained_feature(args, pr, model, 'audio')
        self.vision_features, vision_feat_dim = self.get_trained_feature(args, pr, model, 'image')
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        feat_dim = 2 * audio_feat_dim + 2 * vision_feat_dim
        self.fc = nn.Linear(feat_dim, pr.num_classes)
        # init the fc layer
        self.fc.weight.data.normal_(mean=0.0, std=0.01)
        self.fc.bias.data.zero_()
        # import pdb; pdb.set_trace()

        if args.freeze:
            for param in self.audio_features.parameters():
                param.requires_grad = False
            for param in self.vision_features.parameters():
                param.requires_grad = False

    def forward(self, audio_1, audio_2, image_1, image_2):
        if self.pr.format in ['mel', 'spec']:
            audio_1 = self.audio_transform(audio_1)
            audio_2 = self.audio_transform(audio_2)


        image_1 = self.vision_features(image_1)
        image_2 = self.vision_features(image_2)
        audio_1 = self.audio_features(audio_1)
        audio_2 = self.audio_features(audio_2)

        image = torch.cat([image_1, image_2], dim=1)
        audio = torch.cat([audio_1, audio_2], dim=1)
        
        image = self.avgpool(image).view(image.size(0), -1)
        audio = self.avgpool(audio).view(audio.size(0), -1)
        feat = torch.cat([image, audio], dim=-1)
        output = self.fc(feat).squeeze()
        return output


class AVSingleLinearProbewithBoth(AudioEncoder):
    # Audio Absolute Depth Net
    def __init__(self, args, pr, model=None, device=None):
        # import pdb; pdb.set_trace()
        super(AVSingleLinearProbewithBoth, self).__init__(args, pr, device=device)
        
        self.pr = pr
        # self.device = device
        self.input_type = args.input
        self.num_classes = pr.num_classes

        self.audio_features, audio_feat_dim = self.get_trained_feature(args, pr, model, 'audio')
        self.vision_features, vision_feat_dim = self.get_trained_feature(args, pr, model, 'image')
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        feat_dim = audio_feat_dim + vision_feat_dim
        self.fc = nn.Linear(feat_dim, pr.num_classes)
        # init the fc layer
        self.fc.weight.data.normal_(mean=0.0, std=0.01)
        self.fc.bias.data.zero_()
        # import pdb; pdb.set_trace()

        if args.freeze:
            for param in self.audio_features.parameters():
                param.requires_grad = False
            for param in self.vision_features.parameters():
                param.requires_grad = False

    def forward(self, audio_1, image_1):
        if self.pr.format in ['mel', 'spec']:
            audio_1 = self.audio_transform(audio_1)

        image_1 = self.vision_features(image_1)
        audio_1 = self.audio_features(audio_1)
        image_1 = self.avgpool(image_1).view(image_1.size(0), -1)
        audio_1 = self.avgpool(audio_1).view(audio_1.size(0), -1)
        
        feat = torch.cat([image_1, audio_1], dim=-1)
        output = self.fc(feat).squeeze()
        return output
