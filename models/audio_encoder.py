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
from config import params
import models



def conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=None):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01),
        nn.LeakyReLU(inplace=True)
    )
# -------------------------------------------------------------------------------------- # 


class AudioEncoder(nn.Module):
    # base encoder
    def __init__(self, args, pr, device=None):
        super(AudioEncoder, self).__init__()
        self.pr = pr
        self.num_classes = pr.num_classes

        # audio transformation:
        self.spect = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=pr.samp_sr, 
                win_length=pr.win_length,
                hop_length=pr.hop_length,
                n_fft=pr.n_fft, 
                f_min=pr.f_min,
                f_max=pr.f_max,
                n_mels=pr.n_mel
            )
        )

    def audio_transform(self, audio, augment=False):
        N, C, A = audio.size()
        audio = audio.view(N * C, A)
        audio = self.spect(audio)
        audio = audio.transpose(-1, -2)
        audio = audio[:, :-1, :]
        
        audio = torch.log(audio + self.pr.log_offset)
        _, T, F = audio.size()
        audio = audio.view(N, C, T, F)
        if augment:
            mask = torch.rand(N, 1, 1, 1) + 0.5
            audio = audio * mask.to(audio.device)
        return audio
    
        
    def get_trained_feature(self, args, pr, model, input_type=None):
        input_type = args.input if input_type == None else input_type
        
        if pr.load_model in ['AVEncoderBinary']:
            if input_type == 'image':
                feature = model.visionnet
                if pr.visionnet in ['VisionRDNet', 'VisionADNet']:
                    if pr.vision_backbone == 'resnet18':
                        features, out_dim = self.get_truncated_resnet(args, feature.net)
            elif input_type == 'audio':
                feature = model.audionet
                if pr.audionet in ['AudioRDNet', 'AudioADNet']: 
                    if pr.audio_backbone == 'vggish':
                        features, out_dim  = self.get_truncated_vggish(args, feature.net, num_layers=15 if args.no_bn else 21)
            elif input_type == 'both':
                featrues = None
                out_dim = 1
        elif pr.load_model in ['AudioRDNet', 'AudioADNet']:
            if pr.audio_backbone == 'vggish':
                features, out_dim  = self.get_truncated_vggish(args, model.net, num_layers=15 if args.no_bn else 21)
        elif pr.load_model in ['VisionRDNet', 'VisionADNet']:
            if pr.vision_backbone == 'resnet18':
                features, out_dim = self.get_truncated_resnet(args, model.net)
        return features, out_dim


    def get_truncated_resnet(self, args, resnet):
        if args.freeze:
            feature = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
                resnet.layer4,
                # resnet.avgpool,
            )
            dim = 512
        else:
            feature = resnet
            dim = 128
        return feature, dim

    def get_truncated_vggish(self, args, vggish_full, num_layers=-1):
        if args.freeze:
            if num_layers == -1:
                feature = nn.Sequential(
                    *list(vggish_full.features.children())
                )
            else:
                feature = nn.Sequential(
                    *list(vggish_full.features.children())[:num_layers]
                )
            dim = 512
        else:
            feature = vggish_full
            dim = 128
        return feature, dim



# ---------------------- Paired Audio  Network --------------------------------------------- # 

class AudioRDNet(AudioEncoder):
    # Audio Relative Depth Net
    def __init__(self, args, pr, output_dim=None, device=None, backbone=None):
        super(AudioRDNet, self).__init__(args, pr, device)
        if not backbone:
            backbone = args.backbone
        network = getattr(models, backbone)
        if backbone == 'vggish':
            pretrained = False
            self.net = network(mono=pr.mono, bn=not args.no_bn, pretrain=pretrained, num_classes=pr.feat_dim)
        
        dim_out = output_dim if output_dim else pr.num_classes
        self.fusion = nn.Sequential(
            nn.Linear(2 * pr.feat_dim, pr.feat_dim),
            nn.ReLU(True),
            nn.Linear(pr.feat_dim, dim_out)
        )
    
    def forward(self, audio_1, audio_2, augment=False, return_feat=False):
        # import pdb; pdb.set_trace()
        ''' 
            audio_1: (N, K)
            audio_2: (N, K)
        '''
    
        audio = torch.cat([audio_1, audio_2], dim=0)
        audio = self.audio_transform(audio, augment)
        output = self.net(audio).squeeze()
        output = output.contiguous().view(2, -1, output.size(-1))
        if return_feat:
            return output[0], output[1]
        
        output = torch.cat([output[0], output[1]], dim=-1)
        output = self.fusion(output).squeeze(1)
        return output


# ---------------------- Single Audio  Network --------------------------------------------- # 

class AudioADNet(AudioEncoder):
    # Audio Absolute Depth Net
    def __init__(self, args, pr, output_dim=None, device=None, backbone=None):
        super(AudioADNet, self).__init__(args, pr, device)
        if not backbone:
            backbone = args.backbone
        network = getattr(models, backbone)
        self.fc = nn.Identity(1)
        dim_out = output_dim if output_dim else pr.num_classes
        if backbone == 'vggish':
            self.net = network(mono=pr.mono, bn=not args.no_bn, pretrain=args.pretrained)
            self.fc = nn.Sequential(
                nn.ReLU(True),
                nn.Linear(128, dim_out)
            )
    
    def forward(self, audio, augment=False, return_feat=False):
        # import pdb; pdb.set_trace()
        ''' 
            audio: (N, K)
        '''
        audio = self.audio_transform(audio, augment)
        output = self.net(audio)
        if return_feat:
            return output
        output = self.fc(output).squeeze(1)
        return output