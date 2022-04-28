import os

import numpy as np

import sys
sys.path.append('..')
from utils import utils

Params = utils.Params


def base(name):
    pr = Params(
        frame_rate = 15,
        samp_sr = 16000,
        clip_length = 0.96,
        log_spec = True,
        hop_length = 160,
        f_min=0,
        f_max=None,
        log_offset=1e-10,
        n_mel = 64,
        win_length=400,
        n_fft=512,
        spec_min=-100.,
        spec_max = 100.,
        num_samples = 0,
        mono = True,
        seed=1234,
        img_size=224,
        crop_size=224,
        flip_prob = 0.5,
        gamma = 0.3,
        class_info = {
            '1st >= 2nd': 0, 
            '1st < 2nd': 1,
        },
        num_classes = 1,
        feat_dim = 128,
        format = 'mel',
        filter_distance = 0.1,
        depth_encoding = False,
        depth_max = 3.0766138931115465,
        objective = None,
        net = None,
        dataloader = None,
        loss = None,
        lr_milestones = [20, 40, 60, 80, 100],
        list_train = '',
        list_val = '',
        list_test = '',
        n_group = 8
    )

    return pr

# ---------------------- Static Recordings ------------------------------ # 
def static_obstacle(**kwargs):
    pr = base('depth-mono', **kwargs)
    pr.num_samples = int(round(pr.samp_sr * pr.clip_length))
    pr.dataloader = 'SFSstaticSingleDataset'
    pr.net = 'AudioADNet'
    pr.loss = 'BCLoss'
    pr.objective = 'obstacle'
    pr.list_train = 'data/SFS-Static/data-split/obstacle-detection/train.csv'
    pr.list_val = 'data/SFS-Static/data-split/obstacle-detection/val.csv'
    pr.list_test = 'data/SFS-Static/data-split/obstacle-detection/test.csv'
    return pr


def static_relative_depth(**kwargs):
    pr = base('depth-mono', **kwargs)
    pr.num_samples = int(round(pr.samp_sr * pr.clip_length))
    pr.dataloader = 'SFSstaticPairDataset'
    pr.net = 'AudioRDNet'
    pr.loss = 'BCLoss'
    pr.objective = 'depth_order'
    pr.list_train = 'data/SFS-Static/data-split/relative-depth/train.csv'
    pr.list_val = 'data/SFS-Static/data-split/relative-depth/val.csv'
    pr.list_test = 'data/SFS-Static/data-split/relative-depth/test.csv'
    return pr


# ---------------------- Motion Recordings ------------------------------ # 

def motion_obstacle(**kwargs):
    pr = base('depth-mono', **kwargs)
    pr.num_samples = int(round(pr.samp_sr * pr.clip_length))
    pr.dataloader = 'SFSmotionSingleDataset'
    pr.net = 'AudioADNet'
    pr.loss = 'BCLoss'
    pr.objective = 'obstacle'
    pr.list_train = 'data/SFS-Motion/data-split/obstacle-detection/train.csv'
    pr.list_val = 'data/SFS-Motion/data-split/obstacle-detection/val.csv'
    pr.list_test = 'data/SFS-Motion/data-split/obstacle-detection/test.csv'
    return pr


def motion_relative_depth(**kwargs):
    pr = base('depth-mono', **kwargs)
    pr.num_samples = int(round(pr.samp_sr * pr.clip_length))
    pr.dataloader = 'SFSmotionPairDataset'
    pr.net = 'AudioRDNet'
    pr.loss = 'BCLoss'
    pr.objective = 'depth_order'
    pr.list_train = 'data/SFS-Motion/data-split/relative-depth/train.csv'
    pr.list_val = 'data/SFS-Motion/data-split/relative-depth/val.csv'
    pr.list_test = 'data/SFS-Motion/data-split/relative-depth/test.csv'
    return pr

def motion_avorder(**kwargs):
    pr = motion_relative_depth(**kwargs)
    pr.net = 'AVEncoderBinary'
    pr.visionnet = 'VisionRDNet'
    pr.vision_backbone = 'resnet18'
    pr.audionet = 'AudioRDNet'
    pr.audio_backbone = 'vggish'
    pr.loss = 'BCLoss'
    return pr


# -------------------- Linear Probe ---------------------- #

def av_lincls_RD(**kwargs):
    pr = motion_relative_depth(**kwargs)
    pr.net = 'AVPairLinearProbe'
    pr.load_model = 'AVEncoderBinary'
    pr.visionnet = 'VisionRDNet'
    pr.vision_backbone = 'resnet18'
    pr.audionet = 'AudioRDNet'
    pr.audio_backbone = 'vggish'
    pr.loss = 'BCLoss'
    pr.num_classes = 1
    pr.objective = 'depth_order'
    return pr



