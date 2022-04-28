import csv
import glob
import h5py
import io
import json
import librosa
import numpy as np
import os
import pickle
from PIL import Image
import random
import scipy
import soundfile as sf
import time
from tqdm import tqdm
import glob
import cv2

import torch
import torch.nn as nn
import torchaudio
import torchvision.transforms as transforms

import sys
sys.path.append('..')
from data import * 


import pdb


class SFSmotionPairDataset(Audio3DBaseDataset):
    def __init__(self, args, pr, list_sample, split='train'):
        super(SFSmotionPairDataset, self).__init__(args, pr, list_sample, split)

    def __getitem__(self, index):
        # import pdb; pdb.set_trace()
        info = self.list_sample[index]
        video_path = info['path']
        
        if self.split == 'train':
            depth_order = np.random.rand() > 0.5
            depth_frame_list = glob.glob(f'{video_path}/Depth/*')
            depth_frame_list.sort()
            depth_frame_list = depth_frame_list[self.pr.frame_rate : -self.pr.frame_rate]
            while True:
                depth_pair = np.random.choice(depth_frame_list, 2, replace=False)
                dist_1, ind_1 = self.read_depth(video_path, depth_pair[0].split('/')[-1])
                dist_2, ind_2 = self.read_depth(video_path, depth_pair[1].split('/')[-1])
                # ensure distance difference isn't too small 
                cond_1 = np.abs(dist_1 - dist_2) >= self.pr.filter_distance
                # ensure audio tracks has no overlap
                cond_2 = np.abs(ind_1 - ind_2) >= 15
                if cond_1 and cond_2:
                    break
            
            # swap the order to balance the depth order training distribution
            if (not depth_order == (dist_1 < dist_2)) and self.pr.objective in ['depth_order', 'depth_ratio']:
                dist_1, dist_2 = dist_2, dist_1
                ind_1, ind_2 = ind_2, ind_1

        else:
            dist_1, ind_1 = self.read_depth(video_path, info['depth_path_1'])
            dist_2, ind_2 = self.read_depth(video_path, info['depth_path_2'])
            depth_order = (dist_1 < dist_2)
            time_order = (ind_1 > ind_2)

        # ------------ load audio ------------- #
        audio_path = os.path.join(video_path, 'Audio', 'sound.wav')
        # Get audio clips from index
        audio_length = int(self.pr.samp_sr * self.pr.clip_length)
        audio_1_start = int(ind_1 / self.pr.frame_rate * self.pr.samp_sr)
        audio_1, _ = self.read_audio(audio_path, start=audio_1_start, stop=audio_1_start + audio_length)

        audio_2_start = int(ind_2 / self.pr.frame_rate * self.pr.samp_sr)
        audio_2, _ = self.read_audio(audio_path, start=audio_2_start, stop=audio_2_start + audio_length)

        if self.split == 'train' and self.aug_wave:
            random_factor = np.random.random() + 0.5
            audio_1 = self.augment_audio(audio_1, random_factor)
            audio_2 = self.augment_audio(audio_2, random_factor)

        audio_1 = torch.from_numpy(audio_1.copy()).float()
        audio_2 = torch.from_numpy(audio_2.copy()).float()

        # ------------ load image ------------- #
        img_1 = self.read_image(video_path, ind_1)
        img_2 = self.read_image(video_path, ind_2)
        
        depth_order = float(depth_order)
        distance = float(np.abs(dist_1 - dist_2))
        depth_ratio = float(np.log(dist_1 / dist_2))

        # normalization depth to [-0.5, -0.5]
        depth_scale = 1.0 / self.pr.depth_max 
        dist_1 = float(dist_1 * depth_scale) - 0.5
        dist_2 = float(dist_2 * depth_scale) - 0.5

        batch = {
            'audio_info': audio_path,
            'audio_1': audio_1,
            'audio_2': audio_2,
            'img_1': img_1,
            'img_2': img_2,
            'depth_1': torch.tensor(dist_1),
            'depth_2': torch.tensor(dist_2),
            'ind_1': torch.tensor(ind_1),
            'ind_2': torch.tensor(ind_2),
            'depth_order': torch.tensor(depth_order),
            'depth_ratio': torch.tensor(depth_ratio),
            'distance': torch.tensor(distance)
        }
        return batch