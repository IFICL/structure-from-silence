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
from utils import sound, sourcesep
from data import * 

import pdb


class SFSmotionSingleDataset(Audio3DBaseDataset):
    def __init__(self, args, pr, list_sample, split='train'):
        super(SFSmotionSingleDataset, self).__init__(args, pr, list_sample, split)

    def __getitem__(self, index):
        # import pdb; pdb.set_trace()
        info = self.list_sample[index]
        video_path = info['path']
        obstacle = 0

        if self.split == 'train':
            depth_frame_list = glob.glob(f'{video_path}/Depth/*')
            depth_frame_list.sort()
            depth_frame_list = depth_frame_list[self.pr.frame_rate : -self.pr.frame_rate]
            
            if self.pr.objective == 'obstacle':
                dist, ind = self.read_depth(video_path, info['depth_path_1'])
                obstacle = float(info['IfObstacle'])
            else:
                depth_frame = np.random.choice(depth_frame_list, 1, replace=False)[0]
                dist, ind = self.read_depth(video_path, depth_frame.split('/')[-1])
        
        else:
            dist, ind = self.read_depth(video_path, info['depth_path_1'])
            if self.pr.objective == 'obstacle':
                obstacle = float(info['IfObstacle'])

        # ------------ load audio ------------- #
        audio_path = os.path.join(video_path, 'Audio', 'sound.wav')
        # Get audio clips from index
        audio_length = int(self.pr.samp_sr * self.pr.clip_length)
        audio_start = int(ind / self.pr.frame_rate * self.pr.samp_sr)
        audio, _ = self.read_audio(audio_path, start=audio_start, stop=audio_start + audio_length)

        if self.split == 'train' and self.aug_wave:
            audio = self.augment_audio(audio)

        audio = torch.from_numpy(audio.copy()).float()
        img = self.read_image(video_path, ind)

        # normalize depth to [0, 1]
        depth_scale = 1 / self.pr.depth_max
        dist = float(dist * depth_scale)


        batch = {
            'audio_info': audio_path,
            'audio': audio,
            'img': img,
            'depth': torch.tensor(dist),
            'obstacle': torch.tensor(float(obstacle))
        }
        return batch


