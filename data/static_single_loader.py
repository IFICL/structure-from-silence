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


class SFSstaticSingleDataset(Audio3DBaseDataset):
    def __init__(self, args, pr, list_sample, split='train'):
        super(SFSstaticSingleDataset, self).__init__(args, pr, list_sample, split)

    def __getitem__(self, index):
        # import pdb; pdb.set_trace()
        info = self.list_sample[index]
        sample_path = info['path']
        audio_path = os.path.join(sample_path, 'sound.wav')
        image_path = os.path.join(sample_path, 'RGB.png')
        depth_path = os.path.join(sample_path, 'Depth.png')
        annotation_path = os.path.join(sample_path, 'annotation.json')
        with open(annotation_path, 'r') as fp:
            annotation = json.load(fp)
        
        depth = float(annotation['depth'])
        depth_max = float(annotation['Max depth in map'])
        obstacle = float(info['IfObstacle'])

        total_length = int(self.pr.samp_sr * float(annotation['Audio Length']))
        audio_length = int(self.pr.samp_sr * self.pr.clip_length)

        if self.split == 'train':
            audio_start = np.random.choice(total_length - audio_length)
        else:
            audio_start = int(float(info['start second']) * self.pr.samp_sr)
        # ------------ load audio ------------- #
        # Get audio clips from index
        audio, _ = self.read_audio(audio_path, start=audio_start, stop=audio_start + audio_length)
        if self.split == 'train' and self.aug_wave:
            audio = self.augment_audio(audio)
        audio = torch.from_numpy(audio.copy()).float()
        
        img = self.read_image(image_path)

        batch = {
            'audio_path': audio_path,
            'image_path': image_path,
            'audio': audio,
            'img': img,
            'depth': torch.tensor(depth),
            'obstacle': torch.tensor(obstacle)
        }
        return batch

    def read_image(self, img_path):
        image = Image.open(img_path).convert('RGB')
        image = self.vision_transform(image)
        return image

