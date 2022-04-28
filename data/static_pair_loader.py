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


class SFSstaticPairDataset(Audio3DBaseDataset):
    def __init__(self, args, pr, list_sample, split='train'):
        super(SFSstaticPairDataset, self).__init__(args, pr, list_sample, split)

    def __getitem__(self, index):
        info = self.list_sample[index]
        scene_path = info['path']

        audio_list = []
        image_list = []
        depth_list = []
        for clip in [info['clip_1'], info['clip_2']]:
            audio_path = os.path.join(scene_path, clip, 'sound.wav')
            image_path = os.path.join(scene_path, clip, 'RGB.png')
            depth_path = os.path.join(scene_path, clip, 'Depth.png')
            annotation_path = os.path.join(scene_path, clip, 'annotation.json')
            with open(annotation_path, 'r') as fp:
                annotation = json.load(fp)
            
            depth = float(annotation['depth'])
            total_length = int(self.pr.samp_sr * float(annotation['Audio Length']))
            audio_length = int(self.pr.samp_sr * self.pr.clip_length)
            if self.split == 'train':
                audio_start = np.random.choice(total_length - audio_length)
            else:
                audio_start = 0
            # ------------ load audio ------------- #
            # Get audio clips from index
            audio, _ = self.read_audio(audio_path, start=audio_start, stop=audio_start + audio_length)

            audio = torch.from_numpy(audio.copy()).float()
            img = self.read_image(image_path)

            audio_list.append(audio)
            image_list.append(img)
            depth_list.append(depth)
        
        if self.split == 'train' and self.aug_wave:
            random_factor = np.random.random() + 0.5
            for i in range(2):
                audio_list[0] = self.augment_audio(audio_list[0], random_factor)
                audio_list[1] = self.augment_audio(audio_list[1], random_factor)


        depth_order = float(depth_list[0] < depth_list[1])
        depth_ratio = float(np.log(depth_list[0] / depth_list[1]))

        batch = {
            'audio_1': audio_list[0],
            'audio_2': audio_list[1],
            'img_1': image_list[0],
            'img_2': image_list[1],
            'depth_1': torch.tensor(depth_list[0]),
            'depth_2': torch.tensor(depth_list[1]),
            'depth_order': torch.tensor(depth_order),
            'depth_ratio': torch.tensor(depth_ratio)
        }
        return batch
    
    def read_image(self, img_path):
        image = Image.open(img_path).convert('RGB')
        image = self.vision_transform(image)
        return image

