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
from PIL import ImageFilter
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


class Audio3DBaseDataset(object):
    def __init__(self, args, pr, list_sample, split='train'):
        self.pr = pr
        self.split = split
        self.seed = pr.seed
        self.aug_wave = args.aug_wave
        # save args parameter
        self.repeat = args.repeat if split == 'train' else 1
        self.max_sample = args.max_sample
        self.class_info = pr.class_info

        self.vision_transform = transforms.Compose(self.generate_vision_transform(args, pr))
        if isinstance(list_sample, str):
            self.list_sample = []
            csv_file = csv.DictReader(open(list_sample, 'r'), delimiter=',')
            for row in csv_file:
                self.list_sample.append(row)
        
        if self.max_sample > 0: 
            self.list_sample = self.list_sample[0:self.max_sample]
        
        self.data_weight = self.get_data_weight()
        if split in ['train', 'test']:
            pr.data_weight = self.data_weight

        self.list_sample = self.list_sample * self.repeat

        random.seed(self.seed)
        np.random.seed(1234)
        num_sample = len(self.list_sample)
        if self.split == 'train':
            random.shuffle(self.list_sample)
        
        # self.class_dist = self.unbalanced_dist()
        print('Audio Dataloader: # sample of {}: {}'.format(self.split, num_sample))


    def __getitem__(self, index):
        # import pdb; pdb.set_trace()
        info = self.list_sample[index]
        batch = info
        return batch

    def getitem_test(self, index):
        self.__getitem__(index)

    def __len__(self): 
        return len(self.list_sample)

    def read_audio(self, audio_path, start=0, stop=None):
        # using soundfile
        audio, audio_rate = sf.read(audio_path, start=start, stop=stop, dtype='float64', always_2d=True)
        # using mono or binatural 
        if self.pr.mono and len(audio.shape) == 2: 
            audio = np.mean(audio, axis=-1, keepdims=True)
        audio = np.transpose(audio, (1, 0))
        return audio, audio_rate
    
    def read_image(self, video_path, ind):
        # import pdb; pdb.set_trace()
        frame_ind = int(ind + (self.pr.clip_length * self.pr.frame_rate) // 2) 
        img_path = f'{video_path}/RGB/c{str(frame_ind).zfill(3)}.png'
        image = Image.open(img_path).convert('RGB')
        image = self.vision_transform(image)
        return image

    def read_depth(self, video_path, depth_frame):
        # import pdb; pdb.set_trace()
        frame_ind = int(depth_frame[1: 4])
        frame_len = int(self.pr.frame_rate * self.pr.clip_length)
        # read depth frame
        depth_frame_list = [f'{video_path}/Depth/d{str(frame_ind + frame_len // 2).zfill(3)}.png']

        depth_images = [cv2.imread(frame, cv2.IMREAD_UNCHANGED) for frame in depth_frame_list]
        depth_images = np.stack(depth_images)
        frame_h = depth_images.shape[1]
        frame_w = depth_images.shape[2]
        # average along time
        dist = np.mean(depth_images[:, frame_h // 4: frame_h // 4 * 3, frame_w // 4: frame_w // 4 * 3])
        # convert back to meters
        dist = dist * 10 / (2**16)
        if dist >= self.pr.depth_max:
            dist = self.pr.depth_max

        return dist, frame_ind

    def normalize_audio(self, samples, desired_rms=0.1, eps=1e-4):
        rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
        samples = samples * (desired_rms / rms)
        return samples 

    def augment_audio(self, audio, random_factor=None):
        if random_factor == None:
            random_factor = np.random.random() + 0.5 # 0.5 - 1.5
        
        audio = audio * random_factor 
        audio[audio > 1.] = 1.
        audio[audio < -1.] = -1.
        return audio
    
    def generate_vision_transform(self, args, pr):
        resize_funct = transforms.Resize((pr.img_size, pr.img_size))

        if self.split == 'train' and args.aug_img:
            flip_funct = transforms.RandomHorizontalFlip(pr.flip_prob)
        else:
            flip_funct = transforms.Lambda(lambda img: img)

        if self.split == 'train' and args.aug_img:
            guass_funct = transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5)
        else:
            guass_funct = transforms.Lambda(lambda img: img)


        vision_transform_list = [
            resize_funct,
            flip_funct,
            guass_funct, 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        return vision_transform_list
    
    def get_data_weight(self):
        # import pdb; pdb.set_trace()
        # default weights for binary and multi-classification
        if self.pr.num_classes == 1:
            weight = 1.0
        else:
            weight = np.array([1.0] * self.pr.num_classes) / self.pr.num_classes
        
        if self.split == 'train':
            if self.pr.objective == 'obstacle':
                dist = np.zeros(2)
                for item in self.list_sample:
                    index = int(item['IfObstacle'])
                    dist[index] += 1
                weight = dist[0] / dist[1]
        return weight


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x