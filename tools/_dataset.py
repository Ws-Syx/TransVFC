# import some common libraries
import numpy as np
import os
import json
import cv2
import random
import tqdm
import math
import matplotlib.pyplot as plt

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.events import EventStorage
from detectron2.modeling import build_model
from detectron2.structures import BoxMode, Boxes, Instances

import detectron2.data.transforms as T
import detectron2.utils.comm as comm

import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel

import torchvision.transforms as transforms  


class TrainSingleDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, json_path, video_path, randomly_flip=False):
        # data
        self.video_path = video_path
        self.json_file = json.load(open(json_path))
    
        # data pre-process
        self.randomly_flip = randomly_flip
        self.cfg = cfg.clone()
        self.aug = T.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST], self.cfg.INPUT.MAX_SIZE_TEST
        )
        
    def __len__(self):
        return len(self.json_file['videos'])

    def __getitem__(self, index, t=None):
        # pick the specific video
        video_dict = self.json_file['videos'][index]
        
        # read the frame randomly
        input_index = random.choice(list(range(video_dict['length']))) if t is None else t
        input_frame_name = video_dict['file_names'][input_index]
        input_frame_path = os.path.join(self.video_path, input_frame_name)
        input_frame = cv2.imread(input_frame_path)
        # data augumentation
        input_frame = self.aug.get_transform(input_frame).apply_image(input_frame)
        input_frame = torch.as_tensor(input_frame.astype("float32").transpose(2, 0, 1))
        _, h, w = input_frame.shape

        # read the mask
        input_mask_name = video_dict['mask_names'][input_index]
        input_mask_path = os.path.join(self.video_path, input_mask_name)
        input_mask = cv2.imread(input_mask_path, cv2.IMREAD_GRAYSCALE)
        # data augumentation
        input_mask = cv2.resize(input_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        input_mask = torch.as_tensor(input_mask)
        input_mask[input_mask < 1] = 0
        input_mask[input_mask > 124] = 0

        if self.randomly_flip and random.random() < 0.5:
            input_frame = torch.flip(input_frame, dims=[2])
            input_mask = torch.flip(input_mask, dims=[1])

        input_dict = {'height': video_dict['height'],
                      'width': video_dict['width'],
                      'file_name': input_frame_name,
                      'id': input_frame_name,
                      'image': input_frame,
                      'mask': input_mask
                      }
        
        return input_dict


class TestSingleDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, json_path, video_path, mask_path, randomly_flip=False):
        # data
        self.video_path = video_path
        self.mask_path = mask_path
        self.json_file = json.load(open(json_path))
    
        # data pre-process
        self.randomly_flip = randomly_flip
        self.cfg = cfg.clone()
        self.aug = T.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST], self.cfg.INPUT.MAX_SIZE_TEST
        )
        
    def __len__(self):
        return len(self.json_file['videos'])

    def __getitem__(self, index, t=None):
        # pick the specific video
        video_dict = self.json_file['videos'][index]
        
        # read the frame randomly
        input_index = random.choice(list(range(video_dict['length']))) if t is None else t
        input_frame_name = video_dict['file_names'][input_index]
        input_frame_path = os.path.join(self.video_path, input_frame_name)
        input_frame = cv2.imread(input_frame_path)
        # data augumentation
        input_frame = self.aug.get_transform(input_frame).apply_image(input_frame)
        input_frame = torch.as_tensor(input_frame.astype("float32").transpose(2, 0, 1))
        _, h, w = input_frame.shape

        # read the mask
        input_mask_name = video_dict['mask_names'][input_index]
        input_mask_path = os.path.join(self.mask_path, input_mask_name)
        input_mask = cv2.imread(input_mask_path, cv2.IMREAD_GRAYSCALE)
        # data augumentation
        input_mask = cv2.resize(input_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        input_mask = torch.as_tensor(input_mask)
        input_mask[input_mask < 1] = 0
        input_mask[input_mask > 124] = 0

        if self.randomly_flip and random.random() < 0.5:
            input_frame = torch.flip(input_frame, dims=[2])
            input_mask = torch.flip(input_mask, dims=[1])

        input_dict = {'height': video_dict['height'],
                      'width': video_dict['width'],
                      'file_name': input_frame_name,
                      'id': input_frame_name,
                      'image': input_frame,
                      'mask': input_mask
                      }
        
        return input_dict


class TrainDoubleDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, json_path, video_path, mask_path, augmentation=False):
        self.video_path = video_path
        self.mask_path = mask_path
        self.json_file = json.load(open(json_path))
        self.delta = [-2, -1, 1, 2]
    
        # data pre-process
        self.cfg = cfg.clone()
        self.aug = T.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST], self.cfg.INPUT.MAX_SIZE_TEST
        )
        self.augmentation = augmentation

    def __len__(self):
        return len(self.json_file['videos'])

    def read_image(self, file_path):
        input_frame = cv2.imread(file_path)
        input_frame = self.aug.get_transform(input_frame).apply_image(input_frame)
        input_frame = torch.as_tensor(input_frame.astype("float32").transpose(2, 0, 1))
        new_height, new_width = input_frame.shape[1:]
        return input_frame, new_height, new_width
    
    def read_mask(self, file_path, h, w):
        # read the mask
        input_mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        # data augumentation
        input_mask = cv2.resize(input_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        input_mask = torch.as_tensor(input_mask)
        input_mask[input_mask < 1] = 0
        input_mask[input_mask > 124] = 0
        return input_mask

    def __getitem__(self, index):
        # pick the specific video
        video_dict = self.json_file['videos'][index]

        # ---------------------------------- (1) input frame ----------------------------
        # randomly pick input frame
        input_index = random.choice(list(range(video_dict['length'])))
        # read frame
        input_frame_name = video_dict['file_names'][input_index]
        input_frame_path = os.path.join(self.video_path, input_frame_name)
        input_frame, new_height, new_width = self.read_image(input_frame_path)      
        # read mask
        input_mask_name = video_dict['mask_names'][input_index]
        input_mask_path = os.path.join(self.mask_path, input_mask_name)
        input_mask = self.read_mask(input_mask_path, new_height, new_width)
        
        # ---------------------------------- (2) refer frame ----------------------------
        # randomly pick refer frame
        while True:
            refer_index = input_index + random.choice(self.delta)
            if 0 <= refer_index < video_dict['length']:
                break
        # read frame
        refer_frame_name = video_dict['file_names'][refer_index]
        refer_frame_path = os.path.join(self.video_path, refer_frame_name)
        refer_frame, new_height, new_width = self.read_image(refer_frame_path)
        # read mask
        refer_mask_name = video_dict['mask_names'][refer_index]
        refer_mask_path = os.path.join(self.mask_path, refer_mask_name)
        refer_mask = self.read_mask(refer_mask_path, new_height, new_width)
        
        # -------------------------- augmentation ---------------------------------
        if self.augmentation:
            # randomly flip
            if random.random() < 0.5:
                input_frame = torch.flip(input_frame, dims=[2])
                input_mask = torch.flip(input_mask, dims=[1])
                refer_frame = torch.flip(refer_frame, dims=[2])
                refer_mask = torch.flip(refer_mask, dims=[1])
            # randomly crop
            if random.random() < 0.5:
                # randomly choose a new region
                h, w = input_mask.shape
                random_factor = 0.7 + random.random() * 0.3
                crop_h, crop_w = int(random_factor * h), int(random_factor * w)
                crop_x = random.randint(0, max(0, w - crop_w))  
                crop_y = random.randint(0, max(0, h - crop_h))  
                # crop the image and mask
                input_frame = input_frame[:, crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]  
                input_mask = input_mask[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]  
                refer_frame = refer_frame[:, crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]  
                refer_mask = refer_mask[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]  
                # reshape to original size
                input_frame = torch.nn.functional.interpolate(input_frame.unsqueeze(0), size=(h, w), mode='bilinear')[0]
                input_mask = torch.nn.functional.interpolate(input_mask.unsqueeze(0).unsqueeze(0), size=(h, w), mode='nearest')[0][0]
                refer_frame = torch.nn.functional.interpolate(refer_frame.unsqueeze(0), size=(h, w), mode='bilinear')[0]
                refer_mask = torch.nn.functional.interpolate(refer_mask.unsqueeze(0).unsqueeze(0), size=(h, w), mode='nearest')[0][0]

        # -------------------------- return ---------------------------------

        # create the dict
        input_dict = {'height': video_dict['height'], 'width': video_dict['width'],
                      'file_name': video_dict['file_names'][input_index],
                      'id': video_dict['file_names'][input_index],
                      'image': input_frame, 'mask': input_mask}
        # create the dict
        refer_dict = {'height': video_dict['height'], 'width': video_dict['width'],
                      'file_name': video_dict['file_names'][refer_index],
                      'id': video_dict['file_names'][refer_index],
                      'image': refer_frame, 'mask': refer_mask}

        return {'input': input_dict, 'refer': refer_dict}



class TestSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, json_path, video_path, mask_path, randomly_flip=False):
        # data
        self.video_path = video_path
        self.mask_path = mask_path
        self.json_file = json.load(open(json_path))
    
        # data pre-process
        self.randomly_flip = randomly_flip
        self.cfg = cfg.clone()
        self.aug = T.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST], self.cfg.INPUT.MAX_SIZE_TEST
        )
        
    def __len__(self):
        return len(self.json_file['videos'])

    def __getitem__(self, index, t=None):
        # pick the specific video
        video_dict = self.json_file['videos'][index]
        result = []
        for input_index in range(min(video_dict['length'], 24)):
            # read the frame
            input_frame_name = video_dict['file_names'][input_index]
            input_frame_path = os.path.join(self.video_path, input_frame_name)
            input_frame = cv2.imread(input_frame_path)
            # data augumentation
            input_frame = self.aug.get_transform(input_frame).apply_image(input_frame)
            input_frame = torch.as_tensor(input_frame.astype("float32").transpose(2, 0, 1))
            _, h, w = input_frame.shape

            # read the mask
            input_mask_name = video_dict['mask_names'][input_index]
            input_mask_path = os.path.join(self.mask_path, input_mask_name)
            input_mask = cv2.imread(input_mask_path, cv2.IMREAD_GRAYSCALE)
            # data augumentation
            input_mask = cv2.resize(input_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            input_mask = torch.as_tensor(input_mask)
            input_mask[input_mask < 1] = 0
            input_mask[input_mask > 124] = 0

            if self.randomly_flip and random.random() < 0.5:
                input_frame = torch.flip(input_frame, dims=[2])
                input_mask = torch.flip(input_mask, dims=[1])

            input_dict = {'height': video_dict['height'],
                        'width': video_dict['width'],
                        'file_name': input_frame_name,
                        'id': input_frame_name,
                        'image': input_frame,
                        'mask': input_mask
                        }
            result.append(input_dict)
            
        return result