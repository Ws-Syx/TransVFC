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

class TrainSingleSampler():
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.index_list = range(len(self.dataset))
    
    def __getitem__(self):
        input_list = []
        for i in range(self.batch_size):
            # randomly pick one video in whole dataset
            data_index = random.choice(self.index_list)
            # load data
            data = self.dataset.__getitem__(data_index)
            # add data into batch buffer
            input_list.append(data)
            
        return input_list
    
    def __iter__(self):
        # iterable
        return self

    def __next__(self):
        # infinte iteration
        return self.__getitem__()


class TrainDoubleSampler():
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.index_list = range(len(self.dataset))
    
    def __getitem__(self):
        input_list = []
        refer_list = []
        for i in range(self.batch_size):
            # randomly pick one video in whole dataset
            data_index = random.choice(self.index_list)
            # load data
            data = self.dataset.__getitem__(data_index)
            # add data into batch buffer
            input_list.append(data['input'])
            refer_list.append(data['refer'])
        return input_list, refer_list
    
    def __iter__(self):
        # iterable
        return self

    def __next__(self):
        # infinte iteration
        return self.__getitem__()


class TestSingleSampler():
    def __init__(self, dataset, video_factor=1, frame_factor=1, force_video_length=60):
        self.dataset = dataset
        self.video_num = len(self.dataset)
        self.video_factor = video_factor
        self.frame_factor = frame_factor
        self.force_video_length = force_video_length
        print("video nums: ", self.video_num)
    
    def __iter__(self):
        self.video_index = 0
        self.frame_index = 0
        return self
    
    def __next__(self):
        if self.video_index >= self.video_num:
            raise StopIteration
        else:
            # print(f"video-id: {self.video_index}, frame-id: {self.frame_index}")
            data = self.dataset.__getitem__(self.video_index, self.frame_index)
            old_video_index, old_frame_index = self.video_index, self.frame_index
            self.frame_index += self.frame_factor
            if self.frame_index >= self.dataset.json_file['videos'][self.video_index]['length'] or self.frame_index >= self.force_video_length:
                self.frame_index = 0
                self.video_index += self.video_factor
                print("video_index: ", self.video_index)

            return [data], old_video_index, old_frame_index