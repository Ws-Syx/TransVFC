

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


# from model import FeatureSpaceTransfer

# import sys
# sys.path.append("..")
# from codec.subnet.src.models.feature_codec import FeatureCodec

        
def convert_mse_to_psnr(x):
    return 10 * math.log10(1 / x)
        

class AverageMetric():
    def __init__(self):
        self.data_list = []
    def add(self, x):
        self.data_list.append(x)
    def avg(self):
        return torch.mean(torch.tensor(self.data_list))

def freeze_all_parameters(net):
    for p in net.parameters():
        p.requires_grad = False

def get_specific_cfg(case):
    assert case in {'head', 'tail', 'gt'}
    
    # config from local file
    cfg = get_cfg()
    if case in {'head', 'tail'}:
        cfg.merge_from_file(f"/opt/data/private/syx/FastRCNN-envi/configs/COCO-Detection/faster_rcnn_R_50_FPN_1x_ytvis_{case}.yaml")
    else:
        cfg.merge_from_file(f"/opt/data/private/syx/FastRCNN-envi/configs/COCO-Detection/faster_rcnn_R_50_FPN_1x_ytvis.yaml")
    
    # basic config
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = "../ckpt/faster-rcnn-ytvis.pth"
    cfg.SOLVER.IMS_PER_BATCH = 4 # This is the real "batch size" commonly known to deep learning people
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 40
    
    return cfg


def save_model(model, iter):
    if not os.path.exists('../ckpt/snapshot'):
        os.makedirs('../ckpt/snapshot')
    torch.save(model.state_dict(), "../ckpt/snapshot/iter{}.model".format(iter))


def load_model(model, f, need_count=False):
    # print(f"load pretrained weight: {f}")
    # print("load DCVC format")
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)
        model_dict = model.state_dict()
        result_dict = {}
        
        for key, value in model_dict.items():
            # print(f"successfully load key: {key}")
            result_dict[key] = pretrained_dict[key]
        
        model_dict.update(result_dict)
        model.load_state_dict(model_dict)

    # calculate the total number of finished training
    if need_count:
        f = str(f)
        if f.find('iter') != -1 and f.find('.model') != -1:
            st = f.find('iter') + 4
            ed = f.find('.model', st)
            return int(f[st:ed]) 
        else:
            print("Error in <load_model>")
            exit(0)


def get_detail(epoch, training_details):
    finished_epoch = 0
    for detail in training_details:
        if finished_epoch <= epoch < finished_epoch + detail['epoch']:
            return detail
        finished_epoch += detail['epoch']
    return training_details[-1]


def calculate_iou(pred_mask, gt_mask, num_classes):
    iou_list = []
    for cls in range(num_classes):
        pred_cls = pred_mask == cls
        gt_cls = gt_mask == cls
        intersection = np.logical_and(pred_cls, gt_cls).sum()
        union = np.logical_or(pred_cls, gt_cls).sum()
        if union == 0:
            # Avoid division by zero
            iou = float('nan')
        else:
            iou = intersection / union
        iou_list.append(iou)
    return iou_list


def calculate_miou(iou_list):
    # Remove NaN values before calculating mean
    valid_iou = [iou for iou in iou_list if not np.isnan(iou)]
    miou = np.mean(valid_iou)
    # print("number of classes: ", len(valid_iou))
    return miou


def calculate_pixel_accuracy(pred_mask, gt_mask):
    correct = np.sum(pred_mask == gt_mask)
    total = pred_mask.size
    pixel_accuracy = correct / total
    return pixel_accuracy