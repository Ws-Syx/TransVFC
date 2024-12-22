# basic package
import numpy as np
import os, json, cv2, random, tqdm, math
import matplotlib.pyplot as plt
import logging
import os
from collections import OrderedDict
import sys

# detectron package
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
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

# pytorch package
import torch
from torch.nn import functional as F
from torchvision.models.segmentation import deeplabv3_resnet50
from torch.nn.functional import interpolate

# self-defined package
sys.path.append("..")
from codec.subnet.src.models.feature_codec import FeatureCodec
from _transfer import FeatureSpaceTransfer
from _utils import get_specific_cfg, load_model, freeze_all_parameters

# Deeplabv3 head-part: Initial layers up to the end of layer1
class DeeplabHead(torch.nn.Module):
    def __init__(self, original_model):
        super(DeeplabHead, self).__init__()
        self.initial = torch.nn.Sequential(
            original_model.backbone.conv1,
            original_model.backbone.bn1,
            original_model.backbone.relu,
            original_model.backbone.maxpool,
            original_model.backbone.layer1
        )

    def forward(self, x):
        return self.initial(x)

# Deeplabv3 tail-part: Remaining layers, including ASPP and upsampling layers
class DeeplabTail(torch.nn.Module):
    def __init__(self, original_model):
        super(DeeplabTail, self).__init__()
        self.layer2 = original_model.backbone.layer2
        self.layer3 = original_model.backbone.layer3
        self.layer4 = original_model.backbone.layer4
        self.classifier = original_model.classifier
        self.aux = original_model.aux_classifier

    def forward(self, res2):
        input_shape = (res2.shape[-2]*4, res2.shape[-1]*4)
        res3 = self.layer2(res2)
        res4 = self.layer3(res3)
        res5 = self.layer4(res4)
        
        mask = self.classifier(res5)
        mask = F.interpolate(mask, size=input_shape, mode="bilinear", align_corners=False)
        result = {'out': mask,
                  'res2': res2, 
                  'res3': res3,
                  'res4': res4,
                  'res5': res5}
        return result


class ModelForTrain():
    def __init__(self, deeplab_model):
        # faster-rcnn head-part
        self.cfg_head = get_specific_cfg('head')
        self.faster_head = build_model(self.cfg_head); self.faster_head.train()
        checkpointer = DetectionCheckpointer(self.faster_head)
        checkpointer.load(self.cfg_head.MODEL.WEIGHTS)
        freeze_all_parameters(self.faster_head)

        # faster-rcnn tail-part
        self.cfg_tail = get_specific_cfg('tail')
        self.faster_tail = build_model(self.cfg_tail); self.faster_tail.train()
        checkpointer = DetectionCheckpointer(self.faster_tail)
        checkpointer.load(self.cfg_tail.MODEL.WEIGHTS)
        freeze_all_parameters(self.faster_tail)

        # deeplabv3 head-part
        self.deeplab_head = DeeplabHead(deeplab_model)
        self.deeplab_head.eval()
        freeze_all_parameters(self.deeplab_head)

        # deeplabv3 tail-part
        self.deeplab_tail = DeeplabTail(deeplab_model)
        self.deeplab_tail.train()
        freeze_all_parameters(self.deeplab_tail)

        # feature codec
        self.factor = 16
        self.codec = FeatureCodec(self.faster_tail.backbone).cuda()
        freeze_all_parameters(self.codec)

        # feature transfer
        self.transfer = FeatureSpaceTransfer().cuda()

        # data pre-process
        self.cfg = get_specific_cfg('gt')
        self.aug = T.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST], self.cfg.INPUT.MAX_SIZE_TEST
        )


    def evaluate_feature_mse(self, output, mid_gt):
        output_gt = self.deeplab_tail(mid_gt)
        result = {}
        result['res3'] = torch.mean((output['res3'] - output_gt['res3']) ** 2)
        result['res4'] = torch.mean((output['res4'] - output_gt['res4']) ** 2)
        result['res5'] = torch.mean((output['res5'] - output_gt['res5']) ** 2)
        result['high'] = (result['res3'] + result['res4'] + result['res5']) / 3
        result['mid'] = torch.mean((output['res2'] - output_gt['res2']) ** 2)
        return result

    
    def __call__(self, input_dict_list0, input_dict_list1):
        with EventStorage(0) as temp:
            with torch.no_grad():
                # (1) mid-features extractor
                # extract faster-rcnn's mid feature
                faster_mid0 = self.faster_head(input_dict_list0)['res2']
                faster_mid1 = self.faster_head(input_dict_list1)['res2']

                # extract deeplab's mid feature
                input_image0 = self.faster_head.preprocess_image(input_dict_list0).tensor
                input_image1 = self.faster_head.preprocess_image(input_dict_list1).tensor
                input_mask0 = self.faster_head.preprocess_sem_mask(input_dict_list0).tensor
                input_mask1 = self.faster_head.preprocess_sem_mask(input_dict_list1).tensor
                deeplab_mid0 = self.deeplab_head(input_image0)
                deeplab_mid1 = self.deeplab_head(input_image1)
                assert faster_mid0.shape == deeplab_mid0.shape, "The shape of feastre_mid is not equal to deeplab_mid"

                # reshape to a valid shape for codec
                # mid-features
                old_h, old_w = faster_mid0.shape[-2:]
                new_h, new_w = round(old_h / self.factor) * self.factor, round(old_w / self.factor) * self.factor
                
                faster_mid0 = interpolate(faster_mid0, size=(new_h, new_w), mode='bilinear', align_corners=True)
                faster_mid1 = interpolate(faster_mid1, size=(new_h, new_w), mode='bilinear', align_corners=True)

                # (2) codec
                recon_faster_mid0, _, _, bpp0 = self.codec(faster_mid0, faster_mid1)
                recon_faster_mid1, _, _, bpp1 = self.codec(faster_mid1, faster_mid0)
            
            # (3) transfer
            recon_deeplab_mid0, recon_image0 = self.transfer(recon_faster_mid0)
            recon_deeplab_mid1, recon_image1 = self.transfer(recon_faster_mid1)

            # reshape to original shape
            recon_image0 = interpolate(recon_image0, size=(old_h*4, old_w*4), mode='bilinear', align_corners=True)
            recon_image1 = interpolate(recon_image1, size=(old_h*4, old_w*4), mode='bilinear', align_corners=True)
            faster_mid0 = interpolate(faster_mid0, size=(old_h, old_w), mode='bilinear', align_corners=True)
            faster_mid1 = interpolate(faster_mid1, size=(old_h, old_w), mode='bilinear', align_corners=True)
            recon_deeplab_mid0 = interpolate(recon_deeplab_mid0, size=(old_h, old_w), mode='bilinear', align_corners=True)
            recon_deeplab_mid1 = interpolate(recon_deeplab_mid1, size=(old_h, old_w), mode='bilinear', align_corners=True)

            # (4) task-tail
            output0 = self.deeplab_tail(recon_deeplab_mid0)           
            output1 = self.deeplab_tail(recon_deeplab_mid1)

            # (5) get loss function
            # loss of down-stream machine vision task
            loss0 = torch.nn.CrossEntropyLoss()(output0['out'], input_mask0.long())
            loss1 = torch.nn.CrossEntropyLoss()(output1['out'], input_mask1.long())
            task_loss = (loss0 + loss1) / 2
            # loss of transfered feature mse
            mse_dict0 = self.evaluate_feature_mse(output0, deeplab_mid0)
            mse_dict1 = self.evaluate_feature_mse(output1, deeplab_mid1)
            mse_dict = {key: (mse_dict0[key] + mse_dict1[key]) / 2 for key in mse_dict0}
            # loss of of temproy-created image mse
            mse_image0 = torch.mean((recon_image0 - input_image0 / 255.0) ** 2)
            mse_image1 = torch.mean((recon_image1 - input_image1 / 255.0) ** 2)
            mse_image = (mse_image0 + mse_image1) / 2
            
            return {'mse_mid': mse_dict['mid'],
                    'mse_high': mse_dict['high'],
                    'mse_image': mse_image,
                    'task_loss': task_loss,
                    'bpp': (bpp0['bpp'] + bpp1['bpp']) / 2}


class ModelForTest():
    def __init__(self, deeplab_model):
        # faster-rcnn head-part
        self.cfg_head = get_specific_cfg('head')
        self.model_head = build_model(self.cfg_head); self.model_head.eval()
        checkpointer = DetectionCheckpointer(self.model_head)
        checkpointer.load(self.cfg_head.MODEL.WEIGHTS)

        # faster-rcnn tail-part
        self.cfg_tail = get_specific_cfg('tail')
        self.model_tail = build_model(self.cfg_tail); self.model_tail.eval()
        checkpointer = DetectionCheckpointer(self.model_tail)
        checkpointer.load(self.cfg_tail.MODEL.WEIGHTS)

        # deeplabv3 head-part
        self.deeplab_head = DeeplabHead(deeplab_model)
        self.deeplab_head.eval()
        freeze_all_parameters(self.deeplab_head)

        # deeplabv3 tail-part
        self.deeplab_tail = DeeplabTail(deeplab_model)
        self.deeplab_tail.eval()
        freeze_all_parameters(self.deeplab_tail)

        # continues feature codec
        self.codec = FeatureCodec(task_head=self.model_tail.backbone).cuda()
        
        # feature space transfer
        self.transfer = FeatureSpaceTransfer().cuda()

        # data pre-process
        self.cfg = get_specific_cfg('gt')
        self.aug = T.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST], self.cfg.INPUT.MAX_SIZE_TEST
        )
        self.resize_factor = 16
    

    def evaluate_feature_mse(self, output, mid_gt):
        output_gt = self.deeplab_tail(mid_gt)
        result = {}
        result['res3'] = torch.mean((output['res3'] - output_gt['res3']) ** 2)
        result['res4'] = torch.mean((output['res4'] - output_gt['res4']) ** 2)
        result['res5'] = torch.mean((output['res5'] - output_gt['res5']) ** 2)
        result['high'] = (result['res3'] + result['res4'] + result['res5']) / 3
        result['mid'] = torch.mean((output['res2'] - output_gt['res2']) ** 2)
        return result
    

    def __call__(self, inputs, refer_mid_features):
        with torch.no_grad():
            
            # (1) inference-head
            # feature extract
            mask = self.model_head.preprocess_sem_mask([inputs]).tensor
            image = self.model_head.preprocess_image([inputs]).tensor
            input_mid_features = self.model_head([inputs])['res2']
            deeplab_mid_features = self.deeplab_head(image)
            assert input_mid_features.shape == deeplab_mid_features.shape

            # (2) resize the input
            old_h, old_w = input_mid_features.shape[-2], input_mid_features.shape[-1]
            new_h, new_w = int(old_h // self.resize_factor * self.resize_factor), int(old_w // self.resize_factor * self.resize_factor)
            # resize mid-feature to a valid shape
            input_mid_features = interpolate(input_mid_features, size=(new_h, new_w), mode='bilinear', align_corners=True)
            refer_mid_features = interpolate(refer_mid_features, size=(new_h, new_w), mode='bilinear', align_corners=True) if refer_mid_features != None else None
            
            # (3) feature encode and decode
            if refer_mid_features is not None:
                recon_mid_features, warped_mid_features, mse_dict, bpp_dict = self.codec(input_mid_features, refer_mid_features)
            else:
                recon_mid_features = input_mid_features
                mse_dict = bpp_dict = None
            
            # (4) transfer
            transfered_mid_features, recon_image = self.transfer(recon_mid_features)

            # (5) inference-tail: forward rest of network
            recon_mid_features = interpolate(recon_mid_features, size=(old_h, old_w), mode='bilinear', align_corners=True)
            transfered_mid_features = interpolate(transfered_mid_features, size=(old_h, old_w), mode='bilinear', align_corners=True)
            predictions = self.deeplab_tail(transfered_mid_features)
            
            # evaluation on p-frame
            if refer_mid_features is not None:
                mse_dict = self.evaluate_feature_mse(predictions, deeplab_mid_features)
            
            return predictions['out'], transfered_mid_features, recon_mid_features, mse_dict, bpp_dict