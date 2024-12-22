

# import some common libraries
import numpy as np
import os
import json
import cv2
import random
import tqdm
import matplotlib.pyplot as plt
import argparse
import sys
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np

# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.modeling import build_model

# import torch's package
import torch
from torch.nn.functional import interpolate
from torchvision.models.segmentation import deeplabv3_resnet50

# import self-defined package
from _utils import load_model, AverageMetric, get_specific_cfg, calculate_miou, calculate_iou, calculate_pixel_accuracy
from _dataset import TestSequenceDataset
from _model import ModelForTest


def parse_args(in_args=None):
    parser = argparse.ArgumentParser(description='')
    
    # load a training check-point
    # valid_jpeg_path = "/opt/data/private/syx/dataset/ytvis2019/valid/JPEGImages"
    # valid_json_path = "/opt/data/private/syx/dataset/ytvis2019/valid/instances_val_sub_GT.json"
    parser.add_argument('--valid-jpeg-path', type=str, default="/opt/data/private/syx/dataset/VSPW/selected-valid/data")
    parser.add_argument('--valid-json-path', type=str, default="/opt/data/private/syx/dataset/VSPW/selected-valid/valid.json")
    parser.add_argument('--need-codec', type=str, default="True")
    parser.add_argument('--pretrained-codec', type=str)
    parser.add_argument('--pretrained-transfer', type=str)
    
    return parser.parse_args(in_args)


def build_deeplabv3():
    model = deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = torch.nn.Conv2d(256, 125, kernel_size=(1, 1), stride=(1, 1))  # For main classifier
    model.aux_classifier[4] = torch.nn.Conv2d(256, 125, kernel_size=(1, 1), stride=(1, 1))  # For auxiliary classifier
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    load_model(model, "/opt/data/private/syx/Deeplab-envi/detectron2_v4/ckpt/snapshot/iter70000.model")
    model.eval()
    return model


if __name__ == "__main__":
    args = parse_args()
    cfg = get_specific_cfg('gt')

    # create dataset for test
    video_path = args.valid_jpeg_path
    mask_path = args.valid_jpeg_path
    json_path = args.valid_json_path
    test_dataset = TestSequenceDataset(cfg=cfg, json_path=json_path, mask_path=mask_path, video_path=video_path)

    # create model for test
    model = ModelForTest(build_deeplabv3())
    load_model(model.codec, args.pretrained_codec)
    load_model(model.transfer, args.pretrained_transfer)

    # prepare for dataset
    label_dict = json.load(open('/opt/data/private/syx/dataset/VSPW/VSPW-480p/label_num_dic_final.json'))
    chosen_label = ['person', 'sky', 'ground', 'grass', 'tree', 'car', 'crosswalk', 'bus', 'house', 'traffic_light']
    chosen_index = [int(label_dict[i]) for i in chosen_label]

    # test based on iteration
    bpp_metric = AverageMetric()
    mse_mid_metric = AverageMetric()
    mse_high_metric = AverageMetric()
    pa_result = AverageMetric()
    all_miou_dict = {}
    for i in tqdm.tqdm(range(len(test_dataset))):
        data = test_dataset.__getitem__(i)
        
        for t in range(len(data)):
            # special case for I-frame
            if t % 12 == 0:    
                refer_mid_feature = None

            # inference
            predicted_mask, _, recon_mid_feature, mse_dict, bpp_dict = model(data[t], refer_mid_feature)
            refer_mid_feature = recon_mid_feature

            # evaluation on mse and bpp
            if mse_dict is not None and bpp_dict is not None:
                for key in mse_dict: 
                    mse_dict[key] = round(float(mse_dict[key]), 4)
                for key in bpp_dict: 
                    bpp_dict[key] = round(float(bpp_dict[key]), 4)
                    
                bpp_metric.add(bpp_dict['bpp'])
                mse_mid_metric.add(mse_dict['mid'])
                mse_high_metric.add(mse_dict['high'])
            
            # evaluation on mask
            origin_mask = model.model_head.preprocess_sem_mask([data[t]]).tensor.cpu().numpy()
            predicted_mask = predicted_mask.argmax(dim=1).cpu().numpy()[0]
            # evaluation  mIOU
            iou_list = calculate_iou(predicted_mask, origin_mask, 125)
            if i not in all_miou_dict:
                all_miou_dict[i] = {}
            all_miou_dict[i][t] = {'iou_list': iou_list,
                                   'miou': calculate_miou(iou_list),
                                   'chosen_miou': calculate_miou([iou_list[i] for i in chosen_index])}
            # evaluation pixel accuracy
            pixel_accuracy = calculate_pixel_accuracy(predicted_mask, origin_mask)
            pa_result.add(pixel_accuracy)
            print("chosen_miou: ", calculate_miou([iou_list[i] for i in chosen_index]))

    # calculate the mIOU and chosen-mIOU in each video
    for video_index in all_miou_dict:
        miou = np.mean(np.array([i['miou'] for i in all_miou_dict[video_index].values()]))
        chosend_miou = np.mean(np.array([i['chosen_miou'] for i in all_miou_dict[video_index].values()]))
        all_miou_dict[video_index]['miou'] = miou
        all_miou_dict[video_index]['chosen_miou'] = chosend_miou
        
    # print to console
    mean_miou = np.mean(np.array([i['miou'] for i in all_miou_dict.values()]))
    mean_chosen_miou = np.nanmean(np.array([i['chosen_miou'] for i in all_miou_dict.values()]))
    print(f"mean_miou: {mean_miou}")
    print(f"mean_chosen_miou: {mean_chosen_miou}")
    print(f"pa_result: {pa_result.avg()}")
    print(f"bpp: {bpp_metric.avg()}")
    print(f"mse_mid: {mse_mid_metric.avg()}")
    print(f"mse_high: {mse_high_metric.avg()}")    