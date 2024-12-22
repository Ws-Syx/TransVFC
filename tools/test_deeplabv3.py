from detectron2.config import get_cfg
# from detectron2.engine import default_argument_parser, default_setup, default_writers, launch

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from tensorboardX import SummaryWriter


import matplotlib.pyplot as plt
import numpy as np
import json, tqdm, argparse

# self-defined package
from utils import load_model, save_model, get_specific_cfg, AverageMetric
from dataset import TestSingleDataset, TestSingleSampler

def parse_args():
    parser = argparse.ArgumentParser(description="Example testing script")
    parser.add_argument('--json-path', type=str)
    parser.add_argument('--mask-path', type=str)
    parser.add_argument('--video-path', type=str)
    args = parser.parse_args()
    return args


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


if __name__ == "__main__":
    args = parse_args()

    # create the model and load the pretrained weight
    model = deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = torch.nn.Conv2d(256, 125, kernel_size=(1, 1), stride=(1, 1))  # For main classifier
    model.aux_classifier[4] = torch.nn.Conv2d(256, 125, kernel_size=(1, 1), stride=(1, 1))  # For auxiliary classifier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    load_model(model, "/opt/data/private/syx/Deeplab-envi/detectron2_v4/ckpt/snapshot/iter70000.model")
    model.eval()

    # build the dataset and dataloader
    cfg = get_specific_cfg('head')
    train_dataset = TestSingleDataset(cfg=cfg,
                                json_path=args.json_path,
                                video_path=args.video_path,
                                mask_path=args.mask_path,
                                randomly_flip=False)
    train_loader = TestSingleSampler(train_dataset, video_factor=1, frame_factor=1, force_video_length=24)
    pre_processor = build_model(cfg)

    # open the label-index json file
    label_dict = json.load(open('/opt/data/private/syx/dataset/VSPW/VSPW-480p/label_num_dic_final.json'))
    chosen_label = ['person', 'sky', 'ground', 'grass', 'tree', 'car', 'crosswalk', 'bus', 'house', 'traffic_light']
    chosen_index = [int(label_dict[i]) for i in chosen_label]

    all_miou_dict = {}
    pa_result = AverageMetric()

    for data, video_index, frame_index in tqdm.tqdm(train_loader):
        
        # pre-process to get data
        mask = pre_processor.preprocess_sem_mask(data).tensor
        image = pre_processor.preprocess_image(data).tensor

        # inference
        prediction = model(image)
        
        # post-process to get result
        # get gt mask
        origin_mask = mask[0].cpu().numpy()
        # get predicted mask
        predicted_mask = prediction['out'].argmax(dim=1)
        predicted_mask = predicted_mask.cpu().numpy()[0]

        # evaluation
        # (1) mIOU
        iou_list = calculate_iou(predicted_mask, origin_mask, 125)
        if video_index not in all_miou_dict:
            all_miou_dict[video_index] = {}
        all_miou_dict[video_index][frame_index] = {'iou_list': iou_list,
                                                'miou': calculate_miou(iou_list),
                                                'chosen_miou': calculate_miou([iou_list[i] for i in chosen_index])}
        # (2) Pixel Accuary
        pixel_accuracy = calculate_pixel_accuracy(predicted_mask, origin_mask)
        pa_result.add(pixel_accuracy)

    # calculate the mIOU and chosen-mIOU in each video
    for video_index in all_miou_dict:
        miou = np.mean(np.array([i['miou'] for i in all_miou_dict[video_index].values()]))
        chosend_miou = np.mean(np.array([i['chosen_miou'] for i in all_miou_dict[video_index].values()]))
        all_miou_dict[video_index]['miou'] = miou
        all_miou_dict[video_index]['chosen_miou'] = chosend_miou

    # print to console
    mean_miou = np.mean(np.array([i['miou'] for i in all_miou_dict.values()]))
    mean_chosen_miou = np.nanmean(np.array([i['chosen_miou'] for i in all_miou_dict.values()]))
    print("mean_miou: ", mean_miou)
    print("mean_chosen_miou: ", mean_chosen_miou)
    print("pa_result: ", pa_result.avg())