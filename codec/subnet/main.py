# -*- coding: gbk -*-
import os
import argparse
import torch
import cv2
import logging
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import sys
import math
import json
import datetime
from pytorch_msssim import ms_ssim
from dataset import DataSet, UVGDataSet
from tensorboardX import SummaryWriter
from drawuvg import uvgdrawplt
from src.models.DCVC_net import *

torch.backends.cudnn.enabled = True
gpu_num = torch.cuda.device_count()
num_workers = gpu_num*4
print('gpu_num:', gpu_num)
cur_lr = base_lr = 0.0001  # * gpu_num
train_lambda = 2048
print_step = 100
cal_step = 10
# print_step = 10
warmup_step = 0  # // gpu_num
gpu_per_batch = 4
test_step = 10000  # // gpu_num
tot_epoch = 100
tot_step = 3000000
decay_interval = 1800000
lr_decay = 0.1
logger = logging.getLogger("VideoCompression")
tb_logger = None
global_step = 0
ifarr = 0
ifout = 0
recon_path = 'recon/recon.bin'

once_strings = []
def print_once(strings):
    if strings in once_strings:
        return
    print(strings)
    once_strings.append(strings)
print_once("=== main ===")

def geti(lamb):
    if lamb == 2048:
        return 'H265L20'
    elif lamb == 1024:
        return 'H265L23'
    elif lamb == 512:
        return 'H265L26'
    elif lamb == 256:
        return 'H265L29'
    else:
        print("cannot find lambda : %d"%(lamb))
        exit(0)

ref_i_dir = geti(train_lambda)


parser = argparse.ArgumentParser(description='FVC reimplement')

parser.add_argument('-l', '--log', default='',
                    help='output training details')
parser.add_argument('-p', '--pretrain', default='',
                    help='load pretrain model')
parser.add_argument('--test', action='store_true')
parser.add_argument('--testuvg', action='store_true')
parser.add_argument('--testvtl', action='store_true')
parser.add_argument('--testmcl', action='store_true')
parser.add_argument('--testauc', action='store_true')
parser.add_argument('--rerank', action='store_true')
parser.add_argument('--allpick', action='store_true')
parser.add_argument('--config', dest='config', required=True,
                    help='hyperparameter of Reid in json format')


def parse_config(config):
    config = json.load(open(args.config))
    global tot_epoch, tot_step, test_step, base_lr, cur_lr, lr_decay, decay_interval, train_lambda, ref_i_dir, ifend, iftest, ifmsssim, msssim_lambda, ifout
    if 'tot_epoch' in config:
        tot_epoch = config['tot_epoch']
    if 'tot_step' in config:
        tot_step = config['tot_step']
    if 'test_step' in config:
        test_step = config['test_step']
        print('teststep : ', test_step)
    if 'train_lambda' in config:
        train_lambda = config['train_lambda']
        ref_i_dir = geti(train_lambda)
    if 'lr' in config:
        if 'base' in config['lr']:
            base_lr = config['lr']['base']
            cur_lr = base_lr
        if 'decay' in config['lr']:
            lr_decay = config['lr']['decay']
        if 'decay_interval' in config['lr']:
            decay_interval = config['lr']['decay_interval']
    if 'ifend' in config:
        ifend = config['ifend']
    if 'iftest' in config:
        iftest = config['iftest']
    if 'ifmsssim' in config:
        ifmsssim = config['ifmsssim']
    if 'msssim_lambda' in config:
        msssim_lambda = config['msssim_lambda']
    if 'ifout' in config:
        ifout = config['ifout']


def adjust_learning_rate(optimizer, global_step):
    global cur_lr
    global warmup_step
    if global_step < warmup_step:
        lr = base_lr * global_step / warmup_step
    elif global_step < decay_interval:  # // gpu_num:
        lr = base_lr
    else:
        lr = base_lr * (lr_decay ** (global_step // decay_interval))
    cur_lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def Var(x):
    return Variable(x.cuda())


def save_image_tensor2cv2(input_tensor: torch.Tensor, filename):
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    
    input_tensor = input_tensor.to(torch.device('cpu'))

    input_tensor = input_tensor.squeeze()
    
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    
    input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, input_tensor)


def testuvg(global_step, testfull=False):
    with torch.no_grad():
        test_loader = DataLoader(dataset=test_dataset, shuffle=False, num_workers=4, batch_size=1, pin_memory=True)
        net.eval()

        sumbpp = 0
        sumpsnr = 0
        summsssim = 0
        cnt = 0
        t0 = datetime.datetime.now()
        for batch_idx, input in enumerate(test_loader):
            if batch_idx % 10 == 0:
                print("testing : %d/%d" % (batch_idx, len(test_loader)))
                if batch_idx > 0:
                    t1 = datetime.datetime.now()
                    deltatime = t1 - t0
                    deltatime = deltatime.seconds + 1e-6 * deltatime.microseconds
                    info = "sumavgbpp: %.6lf  sumavgpsnr: %.6lf  time: %.6lf"%(sumbpp/cnt, sumpsnr/cnt, deltatime)
                    print(info)
                    t0 = t1
                    if testfull == False and batch_idx/10 > 3:
                        return
            input_images = input[0]
            ref_image = input[1]
            ref_bpp = input[2]
            ref_psnr = input[3]
            ref_msssim = input[4]
            seqlen = input_images.size()[1]
            sumbpp += torch.mean(ref_bpp).detach().numpy()
            sumpsnr += torch.mean(ref_psnr).detach().numpy()
            summsssim += torch.mean(ref_msssim).detach().numpy()
            cnt += 1
            for i in range(seqlen):
                input_image = input_images[:, i, :, :, :]
                inputframe, refframe = Var(input_image), Var(ref_image)

                clipped_recon_image, distortion1, distortion2, bpp_y, bpp_z, bpp_mv_y, bpp_mv_z, bpp = net(
                    inputframe, refframe)
                if ifarr == 1:
                    clipped_recon_image, distortion2, bpp = net.encode_decode(refframe, inputframe, recon_path)
                
                bpp_c = torch.mean(bpp).cpu().detach().numpy()
                psnr_c = torch.mean(10 * (torch.log(1. / distortion2) / np.log(10))).cpu().detach().numpy()
                msssim_c = ms_ssim(clipped_recon_image.cpu().detach(), input_image, data_range=1.0,
                                     size_average=True).numpy()
                sumbpp += bpp_c
                sumpsnr += psnr_c
                summsssim += msssim_c
                cnt += 1
                ref_image = clipped_recon_image
                
                if ifout==1:
                    if (batch_idx % 5 == 0) & (i == 1):
                        recon_path = "recon/"
                        img_name = 'recon-' + str(batch_idx) + '-1.png'
                        save_image_tensor2cv2(ref_image, os.path.join(recon_path, img_name))
                        img_name = 'input-' + str(batch_idx) + '-1.png'
                        save_image_tensor2cv2(input_image, os.path.join(recon_path, img_name))
                

        log = "global step %d : " % (global_step) + "\n"
        logger.info(log)
        sumbpp /= cnt
        sumpsnr /= cnt
        summsssim /= cnt
        log = "UVGdataset : average bpp : %.6lf, average psnr : %.6lf, average msssim: %.6lf\n" % (
            sumbpp, sumpsnr, summsssim)
        logger.info(log)
        uvgdrawplt([sumbpp], [sumpsnr], [summsssim], global_step, testfull=testfull)


def train(epoch, global_step):
    print("epoch", epoch)
    global gpu_per_batch
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, num_workers=num_workers, batch_size=gpu_per_batch,
                              pin_memory=True)
    net.train()

    global optimizer
    global cur_lr #
    bat_cnt = 0
    cal_cnt = 0
    sumloss = 0
    sumpsnr = 0
    sumfeatpsnr = 0
    sumbpp = 0
    sumbpp_feature = 0
    sumbpp_z = 0
    # sumbpp_offset = 0
    sumbpp_mv_y = 0
    sumbpp_mv_z = 0
    tot_iter = len(train_loader)
    t0 = datetime.datetime.now()
    pre_stage_process = 0
    pre_sum_param = 0

    # param_flow_part = net.opticFlow.parameters()
    for batch_idx, input in enumerate(train_loader):
        global_step += 1
        bat_cnt += 1
        input_image, ref_image = Var(input[0]), Var(input[1])
        quant_noise_feature, quant_noise_z, quant_noise_mv = Var(input[2]), Var(input[3]), Var(input[4])
        quant_noise_zmv = Var(input[5])
        # ta = datetime.datetime.now()
        clipped_recon_image, distortion1, distortion2, bpp_y, bpp_z, bpp_mv_y, bpp_mv_z, bpp = net(input_image,
                                                                                 ref_image,
                                                                                 quant_noise_feature,
                                                                                 quant_noise_z,
                                                                                 quant_noise_mv,
                                                                                 quant_noise_zmv)

        # tb = datetime.datetime.now()

        distortion1, distortion2, bpp_y, bpp_z, bpp_mv_y, bpp_mv_z, bpp = torch.mean(distortion1), torch.mean(
            distortion2), torch.mean(bpp_y), torch.mean(bpp_z), torch.mean(bpp_mv_y), torch.mean(
                bpp_mv_z), torch.mean(bpp)

        
        if stage_progress == 0:
            rd_loss = train_lambda * distortion1 + bpp_mv_y + bpp_mv_z
        elif stage_progress == 1:
            rd_loss = train_lambda * distortion2
        elif stage_progress == 2:
            rd_loss = train_lambda * distortion2 + bpp_y + bpp_z
        else:
            rd_loss = train_lambda * distortion2 + bpp
        
        optimizer.zero_grad()
        rd_loss.backward()
        
        def clip_gradient(optimizer, grad_clip):
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        param.grad.data.clamp_(-grad_clip, grad_clip)
        clip_gradient(optimizer, 0.5)
        optimizer.step()

        if global_step % cal_step == 0:
            cal_cnt += 1
            if distortion1 > 0:
                feat_psnr = 10 * (torch.log(1 * 1 / distortion1) / np.log(10)).cpu().detach().numpy()
            else:
                feat_psnr = 100
            if distortion2 > 0:
                psnr = 10 * (torch.log(1 * 1 / distortion2) / np.log(10)).cpu().detach().numpy()
            else:
                psnr = 100
            
            loss_ = rd_loss.cpu().detach().numpy()

            sumloss += loss_
            sumpsnr += psnr
            sumfeatpsnr += feat_psnr
            sumbpp += bpp.cpu().detach()
            sumbpp_feature += bpp_y.cpu().detach()
            sumbpp_z += bpp_z.cpu().detach()
            sumbpp_mv_y += bpp_mv_y.cpu().detach()
            sumbpp_mv_z += bpp_mv_z.cpu().detach()

        if (batch_idx % print_step) == 0 and bat_cnt > 1:
            tb_logger.add_scalar('lr', cur_lr, global_step)
            tb_logger.add_scalar('rd_loss', sumloss / cal_cnt, global_step)
            tb_logger.add_scalar('psnr', sumpsnr / cal_cnt, global_step)
            tb_logger.add_scalar('feat_psnr', sumfeatpsnr / cal_cnt, global_step)
            tb_logger.add_scalar('bpp', sumbpp / cal_cnt, global_step)
            tb_logger.add_scalar('bpp_feature', sumbpp_feature / cal_cnt, global_step)
            tb_logger.add_scalar('bpp_z', sumbpp_z / cal_cnt, global_step)
            tb_logger.add_scalar('bpp_offset', sumbpp_mv_y / cal_cnt, global_step)
            tb_logger.add_scalar('bpp_offset', sumbpp_mv_z / cal_cnt, global_step)
            t1 = datetime.datetime.now()
            deltatime = t1 - t0
            log = 'Train Epoch : {:02} [{:4}/{:4} ({:3.0f}%)] Avgloss:{:.6f} lr:{} time:{}'.format(epoch, batch_idx,
                                                                                                   len(train_loader),
                                                                                                   100. * batch_idx / len(
                                                                                                       train_loader),
                                                                                                   sumloss / cal_cnt,
                                                                                                   cur_lr, (
                                                                                                           deltatime.seconds + 1e-6 * deltatime.microseconds) / bat_cnt)
            print(log)
            log = 'details : featpsnr: {:.2f} psnr : {:.2f} '.format(sumfeatpsnr / cal_cnt,sumpsnr / cal_cnt)
            print(log)
            log = 'bpp : {:.6f}  bpp_y: {:.6f}   bpp_z: {:.6f}  bpp_mv_y : {:.6f}  bpp_mv_z : {:.6f}'.format(sumbpp / cal_cnt,
                                                                                                            sumbpp_feature / cal_cnt,
                                                                                                            sumbpp_z / cal_cnt,
                                                                                                            sumbpp_mv_y / cal_cnt,
                                                                                                            sumbpp_mv_z / cal_cnt,
                                                                                                            )
            print(log)
            bat_cnt = 0
            cal_cnt = 0
            sumbpp = sumbpp_feature = sumbpp_z = sumbpp_mv_y = sumbpp_mv_z = sumloss = sumpsnr = sumfeatpsnr = 0
            t0 = t1
    log = 'Train Epoch : {:02} Loss:\t {:.6f}\t lr:{}'.format(epoch, sumloss / bat_cnt, cur_lr)
    logger.info(log)
    return global_step


if __name__ == "__main__":
    args = parser.parse_args()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s] %(message)s')
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    if args.log != '':
        filehandler = logging.FileHandler(args.log)
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    logger.setLevel(logging.INFO)
    logger.info("FVC training")
    logger.info("config : ")
    logger.info(open(args.config).read())
    parse_config(args.config)
    
    model = DCVC_net()

    if args.pretrain != '':
        print("loading pretrain : ", args.pretrain)
        global_step = load_model(model, args.pretrain)
    net = model.cuda()
    
    net = torch.nn.DataParallel(net, list(range(gpu_num)))
    bp_parameters = net.parameters()
    optimizer = optim.Adam(bp_parameters, lr=base_lr)
    
    global train_dataset, test_dataset
    if args.testuvg:
        test_dataset = UVGDataSet(refdir=ref_i_dir, testfull=True)
        print('testing UVG')
        testuvg(global_step, testfull=True)
        exit(0)

    tb_logger = SummaryWriter('./events')
    train_dataset = DataSet("../../dataset/vimeo_septuplet/test.txt")
    test_dataset = UVGDataSet(refdir=ref_i_dir, testfull=True)
    stepoch = global_step // (train_dataset.__len__() // (gpu_per_batch))  # * gpu_num))

    stage_progress_4 = [80765*4, 80765*7, 80765*10, 80765*16, 80765*24]
    stage_progress = len(stage_progress_4)-1
    pre_stage_process = 0
    lrs = [1e-4, 1e-4, 1e-4, 1e-4, 1e-5]
    for epoch in range(stepoch, tot_epoch):
        
        for i in range(len(stage_progress_4)):
            if global_step < stage_progress_4[i] - 1:
                stage_progress = i
                break 
        
        
        def Change_optim(stage):
            def freezeMV(flag):
                for p in net.module.opticFlow.parameters():
                    p.requires_grad = flag
                for p in net.module.mvEncoder.parameters():
                    p.requires_grad = flag
                for p in net.module.mvDecoder_part1.parameters():
                    p.requires_grad = flag
                for p in net.module.mvDecoder_part2.parameters():
                    p.requires_grad = flag
            if stage == 0:
                freezeMV(True)
            elif stage == 1:
                freezeMV(False)
            elif stage == 2:
                freezeMV(False)
            else:
                freezeMV(True)
        
        print("Processing training step: ", stage_progress+1, "/ 5")
        Change_optim(stage_progress)
        cur_lr = lrs[stage_progress]
            
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr = cur_lr)
        pre_stage_process = stage_progress

        if global_step > tot_step:
            save_model(model, global_step)
            print("Finish training")
            break
        global_step = train(epoch, global_step)
        save_model(model, global_step)

        if global_step > 80765*14:
            testuvg(global_step, testfull=True)
