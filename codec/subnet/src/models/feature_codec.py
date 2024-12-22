import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import time

from .video_net import ME_Spynet, GDN, flow_warp, ResBlock, ResBlock_LeakyReLU_0_Point_1
from ..entropy_models.video_entropy_models import BitEstimator, GaussianEncoder
from ..utils.stream_helper import get_downsampled_shape
from ..layers.layers import MaskedConv2d, subpel_conv3x3

class FeatureCodec(nn.Module):
    def __init__(self, task_head):
        super().__init__()
        
        channel_raw = 256
        channel_feature = 64
        channel_condition = 32
        channel_mv = 64
        channel_y = 96
        channel_z = 64
        channel_mv_y = 64
        channel_mv_z = 64

        # channel compression for mid-feature
        self.featureExtract = FeatureExtract(in_channel=channel_raw, out_channel=channel_feature)
        self.featureRecon = FeatureRecon(in_channel=channel_feature, out_channel=channel_raw)

        # channel compression for high-feature
        self.inputHighFeatureExtract_p2 = TinyFeatureExtract(in_channel=channel_raw, out_channel=channel_condition)
        self.inputHighFeatureExtract_p3 = TinyFeatureExtract(in_channel=channel_raw, out_channel=channel_condition)
        self.inputHighFeatureExtract_p4 = TinyFeatureExtract(in_channel=channel_raw, out_channel=channel_condition)
        self.inputHighFeatureExtract_p5 = nn.Sequential(
                                                TinyFeatureExtract(in_channel=channel_raw, out_channel=channel_condition),
                                                nn.ConvTranspose2d(channel_condition, channel_condition, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.inputHighFeatureExtract_p6 = nn.Sequential(
                                                TinyFeatureExtract(in_channel=channel_raw, out_channel=channel_condition),
                                                nn.ConvTranspose2d(channel_condition, channel_condition, kernel_size=3, stride=2, padding=1, output_padding=1),
                                                nn.ConvTranspose2d(channel_condition, channel_condition, kernel_size=3, stride=2, padding=1, output_padding=1))
        
        self.warpedHighFeatureExtract_p2 = TinyFeatureExtract(in_channel=channel_raw, out_channel=channel_condition)
        self.warpedHighFeatureExtract_p3 = TinyFeatureExtract(in_channel=channel_raw, out_channel=channel_condition)
        self.warpedHighFeatureExtract_p4 = TinyFeatureExtract(in_channel=channel_raw, out_channel=channel_condition)
        self.warpedHighFeatureExtract_p5 = nn.Sequential(
                                                TinyFeatureExtract(in_channel=channel_raw, out_channel=channel_condition),
                                                nn.ConvTranspose2d(channel_condition, channel_condition, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.warpedHighFeatureExtract_p6 = nn.Sequential(
                                                TinyFeatureExtract(in_channel=channel_raw, out_channel=channel_condition),
                                                nn.ConvTranspose2d(channel_condition, channel_condition, kernel_size=3, stride=2, padding=1, output_padding=1),
                                                nn.ConvTranspose2d(channel_condition, channel_condition, kernel_size=3, stride=2, padding=1, output_padding=1))

        self.task_head = task_head

        self.bitEstimator_z = BitEstimator(channel_z)
        self.bitEstimator_z_mv = BitEstimator(channel_mv_z)

        self.gaussian_encoder = GaussianEncoder()
        
        self.contextualEncoder = MultiScaleContextualEncoder(channel_feature*2, channel_y, channel_condition)
        self.contextualDecoder = MultiScaleContextualDecoder(channel_y, channel_feature, condition_channel=channel_condition)

        self.motionEstimation = ME(channel_feature, channel_mv)
        self.motinoCompensation = MC(mv_channel=channel_mv, feature_channel=channel_feature, upsample_factor=4)

        self.mvEncoder = Encoder(channel_mv, channel_mv_y)
        self.mvDecoder = Decoder(channel_mv_y, channel_mv)

        self.priorEncoder = PriorEncoderNet(in_channel=channel_y, out_channel=channel_z)
        self.priorDecoder = PriorDecoderNet(in_channel=channel_z, out_channel=2*channel_y)
        
        self.mvPriorEncoder = PriorEncoderNet(in_channel=channel_mv_y, out_channel=channel_mv_z)
        self.mvPriorDecoder = PriorDecoderNet(in_channel=channel_mv_z, out_channel=2*channel_mv_y)

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(channel_y * 12 // 3, channel_y * 9 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channel_y * 9 // 3, channel_y * 6 // 3, 1),
        )

        self.auto_regressive = MaskedConv2d(channel_y, channel_y, kernel_size=5, padding=2, stride=1)

        self.temporalPriorEncoder = TemporalPriorEncoderNet(channel_feature, channel_y)

        self.mxrange = 150
        self.calrealbits = False

    def feature_probs_based_sigma(self, feature, mean, sigma):
        # outputs = self.quantize(
        #     feature, "dequantize", mean
        # )
        outputs = feature
        values = outputs - mean
        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-5, 1e10)
        
        gaussian = torch.distributions.laplace.Laplace(mu, sigma)
        probs = gaussian.cdf(values + 0.5) - gaussian.cdf(values - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50))
        return total_bits, probs

    def iclr18_estrate_bits_z(self, z):
        prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))
        return total_bits, prob

    def iclr18_estrate_bits_z_mv(self, z_mv):
        prob = self.bitEstimator_z_mv(z_mv + 0.5) - self.bitEstimator_z_mv(z_mv - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))
        return total_bits, prob

    def update(self, force=False):
        self.bitEstimator_z_mv.update(force=force)
        self.bitEstimator_z.update(force=force)
        self.gaussian_encoder.update(force=force)


    def forward_fakerealMAC(self, raw_input_feature_mid, raw_refer_feature_mid):
        # -------------------- Space Transfrom -----------
        
        # mid feature
        input_feature_mid = self.featureExtract(raw_input_feature_mid)
        refer_feature_mid = self.featureExtract(raw_refer_feature_mid)
        input_feature_mid = self.featureExtract(raw_input_feature_mid)
        refer_feature_mid = self.featureExtract(raw_refer_feature_mid)

        # -------------------- Motion Branch --------------
        mv = self.motionEstimation(input_feature=input_feature_mid, refer_feature=refer_feature_mid)
        mv_y = self.mvEncoder(mv)
        if self.training:
            compressed_mv_y = mv_y + (torch.round(mv_y) - mv_y).detach()
        else:
            compressed_mv_y = torch.round(mv_y)
        mv_hat = self.mvDecoder(compressed_mv_y)
        warped_feature = self.motinoCompensation(refer_feature=refer_feature_mid, mv=mv_hat)
        raw_warped_feature_mid = self.featureRecon(warped_feature)
        mv_hat = self.mvDecoder(compressed_mv_y)
        warped_feature = self.motinoCompensation(refer_feature=refer_feature_mid, mv=mv_hat)
        raw_warped_feature_mid = self.featureRecon(warped_feature)

        # -------------------- Prepare the condition for residual compression ---------------
        
        # encoding condition: high feature of input frame 
        raw_input_feature_high = self.task_head(raw_input_feature_mid)
        # channel reduce
        input_feature_high = {'p2': self.inputHighFeatureExtract_p2(raw_input_feature_high['p2']),
                              'p3': self.inputHighFeatureExtract_p3(raw_input_feature_high['p3']),
                              'p4': self.inputHighFeatureExtract_p4(raw_input_feature_high['p4']),
                              'p5': self.inputHighFeatureExtract_p5(raw_input_feature_high['p5']),
                              'p6': self.inputHighFeatureExtract_p6(raw_input_feature_high['p6'])}
        # check the shape
        # print(f"mid: {raw_input_feature_mid.shape}")
        # for key in raw_input_feature_high: 
        #     print(f"raw_input_feature_high {key}: {raw_input_feature_high[key].shape}")
        # for key in input_feature_high: 
        #     print(f"input_feature_high {key}: {input_feature_high[key].shape}")
        # exit(0)
        
        # decoding condition: high feature of warped frame
        raw_warped_feature_high = self.task_head(raw_warped_feature_mid)
        # channel reduce
        warped_feature_high = {'p2': self.warpedHighFeatureExtract_p2(raw_warped_feature_high['p2']),
                               'p3': self.warpedHighFeatureExtract_p3(raw_warped_feature_high['p3']),
                               'p4': self.warpedHighFeatureExtract_p4(raw_warped_feature_high['p4']),
                               'p5': self.warpedHighFeatureExtract_p5(raw_warped_feature_high['p5']),
                               'p6': self.warpedHighFeatureExtract_p6(raw_warped_feature_high['p6'])}
        raw_warped_feature_high = self.task_head(raw_warped_feature_mid)
        # channel reduce
        warped_feature_high = {'p2': self.warpedHighFeatureExtract_p2(raw_warped_feature_high['p2']),
                               'p3': self.warpedHighFeatureExtract_p3(raw_warped_feature_high['p3']),
                               'p4': self.warpedHighFeatureExtract_p4(raw_warped_feature_high['p4']),
                               'p5': self.warpedHighFeatureExtract_p5(raw_warped_feature_high['p5']),
                               'p6': self.warpedHighFeatureExtract_p6(raw_warped_feature_high['p6'])}
        # check the shape
        # print(f"mid: {raw_warped_feature_high.shape}")
        # for key in raw_warped_feature_high: 
        #     print(f"raw_warped_feature_high {key}: {raw_warped_feature_high[key].shape}")
        # for key in warped_feature_high: 
        #     print(f"warped_feature_high {key}: {warped_feature_high[key].shape}")
        # exit(0)

        # -------------------- Contextual (residual) Branch ----------
        
        y = self.contextualEncoder(x=torch.cat((input_feature_mid, warped_feature), dim=1), high_feature=input_feature_high)
        if self.training:
            compressed_y = y + (torch.round(y) - y).detach()
        else:
            compressed_y = torch.round(y)
        input_feature_hat = self.contextualDecoder(y=compressed_y, warped_feature=warped_feature, high_feature=warped_feature_high)
        input_feature_hat = self.contextualDecoder(y=compressed_y, warped_feature=warped_feature, high_feature=warped_feature_high)
        raw_recon_feature_mid = self.featureRecon(input_feature_hat)
        raw_recon_feature_mid = self.featureRecon(input_feature_hat)
        
        # -------------------- entropy model --------------------
        # motion branch:
        mv_z = self.mvPriorEncoder(mv_y)
        if self.training:
            compressed_mv_z = mv_z + (torch.round(mv_z) - mv_z).detach()
        else:
            compressed_mv_z = torch.round(mv_z)
        mv_means_hat, mv_scales_hat = self.mvPriorDecoder(compressed_mv_z).chunk(2, 1)
        mv_means_hat, mv_scales_hat = self.mvPriorDecoder(compressed_mv_z).chunk(2, 1)
        
        # contextual branch 1: Temporal Context
        temporal_prior_params = self.temporalPriorEncoder(warped_feature)
        temporal_prior_params = self.temporalPriorEncoder(warped_feature)
        # contextual branch 2: Mean-Scale Hyper-prior
        z = self.priorEncoder(y)
        if self.training:
            compressed_z = z + (torch.round(z) - z).detach()
        else:
            compressed_z = torch.round(z)
        params = self.priorDecoder(compressed_z)
        params = self.priorDecoder(compressed_z)
        # contextual branch 3: Joint Auto-regressive
        ctx_params = self.auto_regressive(compressed_y)
        # Contextual branch: Final Parameters
        gaussian_params = self.entropy_parameters(
            torch.cat((temporal_prior_params, params, ctx_params), dim=1)
        )
        contextual_means_hat, contextual_scales_hat = gaussian_params.chunk(2, 1)

        # -------------------- loss evaluation --------------------
        # (1) calculate bpp
        total_bits_mv_y, _ = self.feature_probs_based_sigma(compressed_mv_y, mv_means_hat, mv_scales_hat)
        total_bits_mv_z, _ = self.iclr18_estrate_bits_z_mv(compressed_mv_z)    
        total_bits_y, _ = self.feature_probs_based_sigma(compressed_y, contextual_means_hat, contextual_scales_hat)
        total_bits_z, _ = self.iclr18_estrate_bits_z(compressed_z) 
        pixel_num = input_feature_mid.shape[0] * (input_feature_mid.shape[2] * 4) * (input_feature_mid.shape[3] * 4)

        bpp_y = total_bits_y / pixel_num
        bpp_z = total_bits_z / pixel_num
        bpp_mv_y = total_bits_mv_y / pixel_num
        bpp_mv_z = total_bits_mv_z / pixel_num
        bpp = bpp_y + bpp_z + bpp_mv_y + bpp_mv_z
        bpp_dict = {'bpp': bpp, 'bpp_y': bpp_y, 'bpp_z': bpp_z, 'bpp_mv_y': bpp_mv_y, 'bpp_mv_z': bpp_mv_z}

        # (2) calculate distortion
        # mid features
        # mse_mid = torch.mean((raw_recon_feature_mid - raw_input_feature_mid).pow(2))
        # mse_warp = torch.mean((raw_warped_feature_mid - raw_input_feature_mid).pow(2))
        # t0 = time.time()
        # raw_recon_feature_high = self.task_head(raw_recon_feature_mid)
        # t1 = time.time()
        # print("useless time for eval: ", t1 - t0)
        
        # high features
        # mse_p2 = torch.mean((raw_recon_feature_high['p2'] - raw_input_feature_high['p2']).pow(2))
        # mse_p3 = torch.mean((raw_recon_feature_high['p3'] - raw_input_feature_high['p3']).pow(2))
        # mse_p4 = torch.mean((raw_recon_feature_high['p4'] - raw_input_feature_high['p4']).pow(2))
        # mse_p5 = torch.mean((raw_recon_feature_high['p5'] - raw_input_feature_high['p5']).pow(2))
        # mse_p6 = torch.mean((raw_recon_feature_high['p6'] - raw_input_feature_high['p6']).pow(2))
        # mse_high = (mse_p2 + mse_p3 + mse_p4 + mse_p5 + mse_p6) / 5
        # mse_dict = {'mse_mid': mse_mid, 'mse_warp': mse_warp, 'mse_high': mse_high, 'mse_p2': mse_p2, 'mse_p3': mse_p3, 'mse_p4': mse_p4, 'mse_p5': mse_p5, 'mse_p6': mse_p6}

        return raw_recon_feature_mid, raw_warped_feature_mid, None, bpp_dict
    

    def forward(self, raw_input_feature_mid, raw_refer_feature_mid):
        # -------------------- Space Transfrom -----------
        
        # mid feature
        input_feature_mid = self.featureExtract(raw_input_feature_mid)
        refer_feature_mid = self.featureExtract(raw_refer_feature_mid)

        # -------------------- Motion Branch --------------
        mv = self.motionEstimation(input_feature=input_feature_mid, refer_feature=refer_feature_mid)
        mv_y = self.mvEncoder(mv)
        if self.training:
            compressed_mv_y = mv_y + (torch.round(mv_y) - mv_y).detach()
        else:
            compressed_mv_y = torch.round(mv_y)
        mv_hat = self.mvDecoder(compressed_mv_y)
        warped_feature = self.motinoCompensation(refer_feature=refer_feature_mid, mv=mv_hat)
        raw_warped_feature_mid = self.featureRecon(warped_feature)

        # -------------------- Prepare the condition for residual compression ---------------
        
        # encoding condition: high feature of input frame 
        raw_input_feature_high = self.task_head(raw_input_feature_mid)
        # channel reduce
        input_feature_high = {'p2': self.inputHighFeatureExtract_p2(raw_input_feature_high['p2']),
                              'p3': self.inputHighFeatureExtract_p3(raw_input_feature_high['p3']),
                              'p4': self.inputHighFeatureExtract_p4(raw_input_feature_high['p4']),
                              'p5': self.inputHighFeatureExtract_p5(raw_input_feature_high['p5']),
                              'p6': self.inputHighFeatureExtract_p6(raw_input_feature_high['p6'])}
        # check the shape
        # print(f"mid: {raw_input_feature_mid.shape}")
        # for key in raw_input_feature_high: 
        #     print(f"raw_input_feature_high {key}: {raw_input_feature_high[key].shape}")
        # for key in input_feature_high: 
        #     print(f"input_feature_high {key}: {input_feature_high[key].shape}")
        # exit(0)
        
        # decoding condition: high feature of warped frame
        raw_warped_feature_high = self.task_head(raw_warped_feature_mid)
        # channel reduce
        warped_feature_high = {'p2': self.warpedHighFeatureExtract_p2(raw_warped_feature_high['p2']),
                               'p3': self.warpedHighFeatureExtract_p3(raw_warped_feature_high['p3']),
                               'p4': self.warpedHighFeatureExtract_p4(raw_warped_feature_high['p4']),
                               'p5': self.warpedHighFeatureExtract_p5(raw_warped_feature_high['p5']),
                               'p6': self.warpedHighFeatureExtract_p6(raw_warped_feature_high['p6'])}
        # check the shape
        # print(f"mid: {raw_warped_feature_high.shape}")
        # for key in raw_warped_feature_high: 
        #     print(f"raw_warped_feature_high {key}: {raw_warped_feature_high[key].shape}")
        # for key in warped_feature_high: 
        #     print(f"warped_feature_high {key}: {warped_feature_high[key].shape}")
        # exit(0)

        # -------------------- Contextual (residual) Branch ----------
        
        y = self.contextualEncoder(x=torch.cat((input_feature_mid, warped_feature), dim=1), high_feature=input_feature_high)
        if self.training:
            compressed_y = y + (torch.round(y) - y).detach()
        else:
            compressed_y = torch.round(y)
        input_feature_hat = self.contextualDecoder(y=compressed_y, warped_feature=warped_feature, high_feature=warped_feature_high)
        raw_recon_feature_mid = self.featureRecon(input_feature_hat)
        
        # -------------------- entropy model --------------------
        # motion branch:
        mv_z = self.mvPriorEncoder(mv_y)
        if self.training:
            compressed_mv_z = mv_z + (torch.round(mv_z) - mv_z).detach()
        else:
            compressed_mv_z = torch.round(mv_z)
        mv_means_hat, mv_scales_hat = self.mvPriorDecoder(compressed_mv_z).chunk(2, 1)
        

        # contextual branch 1: Temporal Context
        temporal_prior_params = self.temporalPriorEncoder(warped_feature)
        # contextual branch 2: Mean-Scale Hyper-prior
        z = self.priorEncoder(y)
        if self.training:
            compressed_z = z + (torch.round(z) - z).detach()
        else:
            compressed_z = torch.round(z)
        params = self.priorDecoder(compressed_z)
        # contextual branch 3: Joint Auto-regressive
        ctx_params = self.auto_regressive(compressed_y)
        # Contextual branch: Final Parameters
        gaussian_params = self.entropy_parameters(
            torch.cat((temporal_prior_params, params, ctx_params), dim=1)
        )
        contextual_means_hat, contextual_scales_hat = gaussian_params.chunk(2, 1)

        # -------------------- loss evaluation --------------------
        # (1) calculate bpp
        total_bits_mv_y, _ = self.feature_probs_based_sigma(compressed_mv_y, mv_means_hat, mv_scales_hat)
        total_bits_mv_z, _ = self.iclr18_estrate_bits_z_mv(compressed_mv_z)    
        total_bits_y, _ = self.feature_probs_based_sigma(compressed_y, contextual_means_hat, contextual_scales_hat)
        total_bits_z, _ = self.iclr18_estrate_bits_z(compressed_z) 
        pixel_num = input_feature_mid.shape[0] * (input_feature_mid.shape[2] * 4) * (input_feature_mid.shape[3] * 4)

        bpp_y = total_bits_y / pixel_num
        bpp_z = total_bits_z / pixel_num
        bpp_mv_y = total_bits_mv_y / pixel_num
        bpp_mv_z = total_bits_mv_z / pixel_num
        bpp = bpp_y + bpp_z + bpp_mv_y + bpp_mv_z
        bpp_dict = {'bpp': bpp, 'bpp_y': bpp_y, 'bpp_z': bpp_z, 'bpp_mv_y': bpp_mv_y, 'bpp_mv_z': bpp_mv_z}

        # (2) calculate distortion
        # mid features
        mse_mid = torch.mean((raw_recon_feature_mid - raw_input_feature_mid).pow(2))
        mse_warp = torch.mean((raw_warped_feature_mid - raw_input_feature_mid).pow(2))
        t0 = time.time()
        raw_recon_feature_high = self.task_head(raw_recon_feature_mid)
        t1 = time.time()
        print("useless time for eval: ", t1 - t0)
        
        # high features
        mse_p2 = torch.mean((raw_recon_feature_high['p2'] - raw_input_feature_high['p2']).pow(2))
        mse_p3 = torch.mean((raw_recon_feature_high['p3'] - raw_input_feature_high['p3']).pow(2))
        mse_p4 = torch.mean((raw_recon_feature_high['p4'] - raw_input_feature_high['p4']).pow(2))
        mse_p5 = torch.mean((raw_recon_feature_high['p5'] - raw_input_feature_high['p5']).pow(2))
        mse_p6 = torch.mean((raw_recon_feature_high['p6'] - raw_input_feature_high['p6']).pow(2))
        mse_high = (mse_p2 + mse_p3 + mse_p4 + mse_p5 + mse_p6) / 5
        mse_dict = {'mse_mid': mse_mid, 'mse_warp': mse_warp, 'mse_high': mse_high, 'mse_p2': mse_p2, 'mse_p3': mse_p3, 'mse_p4': mse_p4, 'mse_p5': mse_p5, 'mse_p6': mse_p6}

        return raw_recon_feature_mid, raw_warped_feature_mid, mse_dict, bpp_dict

# +-------------------------------------------------------------------------------------------+
# |                            Sub-Module Defination                                          |
# +-------------------------------------------------------------------------------------------+

class PriorEncoderNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PriorEncoderNet, self).__init__()
        self.l1 = nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1)
        self.r1 = nn.LeakyReLU(inplace=True)
        self.l2 = nn.Conv2d(out_channel, out_channel, 5, stride=2, padding=2)
        self.r2 = nn.LeakyReLU(inplace=True)
        self.l3 = nn.Conv2d(out_channel, out_channel, 5, stride=2, padding=2)

    def forward(self, x):
        x = self.r1(self.l1(x))
        x = self.r2(self.l2(x))
        x = self.l3(x)
        return x

class PriorDecoderNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PriorDecoderNet, self).__init__()
        self.l1 = nn.ConvTranspose2d(in_channel, in_channel, 5, stride=2, padding=2, output_padding=1)
        self.r1 = nn.LeakyReLU(inplace=True)
        self.l2 = nn.ConvTranspose2d(in_channel, in_channel, 5, stride=2, padding=2, output_padding=1)
        self.r2 = nn.LeakyReLU(inplace=True)
        self.l3 = nn.ConvTranspose2d(in_channel, out_channel, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.r1(self.l1(x))
        x = self.r2(self.l2(x))
        x = self.l3(x)
        return x


class TemporalPriorEncoderNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(TemporalPriorEncoderNet, self).__init__()
        self.l1 = nn.Conv2d(in_channel, in_channel, 5, stride=2, padding=2)
        self.r1 = GDN(in_channel)
        self.l2 = nn.Conv2d(in_channel, out_channel, 5, stride=2, padding=2)
        self.r2 = GDN(out_channel)
        
    def forward(self, x):
        x = self.r1(self.l1(x))
        x = self.r2(self.l2(x))
        return x
    

class DepthwiseConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super(DepthwiseConv2d, self).__init__()
        self.depth_wise = nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, padding=kernel_size//2, groups=in_channel)
        self.point_wise = nn.Conv2d(in_channel, out_channel, kernel_size=1)

    def forward(self, x):
        x = self.depth_wise(x)
        x = self.point_wise(x)
        return x


class DepthResblock(nn.Module):
    def __init__(self, channel, kernel_size):
        super(DepthResblock, self).__init__()
        self.conv1 = DepthwiseConv2d(channel, channel, kernel_size)
        self.relu = nn.LeakyReLU(inplace=False)
        self.conv2 = DepthwiseConv2d(channel, channel, kernel_size)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)
        return x + y


class TinyResblock(nn.Module):
    def __init__(self, channel, kernel_size):
        super(TinyResblock, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel_size, stride=1, padding=kernel_size//2)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size, stride=1, padding=kernel_size//2) 

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)
        return x + y


class FeatureExtract(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FeatureExtract, self).__init__()
        mid_channel_1 = (2 * in_channel + out_channel) // 3
        mid_channel_2 = (in_channel + 2 * out_channel) // 3

        self.conv1 = nn.Conv2d(in_channel, mid_channel_1, 3, stride=1, padding=1)
        self.res1 = TinyResblock(mid_channel_1, 3)
        self.conv2 = nn.Conv2d(mid_channel_1, mid_channel_2, 3, stride=1, padding=1)
        self.res2 = TinyResblock(mid_channel_2, 3)
        self.conv3 = nn.Conv2d(mid_channel_2, out_channel, 3, stride=1, padding=1)
        self.res3 = TinyResblock(out_channel, 3)

    def forward(self, x):
        y = self.conv1(x)
        y = self.res1(y)
        y = self.conv2(y)
        y = self.res2(y)
        y = self.conv3(y)
        y = self.res3(y)
        return y
    

class FeatureRecon(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FeatureRecon, self).__init__()
        mid_channel_1 = (2 * in_channel + out_channel) // 3
        mid_channel_2 = (in_channel + 2 * out_channel) // 3

        self.conv1 = subpel_conv3x3(in_channel, mid_channel_1)
        self.res1 = TinyResblock(mid_channel_1, 3)
        self.conv2 = subpel_conv3x3(mid_channel_1, mid_channel_2)
        self.res2 = TinyResblock(mid_channel_2, 3)
        self.conv3 = subpel_conv3x3(mid_channel_2, out_channel)
        self.res3 = TinyResblock(out_channel, 3)

    def forward(self, x):
        y = self.conv1(x)
        y = self.res1(y)
        y = self.conv2(y)
        y = self.res2(y)
        y = self.conv3(y)
        y = self.res3(y)
        return y


class TinyFeatureExtract(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(TinyFeatureExtract, self).__init__()
        mid_channel = (in_channel + out_channel) // 2
        self.conv1 = nn.Conv2d(in_channel, mid_channel, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(mid_channel, out_channel, 3, stride=1, padding=1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ME(nn.Module):
    def __init__(self, feature_channel, mv_channel):
        super(ME, self).__init__()
        # down sampling
        self.res1 = TinyResblock(feature_channel*2, 3)
        self.conv2 = nn.Conv2d(feature_channel*2, mv_channel, 7, stride=2, padding=3)
        self.res2 = TinyResblock(mv_channel, 3)
        self.conv3 = nn.Conv2d(mv_channel, mv_channel, 7, stride=2, padding=3)
        self.res3 = TinyResblock(mv_channel, 3)
        # up sampling
        self.deconv1 = nn.ConvTranspose2d(mv_channel, mv_channel, 7, padding=3, output_padding=1, stride=2)
        self.deconv2 = nn.ConvTranspose2d(mv_channel, mv_channel, 7, padding=3, output_padding=1, stride=2)
        # skip connection
        self.trans1 = nn.Conv2d(feature_channel*2, mv_channel, 1, stride=1, padding=0)
        self.trans2 = nn.Conv2d(mv_channel, mv_channel, 1, stride=1, padding=0)
        self.trans3 = nn.Conv2d(mv_channel, mv_channel, 1, stride=1, padding=0)

    def forward(self, input_feature, refer_feature):
        # down-sampling
        x = torch.cat((input_feature, refer_feature), dim=1)
        # print(f"x: {x.shape}")
        t1 = self.res1(x)
        # print(f"t1: {t1.shape}")

        t2 = self.conv2(t1)
        t2 = self.res2(t2)
        # print(f"t2: {t2.shape}")

        t3 = self.conv3(t2)
        t3 = self.res3(t3)
        # print(f"t3: {t3.shape}")
        # print(f"(after down-sampling)t1: {t1.shape}, t2: {t2.shape}, t3: {t3.shape}")
        # channel fuse
        t1 = self.trans1(t1)
        t2 = self.trans2(t2)
        t3 = self.trans3(t3)
        # print(f"(after channel-fuse)t1: {t1.shape}, t2: {t2.shape}, t3: {t3.shape}")
        # up-sampling
        y = self.deconv1(t3) + t2
        y = self.deconv2(y) + t1
        # print(f"<ME> y:{y.shape}")
        return y


class MC(nn.Module):
    def __init__(self, mv_channel, feature_channel, upsample_factor):
        super(MC, self).__init__()
        self.conv1 = nn.Conv2d(mv_channel, mv_channel, kernel_size=7, stride=1, padding=3)
        self.res1 = TinyResblock(mv_channel, kernel_size=3)
        self.conv2 = DepthwiseConv2d(mv_channel, mv_channel*upsample_factor, kernel_size=5)
        self.res2 = DepthResblock(mv_channel*upsample_factor, kernel_size=3)
        self.conv3 = DepthwiseConv2d(mv_channel*upsample_factor, mv_channel*upsample_factor*upsample_factor, kernel_size=5)
        self.res3 = DepthResblock(mv_channel*upsample_factor*upsample_factor, kernel_size=3)
        self.pixelConv = nn.Conv2d(mv_channel, feature_channel*upsample_factor*upsample_factor, kernel_size=1)
        self.warpConv = nn.Conv2d(feature_channel*upsample_factor*upsample_factor, feature_channel, kernel_size=7, padding=3)
    
    def forward(self, refer_feature, mv):
        # print(f"refer_feature: {refer_feature.shape}")
        refer_feature = self.conv1(refer_feature)
        # print(f"refer_feature: {refer_feature.shape}")
        refer_feature = self.res1(refer_feature)
        # print(f"refer_feature: {refer_feature.shape}")
        refer_feature = self.conv2(refer_feature)
        # print(f"refer_feature: {refer_feature.shape}")
        refer_feature = self.res2(refer_feature)
        # print(f"refer_feature: {refer_feature.shape}")
        refer_feature = self.conv3(refer_feature)
        # print(f"refer_feature: {refer_feature.shape}")
        refer_feature = self.res3(refer_feature)
        mv = self.pixelConv(mv)
        # print(f'(before warp)mv: {mv.shape}, refer_feature: {refer_feature.shape}')
        warped_feature = self.warpConv(mv * refer_feature)
        # print(f'warped_feature: {warped_feature.shape}')
        return warped_feature
    
    def forward_with_schemes(self, refer_feature, mv):
        # print(f"refer_feature: {refer_feature.shape}")
        refer_feature = self.conv1(refer_feature)
        # print(f"refer_feature: {refer_feature.shape}")
        refer_feature = self.res1(refer_feature)
        # print(f"refer_feature: {refer_feature.shape}")
        refer_feature = self.conv2(refer_feature)
        # print(f"refer_feature: {refer_feature.shape}")
        refer_feature = self.res2(refer_feature)
        # print(f"refer_feature: {refer_feature.shape}")
        refer_feature = self.conv3(refer_feature)
        # print(f"refer_feature: {refer_feature.shape}")
        refer_feature = self.res3(refer_feature)
        mv = self.pixelConv(mv)
        # print(f'(before warp)mv: {mv.shape}, refer_feature: {refer_feature.shape}')
        warped_feature = self.warpConv(mv * refer_feature)
        # print(f'warped_feature: {warped_feature.shape}')
        return warped_feature, refer_feature


class MultiScaleContextualEncoder(nn.Module):
    def __init__(self, in_channel, out_channel, condition_channel):
        super(MultiScaleContextualEncoder, self).__init__()
        # channel reduce
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1)
        self.gdn1 = GDN(out_channel)
        self.res1 = ResBlock_LeakyReLU_0_Point_1(out_channel)
        
        # shape 1/4 -> 1/8
        self.conv2 = nn.Conv2d(out_channel+condition_channel, out_channel, 5, stride=2, padding=2)
        self.gdn2 = GDN(out_channel)
        self.res2 = ResBlock_LeakyReLU_0_Point_1(out_channel)
        
        # chape 1/8 -> 1/16
        self.conv3 = nn.Conv2d(out_channel+condition_channel, out_channel, 5, stride=2, padding=2)
        self.gdn3 = GDN(out_channel)
        self.res3 = ResBlock_LeakyReLU_0_Point_1(out_channel)
        
        # fuse p4, p5, p6 to condition_channel=32
        self.conv_fuse_p4_to_p6 = nn.Sequential(
                                    nn.Conv2d(3*condition_channel, 2*condition_channel, 5, stride=1, padding=2),
                                    nn.Conv2d(2*condition_channel, condition_channel, 5, stride=1, padding=2)
                                )

        # channel-wise conditional fuse
        self.conv4 = nn.Conv2d(out_channel+condition_channel, out_channel, 5, stride=1, padding=2)
        self.gdn4 = GDN(out_channel)
        self.res4 = ResBlock_LeakyReLU_0_Point_1(out_channel)

    def forward(self, x, high_feature):
        # (1) condition: extracted high feature
        # channel = 32
        p2 = high_feature['p2']
        # channel = 32
        p3 = high_feature['p3']
        # channel = 32
        p4_p6 = self.conv_fuse_p4_to_p6(torch.cat((high_feature['p4'], high_feature['p5'], high_feature['p6']), dim=1))

        # (2) encoding process
        # channel alignment
        x = self.res1(self.gdn1(self.conv1(x)))
        # conditional spatial down-sampling
        x = torch.cat((x, p2), dim=1)
        x = self.res2(self.gdn2(self.conv2(x)))
        # conditional spatial down-sampling
        x = torch.cat((x, p3), dim=1)
        x = self.res3(self.gdn3(self.conv3(x)))
        # conditional channel alignment
        x = torch.cat((x, p4_p6), dim=1)
        x = self.res4(self.gdn4(self.conv4(x)))

        return x

class MultiScaleContextualDecoder(nn.Module):
    def __init__(self, in_channle, feature_channel, condition_channel):
        super(MultiScaleContextualDecoder, self).__init__()
        # (1) fuse p4, p5, p6 to condition_channel=32
        self.conv_fuse_p4_to_p6 = nn.Sequential(
                                    nn.Conv2d(3*condition_channel, 2*condition_channel, 5, stride=1, padding=2),
                                    nn.Conv2d(2*condition_channel, condition_channel, 5, stride=1, padding=2)
                                )
        
        # (2) decode process 1
        # shape 1/16 -> 1/8
        self.deconv1 = nn.ConvTranspose2d(in_channle+condition_channel, in_channle, 5, padding=2, output_padding=1, stride=2)
        self.igdn1 = GDN(in_channle, inverse=True)
        self.res1 = ResBlock_LeakyReLU_0_Point_1(in_channle)
        # chape 1/8 -> 1/4
        self.deconv2 = nn.ConvTranspose2d(in_channle+condition_channel, in_channle, 5, padding=2, output_padding=1, stride=2)
        self.igdn2 = GDN(in_channle, inverse=True)
        self.res2 = ResBlock_LeakyReLU_0_Point_1(in_channle)
        # channel alignment
        self.deconv3 = nn.ConvTranspose2d(in_channle+condition_channel, in_channle, 5, padding=2, stride=1)
        self.igdn3 = GDN(in_channle, inverse=True)
        self.res3 = ResBlock_LeakyReLU_0_Point_1(in_channle)

        # (3) decode process 2
        # channel alignment
        self.deconv4 = nn.ConvTranspose2d(in_channle+feature_channel, feature_channel, 3, padding=1, stride=1)
        self.igdn4 = GDN(feature_channel, inverse=True)
        self.res4 = ResBlock_LeakyReLU_0_Point_1(feature_channel)
    
    def forward(self, y, warped_feature, high_feature):
        # (1) condition: extracted high feature
        # channel = 32
        p2 = high_feature['p2']
        # channel = 32
        p3 = high_feature['p3']
        # channel = 32
        p4_p6 = self.conv_fuse_p4_to_p6(torch.cat((high_feature['p4'], high_feature['p5'], high_feature['p6']), dim=1))

        # (2) decode process 1
        y = self.res1(self.igdn1(self.deconv1(torch.cat((y, p4_p6), dim=1))))
        y = self.res2(self.igdn2(self.deconv2(torch.cat((y, p3), dim=1))))
        y = self.res3(self.igdn3(self.deconv3(torch.cat((y, p2), dim=1))))
        
        # (3) decode process 2
        y = self.res4(self.igdn4(self.deconv4(torch.cat((y, warped_feature), dim=1))))
        return y


class Encoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Encoder, self).__init__()
        # channel reduce
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1)
        self.gdn1 = GDN(out_channel)
        self.res1 = ResBlock_LeakyReLU_0_Point_1(out_channel)
        # shape 1/4 -> 1/8
        self.conv2 = nn.Conv2d(out_channel, out_channel, 5, stride=2, padding=2)
        self.gdn2 = GDN(out_channel)
        self.res2 = ResBlock_LeakyReLU_0_Point_1(out_channel)
        # chape 1/8 -> 1/16
        self.conv3 = nn.Conv2d(out_channel, out_channel, 5, stride=2, padding=2)
        self.gdn3 = GDN(out_channel)
        self.res3 = ResBlock_LeakyReLU_0_Point_1(out_channel)

    def forward(self, x):
        x = self.res1(self.gdn1(self.conv1(x)))
        x = self.res2(self.gdn2(self.conv2(x)))
        x = self.res3(self.gdn3(self.conv3(x)))
        return x


class Decoder(nn.Module):
    def __init__(self, in_channle, out_channel):
        super(Decoder, self).__init__()
        # chape 1/16 -> 1/8
        self.deconv1 = nn.ConvTranspose2d(in_channle, in_channle, 5, padding=2, output_padding=1, stride=2)
        self.igdn1 = GDN(in_channle, inverse=True)
        self.res1 = ResBlock_LeakyReLU_0_Point_1(in_channle)
        # chape 1/8 -> 1/4
        self.deconv2 = nn.ConvTranspose2d(in_channle, in_channle, 5, padding=2, output_padding=1, stride=2)
        self.igdn2 = GDN(in_channle, inverse=True)
        self.res2 = ResBlock_LeakyReLU_0_Point_1(in_channle)
        # channel increase
        self.deconv3 = nn.ConvTranspose2d(in_channle, out_channel, 3, padding=1, stride=1)
        self.igdn3 = GDN(out_channel, inverse=True)
        self.res3 = ResBlock_LeakyReLU_0_Point_1(out_channel)
    
    def forward(self, y):
        y = self.res1(self.igdn1(self.deconv1(y)))
        y = self.res2(self.igdn2(self.deconv2(y)))
        y = self.res3(self.igdn2(self.deconv3(y)))
        return y

