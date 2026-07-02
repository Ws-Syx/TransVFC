"""
FeatureCodec: Neural Video Feature Compression

A learnable codec for compressing intermediate features in video understanding
pipelines. Compresses frame features by leveraging motion estimation and
multi-scale contextual coding with a hyperprior entropy model.

Usage:
    from feature_codec import FeatureCodec

    model = FeatureCodec(task_head=my_task_head)
    recon_feature, warped_feature, mse_dict, bpp_dict = model(
        raw_input_feature_mid, raw_refer_feature_mid
    )

Reference:
    "Feature Compression for Video Understanding" (coming soon)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# +===========================================================================+
# |                       Low-Level Primitives                                |
# +===========================================================================+

class LowerBound(torch.autograd.Function):
    """Lower-bound constraint via STE (straight-through estimator)."""
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones_like(inputs) * bound
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors
        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0
        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


class GDN(nn.Module):
    """Generalized Divisive Normalization.

    Args:
        ch: number of channels.
        inverse: if True, applies inverse GDN (IGDN).
    """
    def __init__(self, ch, inverse=False, beta_min=1e-6, gamma_init=0.1,
                 reparam_offset=2 ** -18):
        super().__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = reparam_offset
        self.build(ch)

    def build(self, ch):
        self.pedestal = self.reparam_offset ** 2
        self.beta_bound = (self.beta_min + self.reparam_offset ** 2) ** 0.5
        self.gamma_bound = self.reparam_offset

        beta = torch.sqrt(torch.ones(ch) + self.pedestal)
        self.beta = nn.Parameter(beta)

        eye = torch.eye(ch)
        g = self.gamma_init * eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)
        self.gamma = nn.Parameter(gamma)

    def forward(self, inputs):
        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size()
            inputs = inputs.view(bs, ch, d * w, h)

        _, ch, _, _ = inputs.size()

        beta = LowerBound.apply(self.beta, self.beta_bound)
        beta = beta ** 2 - self.pedestal

        gamma = LowerBound.apply(self.gamma, self.gamma_bound)
        gamma = gamma ** 2 - self.pedestal
        gamma = gamma.view(ch, ch, 1, 1)

        norm_ = F.conv2d(inputs ** 2, gamma, beta)
        norm_ = torch.sqrt(norm_)

        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)
        return outputs


class ResBlock_LeakyReLU_0_Point_1(nn.Module):
    """Residual block with LeakyReLU(0.1) activations."""
    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(d_model, d_model, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return x + self.conv(x)


def subpel_conv3x3(in_ch, out_ch, r=1):
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=3, padding=1),
        nn.PixelShuffle(r),
    )


# +===========================================================================+
# |                      Entropy Coding Primitives                            |
# +===========================================================================+


def _pmf_to_quantized_cdf(pmf, precision=16):
    """Pure-Python PMF → quantized CDF (replaces the C++ MLCodec_CXX path).

    Converts a probability mass function into cumulative counts over a
    ``2**precision`` grid suitable for rANS encoding.
    """
    pmf = [float(p) for p in pmf]
    factor = 1 << precision         # e.g. 65536
    freqs = [max(1, int(round(p * factor))) for p in pmf]
    # Adjust so sum(freqs) == factor
    diff = factor - sum(freqs)
    if abs(diff) >= len(freqs):
        # Large discrepancy; redistribute proportionally
        total = sum(pmf)
        freqs = [max(1, int(round(p / total * factor))) for p in pmf]
        diff = factor - sum(freqs)
    # Distribute remainder to first few entries
    step = 1 if diff > 0 else -1
    i = 0
    while diff != 0 and i < 10000:
        freqs[i % len(freqs)] += step
        diff -= step
        i += 1
    cdf = [0]
    for f in freqs:
        cdf.append(cdf[-1] + f)
    cdf[-1] = factor
    return cdf


class EntropyCoder:
    """Light-weight entropy coder state — only what is needed for ``update()``.

    The ``compress`` / ``decompress`` methods require compiled C++ rANS
    backends (``MLCodec_rans`` / ``MLCodec_CXX``) which are *not* included
    in this open-source release.  The ``update()`` path (PMF → CDF) uses a
    pure-Python fallback and works without the C++ modules.
    """
    def __init__(self, entropy_coder_precision=16):
        self.entropy_coder_precision = int(entropy_coder_precision)
        self._offset = None
        self._quantized_cdf = None
        self._cdf_length = None

    @staticmethod
    def pmf_to_quantized_cdf(pmf, precision=16):
        cdf = _pmf_to_quantized_cdf(pmf, precision)
        return torch.IntTensor(cdf)

    def pmf_to_cdf(self, pmf, tail_mass, pmf_length, max_length):
        cdf = torch.zeros((len(pmf_length), max_length + 2), dtype=torch.int32)
        for i, p in enumerate(pmf):
            prob = torch.cat((p[: pmf_length[i]], tail_mass[i]), dim=0)
            _cdf = self.pmf_to_quantized_cdf(prob, self.entropy_coder_precision)
            cdf[i, : _cdf.size(0)] = _cdf
        return cdf

    def set_cdf_states(self, offset, quantized_cdf, cdf_length):
        self._offset = offset
        self._quantized_cdf = quantized_cdf
        self._cdf_length = cdf_length

    # --- compress / decompress stubs ---------------------------------------
    # These methods require the C++ rANS backend and are omitted from the
    # open-source class definition.  CDF table computation (used by training
    # and model-update) is fully functional.
    # -----------------------------------------------------------------------


class Bitparm(nn.Module):
    """Trainable per-channel parameter modulation for the bit estimator."""
    def __init__(self, channel, final=False):
        super().__init__()
        self.final = final
        self.h = nn.Parameter(torch.nn.init.normal_(
            torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        self.b = nn.Parameter(torch.nn.init.normal_(
            torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        self.a = (None if final else
                  nn.Parameter(torch.nn.init.normal_(
                      torch.empty(channel).view(1, -1, 1, 1), 0, 0.01)))

    def forward(self, x):
        if self.final:
            return torch.sigmoid(x * F.softplus(self.h) + self.b)
        x = x * F.softplus(self.h) + self.b
        return x + torch.tanh(x) * torch.tanh(self.a)


class BitEstimator(nn.Module):
    """Factorised-prior entropy model (non-parametric).

    Learns a 1-D cumulative distribution per channel for bottleneck latents
    (e.g. the hyper-latent ``z``).
    """
    def __init__(self, channel):
        super().__init__()
        self.f1 = Bitparm(channel)
        self.f2 = Bitparm(channel)
        self.f3 = Bitparm(channel)
        self.f4 = Bitparm(channel, final=True)
        self.channel = channel
        self.entropy_coder = None

    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        return self.f4(x)

    def update(self, force=False):
        if self.entropy_coder is not None and not force:
            return
        self.entropy_coder = EntropyCoder()
        with torch.no_grad():
            device = next(self.parameters()).device
            medians = torch.zeros(self.channel, device=device)

            minima = medians + 50
            for i in range(50, 1, -1):
                samples = torch.zeros_like(medians) - i
                samples = samples[None, :, None, None]
                probs = self.forward(samples).squeeze()
                minima = torch.where(probs < torch.zeros_like(medians) + 0.0001,
                                     torch.zeros_like(medians) + i, minima)

            maxima = medians + 50
            for i in range(50, 1, -1):
                samples = torch.zeros_like(medians) + i
                samples = samples[None, :, None, None]
                probs = self.forward(samples).squeeze()
                maxima = torch.where(probs > torch.zeros_like(medians) + 0.9999,
                                     torch.zeros_like(medians) + i, maxima)

            minima = minima.int()
            maxima = maxima.int()
            offset = -minima
            pmf_start = medians - minima
            pmf_length = maxima + minima + 1
            max_length = pmf_length.max()

            samples = torch.arange(max_length, device=device)
            samples = samples[None, :] + pmf_start[:, None, None]
            half = 0.5
            lower = self.forward(samples - half).squeeze(0)
            upper = self.forward(samples + half).squeeze(0)
            pmf = upper - lower
            pmf = pmf[:, 0, :]
            tail_mass = lower[:, 0, :1] + (1.0 - upper[:, 0, -1:])

            quantized_cdf = self.entropy_coder.pmf_to_cdf(
                pmf, tail_mass, pmf_length, max_length)
            cdf_length = pmf_length + 2
            self.entropy_coder.set_cdf_states(offset, quantized_cdf, cdf_length)

    @staticmethod
    def build_indexes(size):
        N, C, H, W = size
        indexes = torch.arange(C).view(1, -1, 1, 1).int()
        return indexes.repeat(N, 1, H, W)

    def compress(self, x):
        indexes = self.build_indexes(x.size())
        return self.entropy_coder.compress(x, indexes)

    def decompress(self, strings, size):
        output_size = (1, self.entropy_coder._quantized_cdf.size(0),
                       size[0], size[1])
        indexes = self.build_indexes(output_size)
        return self.entropy_coder.decompress(strings, indexes)


class GaussianEncoder:
    """Scale-adaptive entropy coder for Laplace-distributed latents.

    Maintains a table of pre-computed quantised CDFs for different scale
    values so that symbols can be encoded / decoded with a scale-index
    lookup.
    """
    def __init__(self):
        self.scale_table = self.get_scale_table()
        self.entropy_coder = None

    @staticmethod
    def get_scale_table(min=0.01, max=16, levels=64):  # pylint: disable=W0622
        return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

    def update(self, force=False):
        if self.entropy_coder is not None and not force:
            return
        self.entropy_coder = EntropyCoder()

        pmf_center = torch.zeros_like(self.scale_table) + 50
        scales = torch.zeros_like(pmf_center) + self.scale_table
        mu = torch.zeros_like(scales)
        gaussian = torch.distributions.laplace.Laplace(mu, scales)
        for i in range(50, 1, -1):
            samples = torch.zeros_like(pmf_center) + i
            probs = gaussian.cdf(samples).squeeze()
            pmf_center = torch.where(probs > torch.zeros_like(pmf_center) + 0.9999,
                                     torch.zeros_like(pmf_center) + i, pmf_center)

        pmf_center = pmf_center.int()
        pmf_length = 2 * pmf_center + 1
        max_length = torch.max(pmf_length).item()

        device = pmf_center.device
        samples = torch.arange(max_length, device=device) - pmf_center[:, None]
        samples = samples.float()
        scales = torch.zeros_like(samples) + self.scale_table[:, None]
        mu = torch.zeros_like(scales)
        gaussian = torch.distributions.laplace.Laplace(mu, scales)
        upper = gaussian.cdf(samples + 0.5)
        lower = gaussian.cdf(samples - 0.5)
        pmf = upper - lower
        tail_mass = 2 * lower[:, :1]

        quantized_cdf = self.entropy_coder.pmf_to_cdf(
            pmf, tail_mass, pmf_length, max_length)
        self.entropy_coder.set_cdf_states(
            -pmf_center, quantized_cdf, pmf_length + 2)

    def build_indexes(self, scales):
        scales = torch.maximum(scales, torch.zeros_like(scales) + 1e-5)
        indexes = scales.new_full(scales.size(),
                                  len(self.scale_table) - 1).int()
        for s in self.scale_table[:-1]:
            indexes -= (scales <= s).int()
        return indexes

    def compress(self, x, scales):
        indexes = self.build_indexes(scales)
        return self.entropy_coder.compress(x, indexes)

    def decompress(self, strings, scales):
        indexes = self.build_indexes(scales)
        return self.entropy_coder.decompress(strings, indexes)

    def set_stream(self, stream):
        self.entropy_coder.set_stream(stream)

    def decode_stream(self, scales):
        indexes = self.build_indexes(scales)
        return self.entropy_coder.decode_stream(indexes)


# +===========================================================================+
# |                     FeatureCodec Sub-Modules                              |
# +===========================================================================+

class PriorEncoderNet(nn.Module):
    """Hyper-prior encoder:  spatial down-sampling for the bottleneck."""
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.l1 = nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1)
        self.r1 = nn.LeakyReLU(inplace=True)
        self.l2 = nn.Conv2d(out_channel, out_channel, 3, stride=2, padding=1)
        self.r2 = nn.LeakyReLU(inplace=True)
        self.l3 = nn.Conv2d(out_channel, out_channel, 3, stride=2, padding=1)

    def forward(self, x):
        x = self.r1(self.l1(x))
        x = self.r2(self.l2(x))
        x = self.l3(x)
        return x


class PriorDecoderNet(nn.Module):
    """Hyper-prior decoder: spatial up-sampling from the hyper-latent."""
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.l1 = nn.ConvTranspose2d(in_channel, in_channel, 3, stride=2,
                                     padding=1, output_padding=1)
        self.r1 = nn.LeakyReLU(inplace=True)
        self.l2 = nn.ConvTranspose2d(in_channel, in_channel, 3, stride=2,
                                     padding=1, output_padding=1)
        self.r2 = nn.LeakyReLU(inplace=True)
        self.l3 = nn.ConvTranspose2d(in_channel, out_channel, 3, stride=1,
                                     padding=1)

    def forward(self, x):
        x = self.r1(self.l1(x))
        x = self.r2(self.l2(x))
        x = self.l3(x)
        return x


class TemporalPriorEncoderNet(nn.Module):
    """Multi-scale temporal prior: fuses warped feature + multi-scale
    conditioning to produce entropy parameters for the current frame."""
    def __init__(self, feature_channel, condition_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(feature_channel + condition_channel, 96,
                               kernel_size=3, stride=2, padding=1)
        self.gdn1 = GDN(96)
        self.conv2 = nn.Conv2d(96 + condition_channel, 128,
                               kernel_size=3, stride=2, padding=1)
        self.gdn2 = GDN(128)
        self.conv3 = nn.Conv2d(128 + condition_channel, 192,
                               kernel_size=3, stride=1, padding=1)
        self.gdn3 = GDN(192)
        self.conv4 = nn.Conv2d(192, out_channel, kernel_size=3, stride=1,
                               padding=1)
        self.conv_fuse = nn.Conv2d(3 * condition_channel, condition_channel,
                                   kernel_size=3, stride=1, padding=1)

    def forward(self, x, high_feature):
        high_feature_p4_p5_p6 = self.conv_fuse(torch.cat(
            (high_feature['p4'], high_feature['p5'], high_feature['p6']),
            dim=1))
        x = self.gdn1(self.conv1(torch.cat((x, high_feature['p2']), dim=1)))
        x = self.gdn2(self.conv2(torch.cat((x, high_feature['p3']), dim=1)))
        x = self.gdn3(self.conv3(
            torch.cat((x, high_feature_p4_p5_p6), dim=1)))
        x = self.conv4(x)
        return x


class DepthwiseConv2d(nn.Module):
    """Depthwise separable convolution (depthwise + pointwise)."""
    def __init__(self, in_channel, out_channel, kernel_size):
        super().__init__()
        self.depth_wise = nn.Conv2d(in_channel, in_channel,
                                    kernel_size=kernel_size,
                                    padding=kernel_size // 2,
                                    groups=in_channel)
        self.point_wise = nn.Conv2d(in_channel, out_channel, kernel_size=1)

    def forward(self, x):
        x = self.depth_wise(x)
        x = self.point_wise(x)
        return x


class DepthResblock(nn.Module):
    """Residual block built from depthwise separable convolutions."""
    def __init__(self, channel, kernel_size):
        super().__init__()
        self.conv1 = DepthwiseConv2d(channel, channel, kernel_size)
        self.relu = nn.LeakyReLU(inplace=False)
        self.conv2 = DepthwiseConv2d(channel, channel, kernel_size)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)
        return x + y


class TinyResblock(nn.Module):
    """Lightweight residual block with two 3×3 convolutions."""
    def __init__(self, channel, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel_size, stride=1,
                               padding=kernel_size // 2)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size, stride=1,
                               padding=kernel_size // 2)

    def forward(self, x):
        return x + self.relu(self.conv2(self.relu(self.conv1(x))))


class FeatureExtract(nn.Module):
    """Channel reduction for mid-level features (256 → 64)."""
    def __init__(self, in_channel, out_channel):
        super().__init__()
        mc1 = (2 * in_channel + out_channel) // 3
        mc2 = (in_channel + 2 * out_channel) // 3
        self.conv1 = nn.Conv2d(in_channel, mc1, 3, stride=1, padding=1)
        self.res1 = TinyResblock(mc1, 3)
        self.conv2 = nn.Conv2d(mc1, mc2, 3, stride=1, padding=1)
        self.res2 = TinyResblock(mc2, 3)
        self.conv3 = nn.Conv2d(mc2, out_channel, 3, stride=1, padding=1)
        self.res3 = TinyResblock(out_channel, 3)

    def forward(self, x):
        y = self.res1(self.conv1(x))
        y = self.res2(self.conv2(y))
        y = self.res3(self.conv3(y))
        return y


class FeatureRecon(nn.Module):
    """Channel expansion + 2× spatial up-sampling (64 → 256)."""
    def __init__(self, in_channel, out_channel):
        super().__init__()
        mc1 = (2 * in_channel + out_channel) // 3
        mc2 = (in_channel + 2 * out_channel) // 3
        self.conv1 = subpel_conv3x3(in_channel, mc1)
        self.res1 = TinyResblock(mc1, 3)
        self.conv2 = subpel_conv3x3(mc1, mc2)
        self.res2 = TinyResblock(mc2, 3)
        self.conv3 = subpel_conv3x3(mc2, out_channel)
        self.res3 = TinyResblock(out_channel, 3)

    def forward(self, x):
        y = self.res1(self.conv1(x))
        y = self.res2(self.conv2(y))
        y = self.res3(self.conv3(y))
        return y


class TinyFeatureExtract(nn.Module):
    """Minimal 2-layer channel reduction for high-level features."""
    def __init__(self, in_channel, out_channel):
        super().__init__()
        mid = (in_channel + out_channel) // 2
        self.conv1 = nn.Conv2d(in_channel, mid, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(mid, out_channel, 3, stride=1, padding=1)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class ME(nn.Module):
    """Motion Estimation: U-Net style flow predictor for feature space."""
    def __init__(self, feature_channel, mv_channel):
        super().__init__()
        self.res1 = TinyResblock(feature_channel * 2, 3)
        self.conv2 = nn.Conv2d(feature_channel * 2, mv_channel, 7, stride=2,
                               padding=3)
        self.res2 = TinyResblock(mv_channel, 3)
        self.conv3 = nn.Conv2d(mv_channel, mv_channel, 7, stride=2, padding=3)
        self.res3 = TinyResblock(mv_channel, 3)
        self.deconv1 = nn.ConvTranspose2d(mv_channel, mv_channel, 7,
                                          padding=3, output_padding=1,
                                          stride=2)
        self.deconv2 = nn.ConvTranspose2d(mv_channel, mv_channel, 7,
                                          padding=3, output_padding=1,
                                          stride=2)
        self.trans1 = nn.Conv2d(feature_channel * 2, mv_channel, 1)
        self.trans2 = nn.Conv2d(mv_channel, mv_channel, 1)
        self.trans3 = nn.Conv2d(mv_channel, mv_channel, 1)

    def forward(self, input_feature, refer_feature):
        x = torch.cat((input_feature, refer_feature), dim=1)
        t1 = self.res1(x)
        t2 = self.res2(self.conv2(t1))
        t3 = self.res3(self.conv3(t2))
        t1, t2, t3 = self.trans1(t1), self.trans2(t2), self.trans3(t3)
        y = self.deconv1(t3) + t2
        y = self.deconv2(y) + t1
        return y


class MC(nn.Module):
    """Motion Compensation: applies motion vectors to warp reference features
    via learned pixel-wise modulation."""
    def __init__(self, mv_channel, feature_channel, upsample_factor):
        super().__init__()
        self.conv1 = nn.Conv2d(mv_channel, mv_channel, kernel_size=7,
                               stride=1, padding=3)
        self.res1 = TinyResblock(mv_channel, kernel_size=3)
        self.conv2 = DepthwiseConv2d(mv_channel,
                                     mv_channel * upsample_factor,
                                     kernel_size=5)
        self.res2 = DepthResblock(mv_channel * upsample_factor, kernel_size=3)
        self.conv3 = DepthwiseConv2d(mv_channel * upsample_factor,
                                     mv_channel * upsample_factor *
                                     upsample_factor, kernel_size=5)
        self.res3 = DepthResblock(mv_channel * upsample_factor *
                                  upsample_factor, kernel_size=3)
        self.pixelConv = nn.Conv2d(mv_channel,
                                   feature_channel * upsample_factor *
                                   upsample_factor, kernel_size=1)
        self.warpConv = nn.Conv2d(feature_channel * upsample_factor *
                                  upsample_factor, feature_channel,
                                  kernel_size=7, padding=3)

    def forward(self, refer_feature, mv):
        refer_feature = self.res1(self.conv1(refer_feature))
        refer_feature = self.res2(self.conv2(refer_feature))
        refer_feature = self.res3(self.conv3(refer_feature))
        mv = self.pixelConv(mv)
        warped_feature = self.warpConv(mv * refer_feature)
        return warped_feature


class MultiScaleContextualEncoder(nn.Module):
    """Multi-scale encoder for residual compression.

    Fuses multi-scale high-level features from the task head as conditioning
    at each spatial level.
    """
    def __init__(self, in_channel, out_channel, condition_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, stride=1,
                               padding=1)
        self.gdn1 = GDN(out_channel)
        self.res1 = ResBlock_LeakyReLU_0_Point_1(out_channel)

        self.conv2 = nn.Conv2d(out_channel + condition_channel, out_channel,
                               5, stride=2, padding=2)
        self.gdn2 = GDN(out_channel)
        self.res2 = ResBlock_LeakyReLU_0_Point_1(out_channel)

        self.conv3 = nn.Conv2d(out_channel + condition_channel, out_channel,
                               5, stride=2, padding=2)
        self.gdn3 = GDN(out_channel)
        self.res3 = ResBlock_LeakyReLU_0_Point_1(out_channel)

        self.conv_fuse_p4_to_p6 = nn.Sequential(
            nn.Conv2d(3 * condition_channel, 2 * condition_channel,
                      5, stride=1, padding=2),
            nn.Conv2d(2 * condition_channel, condition_channel,
                      5, stride=1, padding=2),
        )

        self.conv4 = nn.Conv2d(out_channel + condition_channel, out_channel,
                               5, stride=1, padding=2)
        self.gdn4 = GDN(out_channel)
        self.res4 = ResBlock_LeakyReLU_0_Point_1(out_channel)

    def forward(self, x, high_feature):
        p2 = high_feature['p2']
        p3 = high_feature['p3']
        p4_p6 = self.conv_fuse_p4_to_p6(torch.cat(
            (high_feature['p4'], high_feature['p5'], high_feature['p6']),
            dim=1))

        x = self.res1(self.gdn1(self.conv1(x)))
        x = self.res2(self.gdn2(self.conv2(torch.cat((x, p2), dim=1))))
        x = self.res3(self.gdn3(self.conv3(torch.cat((x, p3), dim=1))))
        x = self.res4(self.gdn4(self.conv4(torch.cat((x, p4_p6), dim=1))))
        return x


class MultiScaleContextualDecoder(nn.Module):
    """Multi-scale decoder for residual reconstruction.

    Mirrors ``MultiScaleContextualEncoder``: fuses warped feature and
    multi-scale high-level conditioning.
    """
    def __init__(self, in_channel, feature_channel, condition_channel):
        super().__init__()
        self.conv_fuse_p4_to_p6 = nn.Sequential(
            nn.Conv2d(3 * condition_channel, 2 * condition_channel,
                      5, stride=1, padding=2),
            nn.Conv2d(2 * condition_channel, condition_channel,
                      5, stride=1, padding=2),
        )

        self.deconv1 = nn.ConvTranspose2d(
            in_channel + condition_channel, in_channel, 5,
            padding=2, output_padding=1, stride=2)
        self.igdn1 = GDN(in_channel, inverse=True)
        self.res1 = ResBlock_LeakyReLU_0_Point_1(in_channel)

        self.deconv2 = nn.ConvTranspose2d(
            in_channel + condition_channel, in_channel, 5,
            padding=2, output_padding=1, stride=2)
        self.igdn2 = GDN(in_channel, inverse=True)
        self.res2 = ResBlock_LeakyReLU_0_Point_1(in_channel)

        self.deconv3 = nn.ConvTranspose2d(
            in_channel + condition_channel, in_channel, 5,
            padding=2, stride=1)
        self.igdn3 = GDN(in_channel, inverse=True)
        self.res3 = ResBlock_LeakyReLU_0_Point_1(in_channel)

        self.deconv4 = nn.ConvTranspose2d(
            in_channel + feature_channel, feature_channel, 3,
            padding=1, stride=1)
        self.igdn4 = GDN(feature_channel, inverse=True)
        self.res4 = ResBlock_LeakyReLU_0_Point_1(feature_channel)

    def forward(self, y, warped_feature, high_feature):
        p2 = high_feature['p2']
        p3 = high_feature['p3']
        p4_p6 = self.conv_fuse_p4_to_p6(torch.cat(
            (high_feature['p4'], high_feature['p5'], high_feature['p6']),
            dim=1))

        y = self.res1(self.igdn1(
            self.deconv1(torch.cat((y, p4_p6), dim=1))))
        y = self.res2(self.igdn2(
            self.deconv2(torch.cat((y, p3), dim=1))))
        y = self.res3(self.igdn3(
            self.deconv3(torch.cat((y, p2), dim=1))))
        y = self.res4(self.igdn4(
            self.deconv4(torch.cat((y, warped_feature), dim=1))))
        return y


class Encoder(nn.Module):
    """Motion-vector encoder: channel alignment + 2× down-sampling."""
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, stride=1,
                               padding=1)
        self.gdn1 = GDN(out_channel)
        self.res1 = ResBlock_LeakyReLU_0_Point_1(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 5, stride=2,
                               padding=2)
        self.gdn2 = GDN(out_channel)
        self.res2 = ResBlock_LeakyReLU_0_Point_1(out_channel)
        self.conv3 = nn.Conv2d(out_channel, out_channel, 5, stride=2,
                               padding=2)
        self.gdn3 = GDN(out_channel)
        self.res3 = ResBlock_LeakyReLU_0_Point_1(out_channel)

    def forward(self, x):
        x = self.res1(self.gdn1(self.conv1(x)))
        x = self.res2(self.gdn2(self.conv2(x)))
        x = self.res3(self.gdn3(self.conv3(x)))
        return x


class Decoder(nn.Module):
    """Motion-vector decoder: 2× up-sampling + channel expansion."""
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(in_channel, in_channel, 5,
                                          padding=2, output_padding=1,
                                          stride=2)
        self.igdn1 = GDN(in_channel, inverse=True)
        self.res1 = ResBlock_LeakyReLU_0_Point_1(in_channel)
        self.deconv2 = nn.ConvTranspose2d(in_channel, in_channel, 5,
                                          padding=2, output_padding=1,
                                          stride=2)
        self.igdn2 = GDN(in_channel, inverse=True)
        self.res2 = ResBlock_LeakyReLU_0_Point_1(in_channel)
        self.deconv3 = nn.ConvTranspose2d(in_channel, out_channel, 3,
                                          padding=1, stride=1)
        self.igdn3 = GDN(out_channel, inverse=True)
        self.res3 = ResBlock_LeakyReLU_0_Point_1(out_channel)

    def forward(self, y):
        y = self.res1(self.igdn1(self.deconv1(y)))
        y = self.res2(self.igdn2(self.deconv2(y)))
        y = self.res3(self.igdn2(self.deconv3(y)))
        return y


# +===========================================================================+
# |                           FeatureCodec                                    |
# +===========================================================================+

class FeatureCodec(nn.Module):
    """Learnable feature-domain video codec.

    Compresses intermediate features produced by a frozen vision backbone
    (the ``task_head``) using motion-compensated temporal prediction and
    multi-scale residual coding with a hyperprior entropy model.

    Args:
        task_head (nn.Module): a frozen module that maps the mid-level
            feature (e.g. FPN P3, 256 channels) to a dictionary of
            multi-scale high-level features ``{'p2',...,'p6'}``.
            This is typically the neck / head of a pre-trained detector.

    Inputs:
        raw_input_feature_mid  (B×256×H×W):  mid-feature of the current frame.
        raw_refer_feature_mid  (B×256×H×W):  mid-feature of the reference frame.

    Returns:
        raw_recon_feature_mid  (B×256×H×W):  reconstructed mid-feature.
        raw_warped_feature_mid (B×256×H×W):  motion-compensated mid-feature.
        mse_dict:          dict with 'mse_mid', 'mse_high', 'mse_warp',
                           'mse_p2' … 'mse_p6'.
        bpp_dict:          dict with 'bpp', 'bpp_y', 'bpp_z',
                           'bpp_mv_y', 'bpp_mv_z'.
    """
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

        self.featureExtract = FeatureExtract(channel_raw, channel_feature)
        self.featureRecon = FeatureRecon(channel_feature, channel_raw)

        self.inputHighFeatureExtract_p2 = TinyFeatureExtract(
            channel_raw, channel_condition)
        self.inputHighFeatureExtract_p3 = TinyFeatureExtract(
            channel_raw, channel_condition)
        self.inputHighFeatureExtract_p4 = TinyFeatureExtract(
            channel_raw, channel_condition)
        self.inputHighFeatureExtract_p5 = nn.Sequential(
            TinyFeatureExtract(channel_raw, channel_condition),
            nn.ConvTranspose2d(channel_condition, channel_condition,
                               kernel_size=3, stride=2, padding=1,
                               output_padding=1))
        self.inputHighFeatureExtract_p6 = nn.Sequential(
            TinyFeatureExtract(channel_raw, channel_condition),
            nn.ConvTranspose2d(channel_condition, channel_condition,
                               kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.ConvTranspose2d(channel_condition, channel_condition,
                               kernel_size=3, stride=2, padding=1,
                               output_padding=1))

        self.warpedHighFeatureExtract_p2 = TinyFeatureExtract(
            channel_raw, channel_condition)
        self.warpedHighFeatureExtract_p3 = TinyFeatureExtract(
            channel_raw, channel_condition)
        self.warpedHighFeatureExtract_p4 = TinyFeatureExtract(
            channel_raw, channel_condition)
        self.warpedHighFeatureExtract_p5 = nn.Sequential(
            TinyFeatureExtract(channel_raw, channel_condition),
            nn.ConvTranspose2d(channel_condition, channel_condition,
                               kernel_size=3, stride=2, padding=1,
                               output_padding=1))
        self.warpedHighFeatureExtract_p6 = nn.Sequential(
            TinyFeatureExtract(channel_raw, channel_condition),
            nn.ConvTranspose2d(channel_condition, channel_condition,
                               kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.ConvTranspose2d(channel_condition, channel_condition,
                               kernel_size=3, stride=2, padding=1,
                               output_padding=1))

        self.task_head = task_head

        self.bitEstimator_z = BitEstimator(channel_z)
        self.bitEstimator_z_mv = BitEstimator(channel_mv_z)
        self.gaussian_encoder = GaussianEncoder()

        self.contextualEncoder = MultiScaleContextualEncoder(
            channel_feature * 2, channel_y, channel_condition)
        self.contextualDecoder = MultiScaleContextualDecoder(
            channel_y, channel_feature, condition_channel=channel_condition)

        self.motionEstimation = ME(channel_feature, channel_mv)
        self.motinoCompensation = MC(mv_channel=channel_mv,
                                     feature_channel=channel_feature,
                                     upsample_factor=4)

        self.mvEncoder = Encoder(channel_mv, channel_mv_y)
        self.mvDecoder = Decoder(channel_mv_y, channel_mv)

        self.priorEncoder = PriorEncoderNet(channel_y, channel_z)
        self.priorDecoder = PriorDecoderNet(channel_z, 2 * channel_y)

        self.mvPriorEncoder = PriorEncoderNet(channel_mv_y, channel_mv_z)
        self.mvPriorDecoder = PriorDecoderNet(channel_mv_z, 2 * channel_mv_y)

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(channel_y * 12 // 3, channel_y * 9 // 3, kernel_size=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channel_y * 9 // 3, channel_y * 6 // 3, kernel_size=1),
        )

        self.temporalPriorEncoder = TemporalPriorEncoderNet(
            feature_channel=channel_feature,
            condition_channel=channel_condition,
            out_channel=2 * channel_y)

        self.mxrange = 150
        self.calrealbits = False

    # ------------------------------------------------------------------
    #  Entropy helpers
    # ------------------------------------------------------------------

    def feature_probs_based_sigma(self, feature, mean, sigma):
        outputs = feature
        values = outputs - mean
        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.laplace.Laplace(mu, sigma)
        probs = gaussian.cdf(values + 0.5) - gaussian.cdf(values - 0.5)
        total_bits = torch.sum(torch.clamp(
            -1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50))
        return total_bits, probs

    def iclr18_estrate_bits_z(self, z):
        prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
        total_bits = torch.sum(torch.clamp(
            -1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))
        return total_bits, prob

    def iclr18_estrate_bits_z_mv(self, z_mv):
        prob = self.bitEstimator_z_mv(z_mv + 0.5) - \
            self.bitEstimator_z_mv(z_mv - 0.5)
        total_bits = torch.sum(torch.clamp(
            -1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))
        return total_bits, prob

    def update(self, force=False):
        """Recompute entropy tables (call before first encode/decode)."""
        self.bitEstimator_z_mv.update(force=force)
        self.bitEstimator_z.update(force=force)
        self.gaussian_encoder.update(force=force)

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------

    def forward(self, raw_input_feature_mid, raw_refer_feature_mid):
        # ---- Space Transform (channel reduction) ----
        input_feature_mid = self.featureExtract(raw_input_feature_mid)
        refer_feature_mid = self.featureExtract(raw_refer_feature_mid)

        # ---- Motion Branch ----
        mv = self.motionEstimation(input_feature=input_feature_mid,
                                   refer_feature=refer_feature_mid)
        mv_y = self.mvEncoder(mv)
        if self.training:
            compressed_mv_y = mv_y + (torch.round(mv_y) - mv_y).detach()
        else:
            compressed_mv_y = torch.round(mv_y)
        mv_hat = self.mvDecoder(compressed_mv_y)
        warped_feature = self.motinoCompensation(
            refer_feature=refer_feature_mid, mv=mv_hat)
        raw_warped_feature_mid = self.featureRecon(warped_feature)

        # ---- Multi-scale conditioning from task head ----
        raw_input_feature_high = self.task_head(raw_input_feature_mid)
        input_feature_high = {
            'p2': self.inputHighFeatureExtract_p2(
                raw_input_feature_high['p2']),
            'p3': self.inputHighFeatureExtract_p3(
                raw_input_feature_high['p3']),
            'p4': self.inputHighFeatureExtract_p4(
                raw_input_feature_high['p4']),
            'p5': self.inputHighFeatureExtract_p5(
                raw_input_feature_high['p5']),
            'p6': self.inputHighFeatureExtract_p6(
                raw_input_feature_high['p6']),
        }

        raw_warped_feature_high = self.task_head(raw_warped_feature_mid)
        warped_feature_high = {
            'p2': self.warpedHighFeatureExtract_p2(
                raw_warped_feature_high['p2']),
            'p3': self.warpedHighFeatureExtract_p3(
                raw_warped_feature_high['p3']),
            'p4': self.warpedHighFeatureExtract_p4(
                raw_warped_feature_high['p4']),
            'p5': self.warpedHighFeatureExtract_p5(
                raw_warped_feature_high['p5']),
            'p6': self.warpedHighFeatureExtract_p6(
                raw_warped_feature_high['p6']),
        }

        # ---- Contextual (residual) Branch ----
        y = self.contextualEncoder(
            x=torch.cat((input_feature_mid, warped_feature), dim=1),
            high_feature=input_feature_high)
        if self.training:
            compressed_y = y + (torch.round(y) - y).detach()
        else:
            compressed_y = torch.round(y)
        input_feature_hat = self.contextualDecoder(
            y=compressed_y, warped_feature=warped_feature,
            high_feature=warped_feature_high)
        raw_recon_feature_mid = self.featureRecon(input_feature_hat)

        # ---- Entropy model ----
        # Motion branch:
        mv_z = self.mvPriorEncoder(mv_y)
        if self.training:
            compressed_mv_z = mv_z + (torch.round(mv_z) - mv_z).detach()
        else:
            compressed_mv_z = torch.round(mv_z)
        mv_means_hat, mv_scales_hat = \
            self.mvPriorDecoder(compressed_mv_z).chunk(2, 1)

        # Contextual branch: temporal + hyper-prior
        temporal_prior_params = self.temporalPriorEncoder(
            warped_feature, high_feature=warped_feature_high)
        z = self.priorEncoder(y)
        if self.training:
            compressed_z = z + (torch.round(z) - z).detach()
        else:
            compressed_z = torch.round(z)
        params = self.priorDecoder(compressed_z)
        gaussian_params = self.entropy_parameters(
            torch.cat((temporal_prior_params, params), dim=1))
        contextual_means_hat, contextual_scales_hat = \
            gaussian_params.chunk(2, 1)

        # ---- Loss evaluation ----
        total_bits_mv_y, _ = self.feature_probs_based_sigma(
            compressed_mv_y, mv_means_hat, mv_scales_hat)
        total_bits_mv_z, _ = self.iclr18_estrate_bits_z_mv(compressed_mv_z)
        total_bits_y, _ = self.feature_probs_based_sigma(
            compressed_y, contextual_means_hat, contextual_scales_hat)
        total_bits_z, _ = self.iclr18_estrate_bits_z(compressed_z)

        pixel_num = (input_feature_mid.shape[0] *
                     (input_feature_mid.shape[2] * 4) *
                     (input_feature_mid.shape[3] * 4))
        bpp_y = total_bits_y / pixel_num
        bpp_z = total_bits_z / pixel_num
        bpp_mv_y = total_bits_mv_y / pixel_num
        bpp_mv_z = total_bits_mv_z / pixel_num
        bpp = bpp_y + bpp_z + bpp_mv_y + bpp_mv_z
        bpp_dict = {'bpp': bpp, 'bpp_y': bpp_y, 'bpp_z': bpp_z,
                    'bpp_mv_y': bpp_mv_y, 'bpp_mv_z': bpp_mv_z}

        mse_mid = torch.mean(
            (raw_recon_feature_mid - raw_input_feature_mid).pow(2))
        mse_warp = torch.mean(
            (raw_warped_feature_mid - raw_input_feature_mid).pow(2))
        raw_recon_feature_high = self.task_head(raw_recon_feature_mid)
        mse_p2 = torch.mean(
            (raw_recon_feature_high['p2'] -
             raw_input_feature_high['p2']).pow(2))
        mse_p3 = torch.mean(
            (raw_recon_feature_high['p3'] -
             raw_input_feature_high['p3']).pow(2))
        mse_p4 = torch.mean(
            (raw_recon_feature_high['p4'] -
             raw_input_feature_high['p4']).pow(2))
        mse_p5 = torch.mean(
            (raw_recon_feature_high['p5'] -
             raw_input_feature_high['p5']).pow(2))
        mse_p6 = torch.mean(
            (raw_recon_feature_high['p6'] -
             raw_input_feature_high['p6']).pow(2))
        mse_high = (mse_p2 + mse_p3 + mse_p4 + mse_p5 + mse_p6) / 5
        mse_dict = {'mse_mid': mse_mid, 'mse_warp': mse_warp,
                    'mse_high': mse_high,
                    'mse_p2': mse_p2, 'mse_p3': mse_p3,
                    'mse_p4': mse_p4, 'mse_p5': mse_p5, 'mse_p6': mse_p6}

        return raw_recon_feature_mid, raw_warped_feature_mid, \
            mse_dict, bpp_dict
