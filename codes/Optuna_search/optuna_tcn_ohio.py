import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from optuna.samplers import GridSampler, TPESampler
from optuna.trial import TrialState

INPUTS = ["Glucose", "MealSize", "Insulin"]
TARGET = "Glucose"
PATIENT_COLUMN = "PID"
TIME_COLUMN = "Time"
SAMPLE_MINUTES = 1.0
INPUT_LEN = 144
FORECAST_MINUTES = 12
OUTPUT_LEN = int(round(FORECAST_MINUTES / SAMPLE_MINUTES))
TRAIN_FRAC = 0.8
TEST_STRIDE = 6

PRIOR_SIGMA0 = 0.17
SIGMA_NOISE0 = 10.3
EPOCHS, BATCH, LR, TRAIN_STRIDE = 100, 256, 3e-3, 4
RETRAIN = True
RETRAIN_EPOCHS = 100
SEED = 0

INCLUDE_LOGDET = True
HYPERPRIOR = "log-normal"
HYPERPRIOR_M, HYPERPRIOR_S = 0.0, 1.0
EVID_STRIDE = 1
EVID_BATCH = 4096

POOL_KERNEL = 3
UPSAMPLE_KERNEL = 3
FINAL_KERNEL = 5
MAX_DILATION_BASE = 5

CATEGORY_BOUNDS = {"C1": (0, 2_500), "C2": (2_500, 5_000), "C3": (5_000, 15_000),
                   "C4": (15_000, 40_000), "C5": (40_000, 65_000), "C6": (65_000, 100_000),
                   "C7": (100_000, 200_000), "C8": (200_000, 300_000)}

ARCH_POOL = {
    "C1": [
        ("C1-A1", [2, 4], 3, 299),
        ("C1-A2", [2, 4], 5, 455),
        ("C1-A3", [2, 4], 7, 611),
        ("C1-A4", [2, 4], 9, 767),
        ("C1-A5", [2, 4], 11, 923),
        ("C1-A6", [4, 8], 3, 1061),
        ("C1-A7", [2, 4], 13, 1079),
        ("C1-A8", [2, 4, 8], 3, 1127),
        ("C1-A9", [2, 4], 15, 1235),
        ("C1-A10", [4, 8], 5, 1661),
        ("C1-A11", [2, 4, 8], 5, 1763),
        ("C1-A12", [4, 8], 7, 2261),
        ("C1-A13", [2, 4, 8], 7, 2399),
    ],
    "C2": [
        ("C2-A1", [4, 8], 9, 2861),
        ("C2-A2", [2, 4, 8], 9, 3035),
        ("C2-A3", [4, 8], 11, 3461),
        ("C2-A4", [2, 4, 8], 11, 3671),
        ("C2-A5", [8, 16], 3, 3977),
        ("C2-A6", [4, 8], 13, 4061),
        ("C2-A7", [4, 8, 16], 3, 4285),
        ("C2-A8", [2, 4, 8], 13, 4307),
        ("C2-A9", [2, 4, 8, 16], 3, 4351),
        ("C2-A10", [4, 8], 15, 4661),
        ("C2-A11", [2, 4, 8], 15, 4943),
    ],
    "C3": [
        ("C3-A1", [8, 16], 5, 6329),
        ("C3-A2", [4, 8, 16], 5, 6805),
        ("C3-A3", [2, 4, 8, 16], 5, 6907),
        ("C3-A4", [8, 16], 7, 8681),
        ("C3-A5", [4, 8, 16], 7, 9325),
        ("C3-A6", [2, 4, 8, 16], 7, 9463),
        ("C3-A7", [8, 16], 9, 11033),
        ("C3-A8", [4, 8, 16], 9, 11845),
        ("C3-A9", [2, 4, 8, 16], 9, 12019),
        ("C3-A10", [8, 16], 11, 13385),
        ("C3-A11", [4, 8, 16], 11, 14365),
        ("C3-A12", [2, 4, 8, 16], 11, 14575),
    ],
    "C4": [
        ("C4-A1", [16, 32], 3, 15377),
        ("C4-A2", [8, 16], 13, 15737),
        ("C4-A3", [8, 16, 32], 3, 16697),
        ("C4-A4", [4, 8, 16], 13, 16885),
        ("C4-A5", [4, 8, 16, 32], 3, 17005),
        ("C4-A6", [2, 4, 8, 16, 32], 3, 17071),
        ("C4-A7", [2, 4, 8, 16], 13, 17131),
        ("C4-A8", [8, 16], 15, 18089),
        ("C4-A9", [4, 8, 16], 15, 19405),
        ("C4-A10", [2, 4, 8, 16], 15, 19687),
        ("C4-A11", [16, 32], 5, 24689),
        ("C4-A12", [8, 16, 32], 5, 26729),
        ("C4-A13", [4, 8, 16, 32], 5, 27205),
        ("C4-A14", [2, 4, 8, 16, 32], 5, 27307),
        ("C4-A15", [16, 32], 7, 34001),
        ("C4-A16", [8, 16, 32], 7, 36761),
        ("C4-A17", [4, 8, 16, 32], 7, 37405),
        ("C4-A18", [2, 4, 8, 16, 32], 7, 37543),
    ],
    "C5": [
        ("C5-A1", [16, 32], 9, 43313),
        ("C5-A2", [8, 16, 32], 9, 46793),
        ("C5-A3", [4, 8, 16, 32], 9, 47605),
        ("C5-A4", [2, 4, 8, 16, 32], 9, 47779),
        ("C5-A5", [16, 32], 11, 52625),
        ("C5-A6", [8, 16, 32], 11, 56825),
        ("C5-A7", [4, 8, 16, 32], 11, 57805),
        ("C5-A8", [2, 4, 8, 16, 32], 11, 58015),
        ("C5-A9", [32, 64], 3, 60449),
        ("C5-A10", [16, 32], 13, 61937),
    ],
    "C6": [
        ("C6-A1", [16, 32, 64], 3, 65905),
        ("C6-A2", [8, 16, 32], 13, 66857),
        ("C6-A3", [8, 16, 32, 64], 3, 67225),
        ("C6-A4", [4, 8, 16, 32, 64], 3, 67533),
        ("C6-A5", [2, 4, 8, 16, 32, 64], 3, 67599),
        ("C6-A6", [4, 8, 16, 32], 13, 68005),
        ("C6-A7", [2, 4, 8, 16, 32], 13, 68251),
        ("C6-A8", [16, 32], 15, 71249),
        ("C6-A9", [8, 16, 32], 15, 76889),
        ("C6-A10", [4, 8, 16, 32], 15, 78205),
        ("C6-A11", [2, 4, 8, 16, 32], 15, 78487),
        ("C6-A12", [32, 64], 5, 97505),
    ],
    "C7": [
        ("C7-A1", [16, 32, 64], 5, 105937),
        ("C7-A2", [8, 16, 32, 64], 5, 107977),
        ("C7-A3", [4, 8, 16, 32, 64], 5, 108453),
        ("C7-A4", [2, 4, 8, 16, 32, 64], 5, 108555),
        ("C7-A5", [32, 64], 7, 134561),
        ("C7-A6", [16, 32, 64], 7, 145969),
        ("C7-A7", [8, 16, 32, 64], 7, 148729),
        ("C7-A8", [4, 8, 16, 32, 64], 7, 149373),
        ("C7-A9", [2, 4, 8, 16, 32, 64], 7, 149511),
        ("C7-A10", [32, 64], 9, 171617),
        ("C7-A11", [16, 32, 64], 9, 186001),
        ("C7-A12", [8, 16, 32, 64], 9, 189481),
        ("C7-A13", [4, 8, 16, 32, 64], 9, 190293),
        ("C7-A14", [2, 4, 8, 16, 32, 64], 9, 190467),
    ],
    "C8": [
        ("C8-A1", [32, 64], 11, 208673),
        ("C8-A2", [16, 32, 64], 11, 226033),
        ("C8-A3", [8, 16, 32, 64], 11, 230233),
        ("C8-A4", [4, 8, 16, 32, 64], 11, 231213),
        ("C8-A5", [2, 4, 8, 16, 32, 64], 11, 231423),
        ("C8-A6", [64, 128], 3, 239681),
        ("C8-A7", [32, 64], 13, 245729),
        ("C8-A8", [32, 64, 128], 3, 261857),
        ("C8-A9", [16, 32, 64], 13, 266065),
        ("C8-A10", [16, 32, 64, 128], 3, 267313),
        ("C8-A11", [8, 16, 32, 64, 128], 3, 268633),
        ("C8-A12", [4, 8, 16, 32, 64, 128], 3, 268941),
        ("C8-A13", [2, 4, 8, 16, 32, 64, 128], 3, 269007),
        ("C8-A14", [8, 16, 32, 64], 13, 270985),
        ("C8-A15", [4, 8, 16, 32, 64], 13, 272133),
        ("C8-A16", [2, 4, 8, 16, 32, 64], 13, 272379),
        ("C8-A17", [32, 64], 15, 282785),
    ],
}

RESULT_COLUMNS = [
    "rank", "selected", "arch", "category", "structure_id", "n_blocks",
    "encoder_channels", "decoder_channels", "kernel_size", "dilation_base",
    "n_weights", "n_bayesian_weights", "n_deterministic_weights", "n_trainable_parameters",
    "log_evidence", "plausibility", "log10_plausibility", "delta_log_evidence",
    "log_evidence_no_logdet", "plausibility_no_logdet", "log10_plausibility_no_logdet",
    "log_evidence_given_sigma", "log_likelihood", "log_prior", "occam_logdet",
    "log_hyperprior", "log_term", "sigma_noise_sq_map", "sigma_pr_sq_map", "tr_H_inv",
    "sse", "n_obs", "train_rmse", "test_rmse", "seconds", "trial",
]


class BBBConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True, priors=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.use_bias = bias

        self.W_mu = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size))
        self.W_rho = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size))
        if bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_channels))
            self.bias_rho = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias_mu", None)
            self.register_parameter("bias_rho", None)

        ps = priors or {}
        prior_mu = float(ps.get("prior_mu", 0.0))
        prior_sigma = float(ps.get("prior_sigma", 0.17))
        self.register_buffer("W_prior_mu", torch.zeros_like(self.W_mu) + prior_mu)
        self.register_buffer("W_prior_sigma", torch.full_like(self.W_mu, prior_sigma))
        if bias:
            self.register_buffer("b_prior_mu", torch.zeros_like(self.bias_mu) + prior_mu)
            self.register_buffer("b_prior_sigma", torch.full_like(self.bias_mu, prior_sigma))

        self.post_mu_init = ps.get("posterior_mu_initial", (0.0, 0.1))
        self.post_rho_init = ps.get("posterior_rho_initial", (-3.0, 0.1))
        self.reset_parameters()

    def reset_parameters(self):
        mu_m, mu_s = self.post_mu_init
        rho_m, rho_s = self.post_rho_init
        with torch.no_grad():
            self.W_mu.normal_(mu_m, mu_s)
            self.W_rho.normal_(rho_m, rho_s)
            if self.use_bias:
                self.bias_mu.normal_(mu_m, mu_s)
                self.bias_rho.normal_(rho_m, rho_s)

    def _sample_wb(self, sample=True):
        if sample:
            W = self.W_mu + F.softplus(self.W_rho) * torch.randn_like(self.W_mu)
            b = None
            if self.use_bias:
                b = self.bias_mu + F.softplus(self.bias_rho) * torch.randn_like(self.bias_mu)
        else:
            W = self.W_mu
            b = self.bias_mu if self.use_bias else None
        return W, b

    def forward(self, x, sample=True):
        W, b = self._sample_wb(sample=sample)
        return F.conv1d(x, W, b, stride=self.stride, padding=self.padding, dilation=self.dilation)

    @staticmethod
    def _kl_div(mu_q, sigma_q, mu_p, sigma_p):
        return 0.5 * (2 * torch.log(sigma_p / sigma_q) - 1
                      + (sigma_q / sigma_p).pow(2)
                      + ((mu_p - mu_q) / sigma_p).pow(2)).sum()

    def kl_loss(self):
        kl = self._kl_div(self.W_mu, F.softplus(self.W_rho), self.W_prior_mu, self.W_prior_sigma)
        if self.use_bias:
            kl += self._kl_div(self.bias_mu, F.softplus(self.bias_rho), self.b_prior_mu, self.b_prior_sigma)
        return kl


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x if self.chomp_size == 0 else x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, priors=None):
        super().__init__()
        self.conv1 = BBBConv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation, priors=priors)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.conv2 = BBBConv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation, priors=priors)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x, sample=True):
        out = self.relu1(self.chomp1(self.conv1(x, sample=sample)))
        out = self.relu2(self.chomp2(self.conv2(out, sample=sample)))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

    def kl_loss(self):
        return self.conv1.kl_loss() + self.conv2.kl_loss()


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, pool_kernel=POOL_KERNEL,
                 priors=None, dilation_base=4):
        super().__init__()
        layers = []
        self.pools = nn.ModuleList()
        self.num_levels = len(num_channels)
        for i in range(self.num_levels):
            dilation_size = dilation_base ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            layers.append(TemporalBlock(in_channels, num_channels[i], kernel_size, stride=1,
                                        dilation=dilation_size, padding=(kernel_size - 1) * dilation_size,
                                        priors=priors))
            if i < self.num_levels - 1:
                self.pools.append(nn.AvgPool1d(kernel_size=pool_kernel, stride=pool_kernel))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, sample=True):
        for i, layer in enumerate(self.layers):
            x = layer(x, sample=sample)
            if i < self.num_levels - 1:
                x = self.pools[i](x)
        return x


class TCNEncoder(nn.Module):
    def __init__(self, num_channels, input_channels=3, kernel_size=3, priors=None, dilation_base=4):
        super().__init__()
        self.network = TemporalConvNet(input_channels, num_channels, kernel_size, priors=priors,
                                       dilation_base=dilation_base)

    def forward(self, x, sample=True):
        return self.network(x.permute(0, 2, 1), sample=sample)


class CausalBayesianDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, priors=None):
        super().__init__()
        pad = kernel_size - 1
        self.conv1 = BBBConv1d(in_channels, out_channels, kernel_size, padding=pad, priors=priors)
        self.chomp1 = Chomp1d(pad)
        self.relu1 = nn.ReLU()
        self.conv2 = BBBConv1d(out_channels, out_channels, kernel_size, padding=pad, priors=priors)
        self.chomp2 = Chomp1d(pad)
        self.relu2 = nn.ReLU()
        self.downsample = (BBBConv1d(in_channels, out_channels, 1, priors=priors)
                           if in_channels != out_channels else None)
        self.relu = nn.ReLU()

    def forward(self, x, sample=True):
        out = self.relu1(self.chomp1(self.conv1(x, sample=sample)))
        out = self.relu2(self.chomp2(self.conv2(out, sample=sample)))
        res = x if self.downsample is None else self.downsample(x, sample=sample)
        return self.relu(out + res)

    def kl_loss(self):
        kl = self.conv1.kl_loss() + self.conv2.kl_loss()
        if self.downsample is not None:
            kl += self.downsample.kl_loss()
        return kl


class TCNDecoder(nn.Module):
    def __init__(self, num_channels, input_channels, output_size, output_seq_len,
                 kernel_size=3, upsample_kernel=UPSAMPLE_KERNEL, priors=None):
        super().__init__()
        self.layers = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.output_seq_len = output_seq_len
        in_ch = input_channels
        for i, ch in enumerate(num_channels):
            if i > 0:
                self.upsamples.append(nn.Upsample(scale_factor=upsample_kernel, mode="linear",
                                                  align_corners=False))
            self.layers.append(CausalBayesianDecoderBlock(in_ch, ch, kernel_size, priors=priors))
            in_ch = ch
        self.final = BBBConv1d(in_ch, output_size, kernel_size=FINAL_KERNEL, padding=FINAL_KERNEL - 1,
                               priors=priors)
        self.final_chomp = Chomp1d(FINAL_KERNEL - 1)

    def forward(self, x, sample=True):
        for i, layer in enumerate(self.layers):
            if i > 0:
                x = self.upsamples[i - 1](x)
            x = layer(x, sample=sample)
        x = self.final_chomp(self.final(x, sample=sample)).transpose(1, 2)
        return x[:, -self.output_seq_len:, :]

    def kl_loss(self):
        kl = 0
        for layer in self.layers:
            kl += layer.kl_loss()
        return kl + self.final.kl_loss()


class TCNModel(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, output_size, output_seq_length,
                 input_channels=3, kernel_size=3, priors=None, dilation_base=4):
        super().__init__()
        self.encoder = TCNEncoder(encoder_channels, input_channels=input_channels, kernel_size=kernel_size,
                                  priors=priors, dilation_base=dilation_base)
        self.decoder = TCNDecoder(decoder_channels, encoder_channels[-1], output_size, output_seq_length,
                                  kernel_size=kernel_size, priors=priors)

    def forward(self, x, sample=True):
        return self.decoder(self.encoder(x, sample=sample), sample=sample)

    def kl_loss(self):
        kl = 0.0
        for block in self.encoder.network.layers:
            kl += block.kl_loss()
        return kl + self.decoder.kl_loss()


class BayesTCN(TCNModel):
    def __init__(self, c_in, horizon, arch, input_len, prior_sigma=0.17):
        validate_architecture(arch, input_len, horizon)
        super().__init__(
            encoder_channels=arch["encoder_channels"],
            decoder_channels=arch["decoder_channels"],
            output_size=1,
            output_seq_length=horizon,
            input_channels=c_in,
            kernel_size=arch["kernel_size"],
            dilation_base=arch["dilation_base"],
            priors={"prior_mu": 0.0, "prior_sigma": float(prior_sigma),
                    "posterior_mu_initial": (0.0, 0.1), "posterior_rho_initial": (-3.0, 0.1)},
        )

    def forward(self, x, sample=True):
        return super().forward(x, sample=sample).squeeze(-1)


def validate_architecture(arch, input_len, horizon):
    enc, dec = arch["encoder_channels"], arch["decoder_channels"]
    k, base = arch["kernel_size"], arch["dilation_base"]
    if not enc or dec != enc[::-1]:
        raise ValueError("Decoder channels must mirror a nonempty encoder.")
    if any(not isinstance(c, int) or c < 1 or c > 512 for c in enc):
        raise ValueError("Channel widths must be integers in [1, 512].")
    if any(right != 2 * left for left, right in zip(enc, enc[1:])):
        raise ValueError("Encoder channels must double between successive blocks.")
    if not isinstance(k, int) or not 1 <= k <= 15:
        raise ValueError("kernel_size must be an integer in [1, 15].")
    if not isinstance(base, int) or not 1 <= base <= k:
        raise ValueError("dilation_base must be a positive integer <= kernel_size.")
    factor = POOL_KERNEL ** (len(enc) - 1)
    decoded_len = (input_len // factor) * factor
    if input_len < factor or horizon < 1 or decoded_len < horizon:
        raise ValueError(f"{arch['id']}: input length {input_len} produces decoder length "
                         f"{decoded_len}; need at least {horizon} output samples.")


def parameter_counts(model):
    bayesian = sum(m.W_mu.numel() + (m.bias_mu.numel() if m.use_bias else 0)
                   for m in model.modules() if isinstance(m, BBBConv1d))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    deterministic = trainable - 2 * bayesian
    return dict(n_weights=bayesian + deterministic, n_bayesian_weights=bayesian,
                n_deterministic_weights=deterministic, n_trainable_parameters=trainable)


def posterior(model):
    mus, sigs = [], []
    for m in model.modules():
        if isinstance(m, BBBConv1d):
            mus.append(m.W_mu.detach().reshape(-1))
            sigs.append(F.softplus(m.W_rho.detach()).reshape(-1))
            if m.use_bias:
                mus.append(m.bias_mu.detach().reshape(-1))
                sigs.append(F.softplus(m.bias_rho.detach()).reshape(-1))
    return torch.cat(mus).double(), torch.cat(sigs).double()


def category_architectures(category):
    return [{"id": f"{sid}-b{b}", "structure_id": sid, "encoder_channels": list(enc),
             "decoder_channels": list(enc)[::-1], "kernel_size": k, "dilation_base": b,
             "table_weights": n_table}
            for sid, enc, k, n_table in ARCH_POOL[category]
            for b in range(2, min(k, MAX_DILATION_BASE) + 1)]


def arch_name(arch):
    enc = "-".join(map(str, arch["encoder_channels"]))
    dec = "-".join(map(str, arch["decoder_channels"]))
    return f"{arch['id']}__enc-{enc}__dec-{dec}__k{arch['kernel_size']}__d{arch['dilation_base']}"


class WindowData:
    def __init__(self, csv_path):
        if SAMPLE_MINUTES <= 0 or FORECAST_MINUTES <= 0:
            raise ValueError("Sampling interval and forecast minutes must be positive.")
        if not np.isclose(OUTPUT_LEN * SAMPLE_MINUTES, FORECAST_MINUTES):
            raise ValueError("FORECAST_MINUTES must be an integer number of sampling intervals.")
        if not 0 < TRAIN_FRAC < 1:
            raise ValueError("TRAIN_FRAC must lie strictly between 0 and 1.")
        if min(INPUT_LEN, OUTPUT_LEN, TRAIN_STRIDE, TEST_STRIDE, EVID_STRIDE) < 1:
            raise ValueError("Window lengths and strides must be positive.")

        self.path = resolve_path(csv_path)
        raw = pd.read_csv(self.path)
        required = list(dict.fromkeys(INPUTS + [TARGET, PATIENT_COLUMN, TIME_COLUMN]))
        missing = [col for col in required if col not in raw.columns]
        if missing:
            raise ValueError(f"Missing required CSV columns: {missing}")
        if raw[PATIENT_COLUMN].isna().any():
            raise ValueError(f"{PATIENT_COLUMN} contains missing patient identifiers.")
        for col in dict.fromkeys(INPUTS + [TARGET, TIME_COLUMN]):
            raw[col] = pd.to_numeric(raw[col], errors="raise")
            if not np.isfinite(raw[col].to_numpy()).all():
                raise ValueError(f"{col} contains missing/nonfinite data; no interpolation is applied.")

        parts = [part.sort_values(TIME_COLUMN, kind="stable")
                 for _, part in raw.groupby(PATIENT_COLUMN, sort=False)]
        if not parts:
            raise ValueError("The CSV contains no patient data.")
        df = pd.concat(parts, ignore_index=True)
        n = len(df)

        training_row_mask = np.zeros(n, dtype=bool)
        train_chunks, test_chunks, self.split_summary = [], [], []
        offset = 0
        for pid, part in df.groupby(PATIENT_COLUMN, sort=False):
            length = len(part)
            split = offset + int(TRAIN_FRAC * length)
            training_row_mask[offset:split] = True
            times = part[TIME_COLUMN].to_numpy(dtype=float)
            cuts = np.r_[0, np.flatnonzero(~np.isclose(np.diff(times), SAMPLE_MINUTES, rtol=0, atol=1e-7)) + 1,
                         length]
            patient_train, patient_test = 0, 0
            for left, right in zip(cuts[:-1], cuts[1:]):
                starts = np.arange(offset + left, offset + right - INPUT_LEN - OUTPUT_LEN + 1, dtype=np.int64)
                tr = starts[starts + INPUT_LEN + OUTPUT_LEN <= split]
                te = starts[starts + INPUT_LEN >= split][::TEST_STRIDE]
                train_chunks.append(tr)
                test_chunks.append(te)
                patient_train += len(tr)
                patient_test += len(te)
            self.split_summary.append(dict(patient=str(pid), rows=length, train_rows=split - offset,
                                           test_rows=offset + length - split,
                                           train_windows=patient_train, test_windows=patient_test))
            offset += length

        self.train_starts = np.concatenate(train_chunks)
        self.test_starts = np.concatenate(test_chunks)
        if not len(self.train_starts) or not len(self.test_starts):
            raise ValueError("Insufficient contiguous data for training/test windows at these lengths.")
        self.train_start_mask = np.zeros(n, dtype=bool)
        self.train_start_mask[self.train_starts] = True
        self.evid_starts = self.train_starts[::EVID_STRIDE]
        self.n_train = len(self.train_starts)

        feats = df[INPUTS].to_numpy(np.float32)
        target = df[TARGET].to_numpy(np.float32)
        self.mu_x = feats[training_row_mask].mean(axis=0)
        sd_x = feats[training_row_mask].std(axis=0)
        self.sd_x = np.where(sd_x > 1e-8, sd_x, 1.0).astype(np.float32)
        self.feats_n = torch.tensor((feats - self.mu_x) / self.sd_x)
        self.target_t = torch.tensor(target)
        self.ar_in, self.ar_out = np.arange(INPUT_LEN), np.arange(OUTPUT_LEN)

    def windows(self, starts):
        starts = np.asarray(starts)
        X = self.feats_n[starts[:, None] + self.ar_in]
        Y = self.target_t[starts[:, None] + INPUT_LEN + self.ar_out]
        return X, Y

    def describe(self):
        print(f"Data: {self.path}")
        print(f"Sampling assumption: {SAMPLE_MINUTES:g} minute(s) per sample")
        print(f"X: previous {INPUT_LEN} samples ({INPUT_LEN * SAMPLE_MINUTES:g} min) of {INPUTS} "
              f"-> Y: next {OUTPUT_LEN} Glucose values ({FORECAST_MINUTES:g} min)")
        print(pd.DataFrame(self.split_summary).to_string(index=False))
        print(f"train windows {self.n_train}, evidence windows {len(self.evid_starts)} "
              f"({len(self.evid_starts) * OUTPUT_LEN} observations), test windows {len(self.test_starts)}")


def resolve_path(csv_path):
    path = Path(csv_path).expanduser()
    if not path.is_file() and not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find {csv_path!r}. Put it next to this script or in the working directory.")
    return path


def train(model, data, sigma_noise, epochs, tag):
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, epochs))
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(data.train_starts[::TRAIN_STRIDE] + np.random.randint(TRAIN_STRIDE))
        perm = perm[perm <= data.train_starts[-1]]
        perm = perm[data.train_start_mask[perm]]
        tot = 0.0
        for i in range(0, len(perm), BATCH):
            X, Y = data.windows(perm[i:i + BATCH])
            opt.zero_grad()
            mu = model(X)
            loss = data.n_train * ((Y - mu) ** 2).mean() / (2 * sigma_noise ** 2) + model.kl_loss()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            tot += loss.item() * len(X) / len(perm)
        sched.step()
        if not math.isfinite(tot):
            raise FloatingPointError(f"{tag}: -ELBO is {tot} at epoch {ep}")
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"  [{tag}] epoch {ep:3d}  -ELBO/N {tot / data.n_train:10.4f}  "
                  f"KL {model.kl_loss().item():9.0f}  ({time.time() - t0:.0f} s)")


@torch.no_grad()
def sse_at_mean(model, data, starts):
    model.eval()
    sse, n_obs = 0.0, 0
    for i in range(0, len(starts), EVID_BATCH):
        X, Y = data.windows(starts[i:i + EVID_BATCH])
        sse += ((model(X, sample=False) - Y).double() ** 2).sum().item()
        n_obs += Y.numel()
    return sse, n_obs


def sigma_map(sse, n_obs, theta_mu, theta_sigma):
    return sse / n_obs, float(((theta_mu ** 2 + theta_sigma ** 2).sum() / theta_mu.numel()).item())


def second_log_evidence(sse, n_obs, theta_mu, theta_sigma, s2, a2):
    W = theta_mu.numel()
    tr_H_inv = float((theta_sigma ** 2).sum())
    ln_a, ln_n = np.log(a2), np.log(s2)

    log_likelihood = -0.5 * n_obs * np.log(2 * np.pi * s2) - sse / (2 * s2)
    log_prior = -0.5 * W * np.log(2 * np.pi * a2) - float((theta_mu ** 2).sum()) / (2 * a2)
    occam_logdet = float(torch.log(theta_sigma).sum()) + 0.5 * W * np.log(2 * np.pi)

    if HYPERPRIOR == "log-uniform":
        log_hyperprior = -ln_n - ln_a
    elif HYPERPRIOR == "log-normal":
        m, s = HYPERPRIOR_M, HYPERPRIOR_S
        log_hyperprior = (-ln_n - np.log(s) - 0.5 * np.log(2 * np.pi) - (ln_n - m) ** 2 / (2 * s ** 2)
                          - ln_a - np.log(s) - 0.5 * np.log(2 * np.pi) - (ln_a - m) ** 2 / (2 * s ** 2))
    else:
        log_hyperprior = 0.0

    ln_a_ = ln_a if abs(ln_a) > 1e-12 else 1e-12
    ln_n_ = ln_n if abs(ln_n) > 1e-12 else 1e-12
    sigma_ln_p2 = np.sqrt(2 * ln_a ** 2 / max(W - tr_H_inv / ln_a_, 1e-6))
    sigma_ln_noise = np.sqrt(2 * ln_n ** 2 / max(abs(n_obs - W + tr_H_inv / ln_n_), 1e-6))
    log_term = np.log(2 * np.pi * sigma_ln_p2) + np.log(2 * np.pi * sigma_ln_noise)

    no_logdet = log_likelihood + log_prior + log_hyperprior + log_term
    return dict(
        n_obs=n_obs, sse=sse, sigma_noise_sq_map=s2, sigma_pr_sq_map=a2, tr_H_inv=tr_H_inv,
        log_likelihood=log_likelihood, log_prior=log_prior, occam_logdet=occam_logdet,
        log_hyperprior=log_hyperprior, log_term=log_term,
        log_evidence_given_sigma=log_likelihood + log_prior + occam_logdet,
        log_evidence_no_logdet=no_logdet,
        log_evidence=no_logdet + occam_logdet if INCLUDE_LOGDET else no_logdet,
    )


class EvidenceObjective:
    def __init__(self, data, arch_map, category, model_dir):
        self.data = data
        self.arch_map = arch_map
        self.category = category
        self.model_dir = model_dir
        self.min_weights = CATEGORY_BOUNDS[category][0]
        self.max_weights = CATEGORY_BOUNDS[category][1] - 1

    def __call__(self, trial):
        arch = self.arch_map[trial.suggest_categorical("architecture", list(self.arch_map))]
        name = arch_name(arch)

        for t in trial.study.get_trials(states=(TrialState.COMPLETE,)):
            if t.params == trial.params:
                trial.set_user_attr("duplicate_of", t.number)
                return t.value

        t0 = time.time()
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        model = BayesTCN(len(INPUTS), OUTPUT_LEN, arch, INPUT_LEN, prior_sigma=PRIOR_SIGMA0)
        counts = parameter_counts(model)
        if not self.min_weights <= counts["n_weights"] <= self.max_weights:
            raise optuna.TrialPruned(f"{name}: {counts['n_weights']} weights outside "
                                     f"{self.category} [{self.min_weights}, {self.max_weights}]")
        trial.set_user_attr("category", self.category)
        trial.set_user_attr("n_blocks", len(arch["encoder_channels"]))
        for key in ("structure_id", "encoder_channels", "decoder_channels", "kernel_size", "dilation_base"):
            trial.set_user_attr(key, arch[key])
        for key, value in counts.items():
            trial.set_user_attr(key, int(value))
        print(f"\n=== trial {trial.number}: {name}  ({counts['n_weights']} weights, "
              f"{counts['n_trainable_parameters']} trainable parameters)")

        train(model, self.data, SIGMA_NOISE0, EPOCHS, f"{name} train 1")
        theta_mu, theta_sigma = posterior(model)
        sse, n_obs = sse_at_mean(model, self.data, self.data.evid_starts)
        s2, a2 = sigma_map(sse, n_obs, theta_mu, theta_sigma)
        print(f"  sigma_MAP: sigma_noise = {np.sqrt(s2):.4f}, sigma_pr = {np.sqrt(a2):.4f}")

        if RETRAIN:
            torch.manual_seed(SEED)
            np.random.seed(SEED)
            model = BayesTCN(len(INPUTS), OUTPUT_LEN, arch, INPUT_LEN, prior_sigma=np.sqrt(a2))
            train(model, self.data, np.sqrt(s2), RETRAIN_EPOCHS, f"{name} train 2")
            theta_mu, theta_sigma = posterior(model)
            sse, n_obs = sse_at_mean(model, self.data, self.data.evid_starts)

        ev = second_log_evidence(sse, n_obs, theta_mu, theta_sigma, s2, a2)
        sse_te, n_te = sse_at_mean(model, self.data, self.data.test_starts)
        ev["train_rmse"] = np.sqrt(sse / n_obs)
        ev["test_rmse"] = np.sqrt(sse_te / n_te)
        self.save_model(model, arch, name, counts, s2, a2)

        trial.set_user_attr("arch", name)
        trial.set_user_attr("seconds", round(time.time() - t0, 1))
        for k, v in ev.items():
            trial.set_user_attr(k, float(v))
        print(f"  log evidence {ev['log_evidence']:.4e}  (no log-det {ev['log_evidence_no_logdet']:.4e})"
              f"  train RMSE {ev['train_rmse']:.4f}  test RMSE {ev['test_rmse']:.4f}")
        return ev["log_evidence"]

    def save_model(self, model, arch, name, counts, s2, a2):
        os.makedirs(self.model_dir, exist_ok=True)
        architecture = {k: v for k, v in arch.items() if k != "table_weights"}
        torch.save({"state_dict": model.state_dict(), "model_type": "BayesTCN",
                    "architecture": architecture, "input_channels": len(INPUTS),
                    "input_len": INPUT_LEN, "output_len": OUTPUT_LEN,
                    "inputs": list(INPUTS), "target": TARGET,
                    "sample_minutes": SAMPLE_MINUTES, "forecast_minutes": FORECAST_MINUTES,
                    "patient_column": PATIENT_COLUMN, "time_column": TIME_COLUMN,
                    "train_frac_per_patient": TRAIN_FRAC, "target_units": "mg/dL",
                    "split_summary": self.data.split_summary,
                    "pool_kernel": POOL_KERNEL, "upsample_kernel": UPSAMPLE_KERNEL,
                    "final_kernel_size": FINAL_KERNEL,
                    "prior_sigma": float(np.sqrt(a2)) if RETRAIN else float(PRIOR_SIGMA0),
                    "sigma_noise": float(np.sqrt(s2)),
                    "mu_x": self.data.mu_x.tolist(), "sd_x": self.data.sd_x.tolist(), **counts},
                   os.path.join(self.model_dir, f"bayes_tcn_{name}.pt"))


def plausibility(log_ev):
    log_ev = np.asarray(log_ev, float)
    lse = log_ev.max() + np.log(np.sum(np.exp(log_ev - log_ev.max())))
    return np.exp(log_ev - lse), (log_ev - lse) / np.log(10)


def results_table(study):
    rows = [{"trial": t.number, **t.params, **t.user_attrs}
            for t in study.get_trials(states=(TrialState.COMPLETE,))
            if "duplicate_of" not in t.user_attrs]
    if not rows:
        return None
    res = pd.DataFrame(rows)
    res["plausibility"], res["log10_plausibility"] = plausibility(res["log_evidence"])
    res["plausibility_no_logdet"], res["log10_plausibility_no_logdet"] = plausibility(res["log_evidence_no_logdet"])
    res["delta_log_evidence"] = res["log_evidence"] - res["log_evidence"].max()
    res = res.sort_values("log_evidence", ascending=False).reset_index(drop=True)
    res["rank"] = np.arange(1, len(res) + 1)
    res["selected"] = res["rank"] == 1
    return res[[c for c in RESULT_COLUMNS if c in res.columns]]


def feasible_architectures(category):
    feasible, skipped = [], []
    for arch in category_architectures(category):
        try:
            validate_architecture(arch, INPUT_LEN, OUTPUT_LEN)
            feasible.append(arch)
        except ValueError as err:
            skipped.append(str(err))
    if len(INPUTS) == 3:
        for arch in feasible:
            n_weights = parameter_counts(BayesTCN(len(INPUTS), OUTPUT_LEN, arch, INPUT_LEN))["n_weights"]
            if n_weights != arch["table_weights"]:
                raise ValueError(f"{arch['id']}: model has {n_weights} weights, ARCH_POOL lists "
                                 f"{arch['table_weights']}.")
    return feasible, skipped


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evidence-based Optuna search over Bayesian TCN architectures within one capacity category.")
    parser.add_argument("--data", default="population_ohio.csv",
                        help="population CSV with PID, Time, Glucose, MealSize, Insulin")
    parser.add_argument("--category", default="C4", choices=sorted(ARCH_POOL), help="capacity category")
    parser.add_argument("--sampler", default="grid", choices=["grid", "tpe"],
                        help="grid: every architecture once; tpe: --n-trials proposals")
    parser.add_argument("--n-trials", type=int, default=100, help="number of TPE trials")
    parser.add_argument("--storage", default=None, help="Optuna storage URL, e.g. sqlite:///study.db")
    return parser.parse_args()


def main():
    args = parse_args()
    category = args.category
    out_csv = f"tcn_{category.lower()}_glucose60_evidence_results.csv"
    model_dir = f"tcn_{category.lower()}_glucose60_evidence_models"
    study_name = f"bayes_tcn_{category.lower()}_glucose60_evidence"

    feasible, skipped = feasible_architectures(category)
    print(f"{category} pool: {len(ARCH_POOL[category])} structures, "
          f"{len(feasible)} architectures to run, {len(skipped)} skipped")
    for msg in skipped:
        print(f"  skipped {msg}")
    if not feasible:
        raise SystemExit("no architecture fits INPUT_LEN/OUTPUT_LEN")
    arch_map = {arch["id"]: arch for arch in feasible}

    data = WindowData(args.data)
    data.describe()

    space = {"architecture": list(arch_map)}
    if args.sampler == "grid":
        sampler, n_trials = GridSampler(space, seed=SEED), len(arch_map)
    else:
        sampler, n_trials = TPESampler(seed=SEED), args.n_trials
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=study_name,
                                storage=args.storage, load_if_exists=args.storage is not None)
    print(f"search space {space}  ->  {n_trials} trials ({args.sampler})")

    def write_csv(study_, _trial=None):
        table = results_table(study_)
        if table is not None:
            table.to_csv(out_csv, index=False)

    study.optimize(EvidenceObjective(data, arch_map, category, model_dir), n_trials=n_trials,
                   callbacks=[write_csv], catch=(FloatingPointError, RuntimeError))

    res = results_table(study)
    if res is None:
        raise SystemExit("no architecture finished")
    res.to_csv(out_csv, index=False)
    pd.set_option("display.width", 200)
    print("\n=== architectures ranked by log evidence ===")
    print(res[["rank", "arch", "n_weights", "log_evidence", "delta_log_evidence", "plausibility",
               "log10_plausibility", "train_rmse", "test_rmse"]].to_string(index=False))
    best = res.iloc[0]
    print(f"\nselected (highest evidence): {best['arch']}  blocks={best['n_blocks']}  "
          f"kernel={best['kernel_size']}  dilation_base={best['dilation_base']}  weights={best['n_weights']}")
    print(f"log evidence {best['log_evidence']:.6e}, plausibility {best['plausibility']:.4f}")
    if INCLUDE_LOGDET and res["log_evidence_no_logdet"].idxmax() != 0:
        alt = res.loc[res["log_evidence_no_logdet"].idxmax(), "arch"]
        print(f"note: without the log-det Occam term the selection would be {alt}")
    print(f"saved: {out_csv}, models in {model_dir}/")


if __name__ == "__main__":
    main()
