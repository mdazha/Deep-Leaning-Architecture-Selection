import argparse
import copy
import glob
import json
import math
import os
from types import MethodType

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from sklearn.metrics import mean_squared_error, r2_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

CHANNELS = ["Glucose", "MealSize", "Insulin"]
FOLDS = {1: (9, 10), 2: (6, 7), 3: (1, 2), 4: (3, 4), 5: (5, 8)}
PRIORS = {
    "prior_mu": 0.0,
    "prior_sigma": 0.17,
    "posterior_mu_initial": (0.0, 0.1),
    "posterior_rho_initial": (-3.0, 0.1),
}
SIGMA_NOISE_INIT = 10.3
INIT_LOG_NOISE = 4.66
INIT_LOG_ALPHA = -3.54
HYPER_LR = 1e-2
HYPER_STEPS = 1000
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
STAGE1_EPOCHS = 1000
STAGE2_EPOCHS = 1000
MC_SAMPLES = 50
EVAL_BATCH_SIZE = 512
LOG_EVERY = 50


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
            W_sigma = torch.log1p(torch.exp(self.W_rho))
            W = self.W_mu + W_sigma * torch.randn_like(self.W_mu)
            b = None
            if self.use_bias:
                b_sigma = torch.log1p(torch.exp(self.bias_rho))
                b = self.bias_mu + b_sigma * torch.randn_like(self.bias_mu)
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
        W_sigma = torch.log1p(torch.exp(self.W_rho))
        kl = self._kl_div(self.W_mu, W_sigma, self.W_prior_mu, self.W_prior_sigma)
        if self.use_bias:
            b_sigma = torch.log1p(torch.exp(self.bias_rho))
            kl += self._kl_div(self.bias_mu, b_sigma, self.b_prior_mu, self.b_prior_sigma)
        return kl


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


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

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

    def kl_loss(self):
        return self.conv1.kl_loss() + self.conv2.kl_loss()


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dilation_base=4, pool_kernel=3, priors=None):
        super().__init__()
        layers = []
        self.pools = nn.ModuleList()
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = dilation_base ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, priors=priors)]
            if i < num_levels - 1:
                self.pools.append(nn.AvgPool1d(kernel_size=pool_kernel, stride=pool_kernel))
        self.layers = nn.ModuleList(layers)
        self.num_levels = num_levels

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.num_levels - 1:
                x = self.pools[i](x)
        return x


class TCNEncoder(nn.Module):
    def __init__(self, num_channels, input_channels=3, kernel_size=3, dilation_base=4, priors=None):
        super().__init__()
        self.network = TemporalConvNet(input_channels, num_channels, kernel_size,
                                       dilation_base=dilation_base, priors=priors)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        return self.network(x)


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

    def forward(self, x):
        out = self.relu1(self.chomp1(self.conv1(x)))
        out = self.relu2(self.chomp2(self.conv2(out)))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

    def kl_loss(self):
        kl = self.conv1.kl_loss() + self.conv2.kl_loss()
        if self.downsample is not None:
            kl += self.downsample.kl_loss()
        return kl


class TCNDecoder(nn.Module):
    def __init__(self, num_channels, input_channels, output_size, output_seq_len,
                 kernel_size=3, upsample_kernel=3, priors=None):
        super().__init__()
        self.layers = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.output_seq_len = output_seq_len
        in_ch = input_channels

        for i, ch in enumerate(num_channels):
            if i > 0:
                self.upsamples.append(nn.Upsample(scale_factor=upsample_kernel, mode="linear", align_corners=False))
            self.layers.append(CausalBayesianDecoderBlock(in_ch, ch, kernel_size, priors=priors))
            in_ch = ch

        k_final = 5
        self.final = BBBConv1d(in_ch, output_size, kernel_size=k_final, padding=k_final - 1, priors=priors)
        self.final_chomp = Chomp1d(k_final - 1)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i > 0:
                x = self.upsamples[i - 1](x)
            x = layer(x)
        x = self.final_chomp(self.final(x))
        x = x.transpose(1, 2)
        return x[:, -self.output_seq_len:, :]

    def kl_loss(self):
        kl = 0
        for layer in self.layers:
            kl += layer.kl_loss()
        kl += self.final.kl_loss()
        return kl


class TCNModel(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, output_size, output_seq_length,
                 input_channels=3, kernel_size=3, dilation_base=4, priors=None):
        super().__init__()
        self.encoder = TCNEncoder(encoder_channels, input_channels=input_channels, kernel_size=kernel_size,
                                  dilation_base=dilation_base, priors=priors)
        self.decoder = TCNDecoder(decoder_channels, encoder_channels[-1], output_size, output_seq_length,
                                  kernel_size=kernel_size, priors=priors)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def kl_loss(self):
        kl = 0.0
        for block in self.encoder.network.layers:
            kl += block.kl_loss()
        kl += self.decoder.kl_loss()
        return kl


class ELBO(nn.Module):
    def __init__(self, train_size, sigma_noise=10.3):
        super().__init__()
        self.train_size = train_size
        self.sigma_noise = torch.tensor(sigma_noise, dtype=torch.float32)

    def forward(self, prediction, target, kl):
        mse_term = (F.mse_loss(prediction, target, reduction="mean") * self.train_size) / (2 * self.sigma_noise ** 2)
        log_term = 0.5 * self.train_size * torch.log(2.0 * torch.tensor(np.pi) * self.sigma_noise)
        return log_term + mse_term + kl


class NoisePriorHyperparams(nn.Module):
    def __init__(self, init_log_noise=0.0, init_log_alpha=0.0):
        super().__init__()
        self.log_noise = nn.Parameter(torch.tensor(init_log_noise))
        self.log_alpha = nn.Parameter(torch.tensor(init_log_alpha))

    def forward(self):
        return torch.exp(self.log_noise), torch.exp(self.log_alpha)


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(arch, output_seq_length, priors, device):
    return TCNModel(
        encoder_channels=arch["encoder_channels"],
        decoder_channels=arch["decoder_channels"],
        output_size=1,
        output_seq_length=output_seq_length,
        input_channels=len(CHANNELS),
        kernel_size=arch["kernel_size"],
        dilation_base=arch["dilation_base"],
        priors=priors,
    ).to(device)


def arch_tag(arch):
    enc = "-".join(map(str, arch["encoder_channels"]))
    dec = "-".join(map(str, arch["decoder_channels"]))
    return f"enc-{enc}__dec-{dec}__k{arch['kernel_size']}__b{arch['dilation_base']}"


def count_parameters(model):
    return int(sum(p.numel() for name, p in model.named_parameters() if not name.endswith("_rho")))


def load_population(path):
    df = pd.read_csv(path, delimiter=",")
    missing = [c for c in ["PID"] + CHANNELS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}; the UVA/Padova file needs PID, Glucose, MealSize, Insulin.")
    patients = {}
    for number, (pid, sub) in enumerate(df.groupby("PID", sort=False), start=1):
        patients[number] = {"pid": str(pid), "frame": sub.reset_index(drop=True)}
    return df, patients


def fold_patients(patients, fold):
    held_out = FOLDS[fold]
    if max(held_out) > len(patients):
        raise ValueError(f"Fold {fold} needs at least {max(held_out)} patients; found {len(patients)}.")
    training = [n for n in patients if n not in held_out]
    return training, list(held_out)


def train_model(model, loader, train_size, sigma_noise, epochs, device, use_scheduler=True, label="train"):
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    sch = ReduceLROnPlateau(opt, mode="min", factor=0.8, patience=50) if use_scheduler else None
    criterion = ELBO(train_size=train_size, sigma_noise=sigma_noise).to(device)
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        total_kl = 0.0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            opt.zero_grad()
            output = model(x_batch)
            kl_loss = model.kl_loss()
            loss = criterion(output, y_batch, kl_loss)
            loss.backward()
            nn_utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            total_kl += kl_loss.item()
        avg_loss = total_loss / max(1, len(loader))
        if sch is not None:
            sch.step(avg_loss)
        if ep % LOG_EVERY == 0 or ep == epochs - 1:
            print(f"[{label}] Epoch {ep + 1}, Loss: {avg_loss:.4f}, KL Loss: {total_kl / max(1, len(loader)):.4f}",
                  flush=True)


def extract_mean_and_std(model):
    mus = []
    sigmas = []
    for _, module in model.named_modules():
        if isinstance(module, BBBConv1d):
            mus.append(module.W_mu.data.view(-1))
            sigmas.append(torch.log1p(torch.exp(module.W_rho.data)).view(-1))
            if module.use_bias:
                mus.append(module.bias_mu.data.view(-1))
                sigmas.append(torch.log1p(torch.exp(module.bias_rho.data)).view(-1))
    return torch.cat(mus, dim=0), torch.cat(sigmas, dim=0)


def make_convs_deterministic(model):
    for layer in model.modules():
        if isinstance(layer, BBBConv1d):
            orig_forward = layer.forward

            def forward_no_sample(self, x, orig=orig_forward):
                return orig(x, sample=False)

            layer.forward = MethodType(forward_no_sample, layer)
    return model


@torch.no_grad()
def posterior_draw(model, theta_mu=None):
    draw = copy.deepcopy(model)
    idx = 0
    for _, module in draw.named_modules():
        if isinstance(module, BBBConv1d):
            pairs = [(module.W_mu, module.W_rho)]
            if module.use_bias:
                pairs.append((module.bias_mu, module.bias_rho))
            for mu, rho in pairs:
                if theta_mu is not None:
                    mu.copy_(theta_mu[idx: idx + mu.numel()].view_as(mu))
                idx += mu.numel()
                mu.copy_(mu + torch.log1p(torch.exp(rho)) * torch.randn_like(mu))
    return make_convs_deterministic(draw).eval()


@torch.no_grad()
def predict(model, X, device):
    return torch.cat([model(X[i:i + EVAL_BATCH_SIZE].to(device)) for i in range(0, len(X), EVAL_BATCH_SIZE)])


def negative_log_evidence(model, theta_mu, theta_sigma, X, Y, sigma_noise_sq, alpha_sq, device):
    pred = predict(posterior_draw(model, theta_mu), X, device)
    pred_flat = pred.reshape(-1)
    Y_flat = Y.to(device).reshape(-1)
    N = Y_flat.shape[0]
    sse = torch.sum((pred_flat - Y_flat) ** 2)
    P = theta_mu.shape[0]
    sum_sq_means = torch.sum(theta_mu ** 2 + theta_sigma ** 2)
    sum_log_sigma = torch.sum(torch.log(theta_sigma))
    nle = (0.5 * N * torch.log(2 * torch.tensor(np.pi))
           + 0.5 * N * torch.log(sigma_noise_sq)
           + 0.5 * sse / sigma_noise_sq
           + 0.5 * P * torch.log(2 * torch.tensor(np.pi))
           + 0.5 * P * torch.log(alpha_sq)
           + 0.5 * sum_sq_means / alpha_sq
           - sum_log_sigma)
    return nle


def optimize_noise_and_prior(model, theta_mu, theta_sigma, X, Y, device,
                             init_log_noise=INIT_LOG_NOISE, init_log_alpha=INIT_LOG_ALPHA,
                             lr=HYPER_LR, num_steps=HYPER_STEPS):
    hyper = NoisePriorHyperparams(init_log_noise, init_log_alpha).to(device)
    opt = torch.optim.Adam(hyper.parameters(), lr=lr)
    model.eval()
    for step in range(num_steps):
        opt.zero_grad()
        sigma_noise_sq, alpha_sq = hyper()
        nle = negative_log_evidence(model, theta_mu, theta_sigma, X, Y, sigma_noise_sq, alpha_sq, device)
        nle.backward()
        opt.step()
        if step % 20 == 0 or step == num_steps - 1:
            print(f"Step {step:03d} | -LogEvidence: {nle.item():.4f} "
                  f"| sigma_noise^2={sigma_noise_sq.item():.4f} | alpha^2={alpha_sq.item():.4f}", flush=True)
    final_sigma_noise_sq, final_alpha_sq = hyper()
    return final_sigma_noise_sq.detach().cpu().item(), final_alpha_sq.detach().cpu().item()


@torch.no_grad()
def export_evidence_constants(model, theta_mu, theta_sigma, X, Y, device, out_path):
    model.eval()
    pred = predict(posterior_draw(model, theta_mu), X, device)
    pred_flat = pred.reshape(-1).cpu().numpy()
    Y_flat = Y.reshape(-1).cpu().numpy()
    sse = float(np.sum((pred_flat - Y_flat) ** 2))
    N = int(Y_flat.size)
    P = int(theta_mu.numel())
    sum_sq_means = float(torch.sum(theta_mu ** 2 + theta_sigma ** 2).cpu().item())
    sum_log_sigma = float(torch.sum(torch.log(theta_sigma)).cpu().item())
    c_like = 0.5 * N * np.log(2.0 * np.pi)
    c_prior = 0.5 * P * np.log(2.0 * np.pi)
    np.savez(out_path, N=N, P=P, sse=sse, sum_sq_means=sum_sq_means, sum_log_sigma=sum_log_sigma,
             c_like=c_like, c_prior=c_prior)


def compute_log_evidence(model, X_train, Y_train, theta_mu, theta_sigma, sigma_noise_sq_opt, alpha_sq_opt, device,
                         hyperprior_type="log-normal", hyperprior_m=0, hyperprior_s=1):
    tr_H_inv = float(torch.sum(theta_sigma ** 2).cpu().numpy())
    P = int(theta_mu.shape[0])
    N_D = int(X_train.shape[0])

    ln_alpha_sq = np.log(alpha_sq_opt)
    ln_sigma_noise_sq = np.log(sigma_noise_sq_opt)

    numerator_p2 = 2 * ln_alpha_sq ** 2
    denominator_p2 = P - (1 / ln_alpha_sq) * tr_H_inv
    if denominator_p2 <= 0:
        raise ValueError("Denominator for sigma^2_ln(sigma_p^2) is non-positive.")
    sigma_ln_p2 = np.sqrt(numerator_p2 / denominator_p2)

    numerator_noise = 2 * ln_sigma_noise_sq ** 2
    denominator_noise = abs(N_D - P + (1 / ln_sigma_noise_sq) * tr_H_inv)
    if denominator_noise <= 0:
        raise ValueError("Denominator for sigma^2_ln(sigma_noise^2) is non-positive.")
    sigma_ln_noise = np.sqrt(numerator_noise / denominator_noise)

    pred_sum = None
    for _ in range(MC_SAMPLES):
        draw = predict(posterior_draw(model), X_train, device).cpu().numpy()
        pred_sum = draw if pred_sum is None else pred_sum + draw
    mean_pred = pred_sum / MC_SAMPLES
    y_true_flat = Y_train.cpu().numpy().flatten()
    sse = abs(float(np.sum((mean_pred.flatten() - y_true_flat) ** 2)))

    sum_log_sigma = np.sum(np.log(theta_sigma.detach().cpu().numpy()))
    log_likelihood = -0.5 * N_D * np.log(2 * np.pi * sigma_noise_sq_opt) - sse / (2 * np.log(sigma_noise_sq_opt))

    sum_sq_means = float(torch.sum(theta_mu ** 2).cpu().numpy())
    log_prior = -0.5 * P * np.log(2 * np.pi * alpha_sq_opt) - sum_sq_means / (2 * alpha_sq_opt)

    if hyperprior_type == "log-uniform":
        log_hyperprior = -np.log(sigma_noise_sq_opt) - np.log(alpha_sq_opt)
    elif hyperprior_type == "log-normal":
        s = hyperprior_s
        m = hyperprior_m
        log_hyperprior = (
            -np.log(sigma_noise_sq_opt) - np.log(s)
            - 0.5 * np.log(2 * np.pi)
            - ((np.log(sigma_noise_sq_opt) - m) ** 2) / (2 * s ** 2)
            - np.log(alpha_sq_opt) - np.log(s)
            - 0.5 * np.log(2 * np.pi)
            - ((np.log(alpha_sq_opt) - m) ** 2) / (2 * s ** 2)
        )
    else:
        raise ValueError(f"Unsupported hyperprior_type: {hyperprior_type}")

    log_term = np.log(2 * np.pi * sigma_ln_p2) + np.log(2 * np.pi * sigma_ln_noise)
    log_evidence = log_likelihood + log_prior + log_hyperprior + log_term

    print("----- Log Evidence Terms -----")
    print(f"P (num params): {P}")
    print(f"N_D (num data): {N_D}")
    print(f"tr_H_inv: {tr_H_inv:.4f}")
    print(f"ln_alpha_sq: {ln_alpha_sq:.4f}")
    print(f"ln_sigma_noise_sq: {ln_sigma_noise_sq:.4f}")
    print(f"sigma_ln_p2: {sigma_ln_p2:.4f}")
    print(f"sigma_ln_noise: {sigma_ln_noise:.4f}")
    print(f"sse (sum squared errors): {sse:.4f}")
    print(f"sum_log_sigma: {sum_log_sigma:.4f}")
    print(f"log_likelihood: {log_likelihood:.4f}")
    print(f"log_prior: {log_prior:.4f}")
    print(f"log_hyperprior: {log_hyperprior:.4f}")
    print(f"log_term: {log_term:.4f}")
    print(f"TOTAL log_evidence: {log_evidence:.4f}")
    print("-----------------------------", flush=True)
    return float(log_evidence)


@torch.no_grad()
def log_predictive_ll_last_series(model, X_test, Y_test, device, n_samples=MC_SAMPLES, sigma_constant=10.3):
    X_last = X_test[-1].unsqueeze(0).to(device)
    Y_last = Y_test[-1].unsqueeze(0).to(device)
    sigma_obs = torch.tensor(float(sigma_constant), device=device)
    logsum = None
    for _ in range(n_samples):
        pred = posterior_draw(model)(X_last)
        lp = torch.distributions.Normal(pred, sigma_obs).log_prob(Y_last)
        logsum = lp if logsum is None else torch.logaddexp(logsum, lp)
    LL = logsum - math.log(n_samples)
    return LL.sum().item(), LL.mean().item()


def save_posterior_npz(model, path):
    np.savez(path, **{name: p.detach().cpu().numpy() for name, p in model.named_parameters()})
    print(f"[posterior] saved all tensors -> {path}")


def warm_start_from_npz(model, npz_path):
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"Warm-start NPZ not found: {os.path.abspath(npz_path)}")
    parameters = dict(model.named_parameters())
    values = {}
    problems = []
    with np.load(npz_path, allow_pickle=False) as saved:
        for name, parameter in parameters.items():
            if name not in saved:
                problems.append(f"missing parameter: {name}")
                continue
            value = saved[name]
            if tuple(value.shape) != tuple(parameter.shape):
                problems.append(f"shape mismatch for {name}: NPZ {value.shape}, model {tuple(parameter.shape)}")
                continue
            if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
                problems.append(f"non-numeric or non-finite parameter: {name}")
                continue
            values[name] = torch.as_tensor(value, dtype=parameter.dtype, device=parameter.device)
    if problems:
        raise ValueError(f"Cannot warm-start from {npz_path}:\n" + "\n".join(problems))
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(values[name])
    print(f"[warm start] Loaded {len(values)} parameter tensors from {npz_path}")


def inv_softplus_torch(x):
    return torch.log(torch.clamp(torch.exp(x) - 1.0, min=1e-12))


def copy_posterior_to_prior(dst_model, src_model):
    src_modules = dict(src_model.named_modules())
    copied = 0
    for n_dst, m_dst in dst_model.named_modules():
        if not isinstance(m_dst, BBBConv1d):
            continue
        m_src = src_modules.get(n_dst)
        if m_src is None or not isinstance(m_src, BBBConv1d):
            raise ValueError(f"No matching source module for {n_dst}")
        with torch.no_grad():
            m_dst.W_prior_mu.copy_(m_src.W_mu.data)
            m_dst.W_prior_sigma.copy_(torch.log1p(torch.exp(m_src.W_rho.data)))
            if m_dst.use_bias and m_src.bias_mu is not None:
                m_dst.b_prior_mu.copy_(m_src.bias_mu.data)
                m_dst.b_prior_sigma.copy_(torch.log1p(torch.exp(m_src.bias_rho.data)))
        copied += 1
    print(f"[prior] Copied posterior to prior for {copied} conv modules.")


def init_posterior_from_prior(model):
    with torch.no_grad():
        for m in model.modules():
            if not isinstance(m, BBBConv1d):
                continue
            m.W_mu.copy_(m.W_prior_mu)
            m.W_rho.copy_(inv_softplus_torch(m.W_prior_sigma.clamp_min(1e-6)))
            if m.use_bias:
                m.bias_mu.copy_(m.b_prior_mu)
                m.bias_rho.copy_(inv_softplus_torch(m.b_prior_sigma.clamp_min(1e-6)))


def run_two_stage_training(arch, output_seq_length, X_train, Y_train, device, warm_start_file=None):
    train_size = len(X_train)
    loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=False)

    model = build_model(arch, output_seq_length, PRIORS, device)
    if warm_start_file is not None:
        warm_start_from_npz(model, warm_start_file)
    train_model(model, loader, train_size, SIGMA_NOISE_INIT, STAGE1_EPOCHS, device,
                use_scheduler=True, label="stage 1")

    theta_mu, theta_sigma = extract_mean_and_std(model)
    sigma_noise_sq_opt, alpha_sq_opt = optimize_noise_and_prior(model, theta_mu, theta_sigma, X_train, Y_train, device)

    priors_retrained = dict(PRIORS)
    priors_retrained["prior_sigma"] = float(np.sqrt(alpha_sq_opt))
    model_retrain = build_model(arch, output_seq_length, priors_retrained, device)
    copy_posterior_to_prior(model_retrain, model)
    init_posterior_from_prior(model_retrain)

    sigma_noise = sigma_noise_sq_opt ** 0.5
    train_model(model_retrain, loader, train_size, sigma_noise, 1, device, use_scheduler=True, label="stage 2")
    train_model(model_retrain, loader, train_size, sigma_noise, STAGE2_EPOCHS, device,
                use_scheduler=False, label="stage 2")

    theta_mu2, theta_sigma2 = extract_mean_and_std(model_retrain)
    return {
        "model": model,
        "model_retrain": model_retrain,
        "theta_mu": theta_mu,
        "theta_sigma": theta_sigma,
        "theta_mu2": theta_mu2,
        "theta_sigma2": theta_sigma2,
        "sigma_noise_sq_opt": sigma_noise_sq_opt,
        "alpha_sq_opt": alpha_sq_opt,
    }


@torch.no_grad()
def mc_predict_windows(model, X_test, Y_test, indices, device, n_samples=MC_SAMPLES):
    X_sel = X_test[indices]
    preds = np.stack([predict(posterior_draw(model), X_sel, device)[..., 0].cpu().numpy()
                      for _ in range(n_samples)], axis=0)
    y_true = Y_test[indices][..., 0].cpu().numpy()
    return preds, y_true


def evaluate_and_save(fit, arch, X_train, Y_train, X_test, Y_test, window_pids, window_starts,
                      input_seq_length, output_seq_length, out_dir, device):
    os.makedirs(out_dir, exist_ok=True)
    tag = arch_tag(arch)

    export_evidence_constants(fit["model"], fit["theta_mu"], fit["theta_sigma"], X_train, Y_train, device,
                              os.path.join(out_dir, f"evidence_constants__{tag}.npz"))
    posterior_path = os.path.join(out_dir, f"posterior_params__{tag}.npz")
    save_posterior_npz(fit["model_retrain"], posterior_path)

    indices = list(range(0, len(X_test), output_seq_length))
    preds, y_true = mc_predict_windows(fit["model_retrain"], X_test, Y_test, indices, device)
    mean_ = preds.mean(axis=0)
    var_ = preds.var(axis=0)
    std_ = np.sqrt(var_)
    lower = mean_ - 2.0 * std_
    upper = mean_ + 2.0 * std_

    frames = []
    for j, idx in enumerate(indices):
        t0 = int(window_starts[idx]) + input_seq_length
        row = {
            "PID": np.repeat(window_pids[idx], output_seq_length),
            "window_index": np.repeat(idx, output_seq_length),
            "time_index": np.arange(t0, t0 + output_seq_length),
            "actual": y_true[j],
            "mean": mean_[j],
            "variance": var_[j],
            "std": std_[j],
            "lower_2sigma": lower[j],
            "upper_2sigma": upper[j],
        }
        for k in range(preds.shape[0]):
            row[f"sample_{k + 1}"] = preds[k, j]
        frames.append(pd.DataFrame(row))
    csv_path = os.path.join(out_dir, f"mc_predictions__{tag}.csv")
    pd.concat(frames, axis=0, ignore_index=True).to_csv(csv_path, index=False)
    print(f"Saved MC predictions for {len(indices)} windows -> {csv_path}")

    y_true_comb = y_true.ravel()
    y_pred_comb = mean_.ravel()
    rmse_comb = float(np.sqrt(mean_squared_error(y_true_comb, y_pred_comb)))
    r2_comb = float(r2_score(y_true_comb, y_pred_comb))
    rl2e_comb = float(np.sqrt(np.sum((y_pred_comb - y_true_comb) ** 2) / np.sum(y_true_comb ** 2)))
    print(f"Combined mean-of-theta prediction RMSE: {rmse_comb:.4f}")
    print(f"Combined mean-of-theta prediction R^2:  {r2_comb:.4f}")
    print(f"Combined mean-of-theta prediction rL2E: {rl2e_comb:.4f}")

    try:
        log_evidence = compute_log_evidence(fit["model_retrain"], X_train, Y_train, fit["theta_mu2"],
                                            fit["theta_sigma2"], fit["sigma_noise_sq_opt"], fit["alpha_sq_opt"],
                                            device, hyperprior_type="log-normal", hyperprior_m=0, hyperprior_s=1)
        print(f"Log Evidence: {log_evidence:.2f}")
    except ValueError as error:
        print(f"Log evidence skipped: {error}")
        log_evidence = float("nan")

    lppl_total, lppl_mean = log_predictive_ll_last_series(fit["model_retrain"], X_test, Y_test, device,
                                                          sigma_constant=fit["sigma_noise_sq_opt"] ** 0.5)
    print(f"LPPL (last series): total={lppl_total:.2f}, mean/elem={lppl_mean:.6f}")

    return {
        "arch_id": arch["id"],
        "level": arch["level"],
        "encoder_channels": arch["encoder_channels"],
        "decoder_channels": arch["decoder_channels"],
        "kernel_size": arch["kernel_size"],
        "dilation_base": arch["dilation_base"],
        "n_params": count_parameters(fit["model_retrain"]),
        "input_seq_length": input_seq_length,
        "output_seq_length": output_seq_length,
        "n_train_windows": int(len(X_train)),
        "n_test_windows": int(len(X_test)),
        "sigma_noise_sq_opt": fit["sigma_noise_sq_opt"],
        "alpha_sq_opt": fit["alpha_sq_opt"],
        "log_evidence": log_evidence,
        "rmse": rmse_comb,
        "r2": r2_comb,
        "rl2e": rl2e_comb,
        "lppl_total": lppl_total,
        "lppl_mean": lppl_mean,
        "posterior_file": os.path.abspath(posterior_path),
    }


def write_json(path, payload):
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def read_json(path):
    with open(path) as handle:
        return json.load(handle)


def level_directory(root, fold, level):
    return os.path.join(root, f"fold{fold}", f"level{level}")


INPUT_SEQ_LENGTH = 1440
MEAL_THRESH = 0.0
STRIDE = 1

CANDIDATES = {
    1: [
        ("L1A1", [2, 4, 8], 9, 9),
        ("L1A2", [2, 4, 8], 11, 8),
        ("L1A3", [2, 4, 8], 11, 9),
        ("L1A4", [2, 4, 8], 13, 8),
        ("L1A5", [2, 4, 8, 16], 7, 5),
        ("L1A6", [4, 8, 16], 9, 9),
        ("L1A7", [4, 8, 16], 11, 8),
        ("L1A8", [4, 8, 16], 11, 9),
        ("L1A9", [2, 4, 8, 16], 11, 4),
        ("L1A10", [4, 8, 16], 13, 8),
    ],
    2: [
        ("L2A1", [4, 8, 16, 32], 7, 5),
        ("L2A2", [2, 4, 8, 16, 32], 7, 3),
        ("L2A3", [8, 16, 32], 9, 9),
        ("L2A4", [2, 4, 8, 16, 32], 9, 3),
        ("L2A5", [8, 16, 32], 11, 8),
        ("L2A6", [8, 16, 32], 11, 9),
        ("L2A7", [4, 8, 16, 32], 11, 4),
        ("L2A8", [2, 4, 8, 16, 32, 64], 3, 3),
        ("L2A9", [8, 16, 32], 13, 8),
    ],
    3: [
        ("L3A1", [8, 16, 32, 64], 7, 5),
        ("L3A2", [4, 8, 16, 32, 64], 7, 3),
        ("L3A3", [16, 32, 64], 9, 9),
        ("L3A4", [4, 8, 16, 32, 64], 9, 3),
        ("L3A5", [16, 32, 64], 11, 8),
        ("L3A6", [16, 32, 64], 11, 9),
        ("L3A7", [8, 16, 32, 64], 11, 4),
        ("L3A8", [4, 8, 16, 32, 64, 128], 3, 3),
        ("L3A9", [16, 32, 64], 13, 8),
        ("L3A10", [2, 4, 8, 16, 32, 64], 13, 2),
    ],
}


def make_arch(arch_id, level, channels, kernel_size, dilation_base):
    return {
        "id": arch_id,
        "level": level,
        "encoder_channels": list(channels),
        "decoder_channels": list(reversed(channels)),
        "kernel_size": kernel_size,
        "dilation_base": dilation_base,
    }


def select_architectures(level, requested):
    pool = {spec[0]: make_arch(spec[0], level, *spec[1:]) for spec in CANDIDATES[level]}
    if not requested:
        return list(pool.values())
    unknown = [a for a in requested if a not in pool]
    if unknown:
        raise ValueError(f"Unknown architectures for level {level}: {unknown}; available: {list(pool)}")
    return [pool[a] for a in requested]


def meal_safe_output_length(df, meal_thresh=MEAL_THRESH):
    meal_mask = df["MealSize"].fillna(0).to_numpy() > meal_thresh
    starts = np.where(meal_mask & np.concatenate(([True], ~meal_mask[:-1])))[0]
    ends = np.where(meal_mask & np.concatenate((~meal_mask[1:], [True])))[0]
    if len(starts) < 2:
        return 60
    gaps = starts[1:] - ends[:-1] - 1
    valid_gaps = gaps[gaps > 0]
    return 1 if len(valid_gaps) == 0 else int(valid_gaps.min())


def create_meal_safe_sequences(patients, numbers, input_len, output_len, stride=STRIDE, meal_thresh=MEAL_THRESH):
    xs, ys, starts, pids = [], [], [], []
    for number in numbers:
        sub = patients[number]["frame"]
        sub_len = len(sub)
        if sub_len < input_len + output_len:
            continue
        meal_mask = sub["MealSize"].fillna(0).to_numpy() > meal_thresh
        arr = sub.loc[:, CHANNELS].to_numpy(dtype=np.float32)
        glu = sub["Glucose"].to_numpy(dtype=np.float32)
        for i in range(0, sub_len - input_len - output_len + 1, stride):
            y_lo = i + input_len
            y_hi = y_lo + output_len
            if meal_mask[y_lo:y_hi].any():
                continue
            xs.append(arr[i:i + input_len])
            ys.append(glu[y_lo:y_hi, None])
            starts.append(i)
            pids.append(patients[number]["pid"])
    if not xs:
        raise ValueError("No meal-safe windows found; check MealSize or the sequence lengths.")
    return (torch.from_numpy(np.asarray(xs, dtype=np.float32)),
            torch.from_numpy(np.asarray(ys, dtype=np.float32)),
            np.asarray(starts, dtype=np.int64),
            np.asarray(pids))


def summarize_level(level_dir):
    rows = [read_json(path) for path in sorted(glob.glob(os.path.join(level_dir, "*", "summary.json")))]
    table = pd.DataFrame(rows).sort_values("log_evidence", ascending=False, na_position="last")
    columns = ["arch_id", "n_params", "encoder_channels", "kernel_size", "dilation_base", "log_evidence",
               "sigma_noise_sq_opt", "alpha_sq_opt", "rmse", "r2", "rl2e", "lppl_total", "lppl_mean",
               "posterior_file"]
    table = table[columns]
    table.to_csv(os.path.join(level_dir, "level_summary.csv"), index=False)
    return table


def parse_args():
    parser = argparse.ArgumentParser(
        description="EVIDENT population-based training and evidence ranking of Bayesian TCNs (UVA/Padova T1D).")
    parser.add_argument("--data", default="diabetes_data_insulin.csv",
                        help="UVA/Padova population CSV with columns PID, Glucose, MealSize, Insulin")
    parser.add_argument("--fold", type=int, default=1, choices=sorted(FOLDS), help="patient-wise fold (Table 2)")
    parser.add_argument("--level", type=int, default=1, choices=sorted(CANDIDATES), help="capacity level")
    parser.add_argument("--archs", nargs="*", default=None, help="architecture IDs within the level, e.g. L2A4")
    parser.add_argument("--out", default="evident_population", help="output directory")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = select_device()

    df, patients = load_population(args.data)
    training, held_out = fold_patients(patients, args.fold)
    input_seq_length = INPUT_SEQ_LENGTH
    output_seq_length = meal_safe_output_length(df)

    print(f"Device: {device}")
    print(f"Fold {args.fold}: training patients {[patients[n]['pid'] for n in training]}, "
          f"held-out patients {[patients[n]['pid'] for n in held_out]}")
    print(f"input_seq_length={input_seq_length}, output_seq_length={output_seq_length}")

    X_train, Y_train, _, _ = create_meal_safe_sequences(patients, training, input_seq_length, output_seq_length)
    X_test, Y_test, starts_test, pids_test = create_meal_safe_sequences(patients, held_out, input_seq_length,
                                                                        output_seq_length)
    print(f"Meal-safe windows: train {len(X_train)}, held-out {len(X_test)}")

    level_dir = level_directory(args.out, args.fold, args.level)
    for arch in select_architectures(args.level, args.archs):
        print(f"\nEvaluating architecture {arch['id']}: {arch}", flush=True)
        seed_everything(args.seed)
        fit = run_two_stage_training(arch, output_seq_length, X_train, Y_train, device)
        arch_dir = os.path.join(level_dir, arch["id"])
        summary = evaluate_and_save(fit, arch, X_train, Y_train, X_test, Y_test, pids_test, starts_test,
                                    input_seq_length, output_seq_length, arch_dir, device)
        summary.update({
            "fold": args.fold,
            "data": os.path.abspath(args.data),
            "training_patients": [patients[n]["pid"] for n in training],
            "held_out_patients": [patients[n]["pid"] for n in held_out],
        })
        write_json(os.path.join(arch_dir, "summary.json"), summary)

    table = summarize_level(level_dir)
    print("\nLevel summary (ranked by log evidence):")
    print(table[["arch_id", "n_params", "log_evidence", "sigma_noise_sq_opt", "alpha_sq_opt", "rmse"]]
          .to_string(index=False))
    ranked = table.dropna(subset=["log_evidence"])
    if len(ranked):
        print(f"Best by log evidence: {ranked.iloc[0]['arch_id']} | value = {ranked.iloc[0]['log_evidence']:.2f}")


if __name__ == "__main__":
    main()
