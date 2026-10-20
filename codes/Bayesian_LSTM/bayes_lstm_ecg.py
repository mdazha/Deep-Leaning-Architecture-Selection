import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

FS = 256
INPUTS = ["ecg_mV", "t_since_R_s", "rr_last_s", "resp_belt_au"]
TARGET = "ecg_mV"
INPUT_LEN, OUTPUT_LEN = 256, 64
PATCH, HIDDEN = 4, 13
N_LAYERS = 2
PRIOR = dict(prior_mu=0.0, prior_sigma=0.17)
LEARN_SIGMA = True
SIGMA_NOISE = 0.025
SIGMA_MIN = 0.005
TRAIN_FRAC = 0.8
EPOCHS, BATCH, LR, TRAIN_STRIDE = 100, 256, 3e-3, 4
TEST_STRIDE, N_MC = 16, 100
SEED = 0

AR_IN, AR_OUT = np.arange(INPUT_LEN), np.arange(OUTPUT_LEN)
BLUE, ORANGE, AQUA, GREY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#9a9994", "#0b0b0b"


def _kl(mu_q, sig_q, mu_p, sig_p):
    return 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p) ** 2
                  + ((mu_p - mu_q) / sig_p) ** 2).sum()


class _Bayes(nn.Module):
    def _init(self, shape, n_out, prior_mu, prior_sigma):
        self.W_mu = nn.Parameter(torch.empty(shape).normal_(0.0, 0.1))
        self.W_rho = nn.Parameter(torch.empty(shape).normal_(-3.0, 0.1))
        self.bias_mu = nn.Parameter(torch.empty(n_out).normal_(0.0, 0.1))
        self.bias_rho = nn.Parameter(torch.empty(n_out).normal_(-3.0, 0.1))
        self.register_buffer("W_prior_mu", torch.full(shape, prior_mu))
        self.register_buffer("W_prior_sigma", torch.full(shape, prior_sigma))
        self.register_buffer("b_prior_mu", torch.full((n_out,), prior_mu))
        self.register_buffer("b_prior_sigma", torch.full((n_out,), prior_sigma))

    def _sample(self, sample):
        if not sample:
            return self.W_mu, self.bias_mu
        W = self.W_mu + F.softplus(self.W_rho) * torch.randn_like(self.W_mu)
        b = self.bias_mu + F.softplus(self.bias_rho) * torch.randn_like(self.bias_mu)
        return W, b

    def kl_loss(self):
        return (_kl(self.W_mu, F.softplus(self.W_rho), self.W_prior_mu, self.W_prior_sigma)
                + _kl(self.bias_mu, F.softplus(self.bias_rho), self.b_prior_mu, self.b_prior_sigma))


class BayesLinear(_Bayes):
    def __init__(self, n_in, n_out, prior_mu=0.0, prior_sigma=0.17):
        super().__init__()
        self._init((n_out, n_in), n_out, prior_mu, prior_sigma)

    def forward(self, x, sample=True):
        W, b = self._sample(sample)
        return F.linear(x, W, b)


class BayesLSTM(nn.Module):
    def __init__(self, c_in, horizon):
        super().__init__()
        n_in = c_in * PATCH
        self.ih = nn.ModuleList()
        self.hh = nn.ModuleList()
        for layer in range(N_LAYERS):
            n_layer_in = n_in if layer == 0 else HIDDEN
            self.ih.append(BayesLinear(n_layer_in, 4 * HIDDEN, **PRIOR))
            self.hh.append(BayesLinear(HIDDEN, 4 * HIDDEN, **PRIOR))
        self.head = BayesLinear(HIDDEN, 2 * horizon, **PRIOR)
        with torch.no_grad():
            for ih in self.ih:
                ih.bias_mu[HIDDEN:2 * HIDDEN] += 1.0
        self.kernel = nn.LSTM(n_in, HIDDEN, num_layers=N_LAYERS, batch_first=True)
        self.kernel.requires_grad_(False)

    def forward(self, x, sample=True):
        B = x.shape[0]
        x = x.reshape(B, INPUT_LEN // PATCH, -1)
        w = {}
        for layer in range(N_LAYERS):
            W_ih, b_ih = self.ih[layer]._sample(sample)
            W_hh, b_hh = self.hh[layer]._sample(sample)
            w[f"weight_ih_l{layer}"], w[f"bias_ih_l{layer}"] = W_ih, b_ih
            w[f"weight_hh_l{layer}"], w[f"bias_hh_l{layer}"] = W_hh, b_hh
        _, (h, _) = functional_call(self.kernel, w, (x,))
        mu, s = self.head(h[-1], sample).chunk(2, dim=1)
        if LEARN_SIGMA:
            sigma = F.softplus(s) + SIGMA_MIN
        else:
            sigma = torch.full_like(mu, SIGMA_NOISE)
        return mu, sigma

    def kl_loss(self):
        return sum(m.kl_loss() for m in self.modules() if isinstance(m, _Bayes))


def neg_elbo(mu, sigma, y, kl, n_train):
    loss = ((y - mu) ** 2).mean() / (2 * SIGMA_NOISE ** 2)
    if LEARN_SIGMA:
        loss = loss + (0.5 * ((y - mu.detach()) / sigma) ** 2 + torch.log(sigma)).mean()
    return loss * n_train + kl


def windows(starts, feats_n, target_t):
    starts = np.asarray(starts)
    X = feats_n[starts[:, None] + AR_IN]
    Y = target_t[starts[:, None] + INPUT_LEN + AR_OUT]
    return X, Y


def train(model, train_starts, feats_n, target_t):
    n_train = len(train_starts)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(train_starts[::TRAIN_STRIDE] + np.random.randint(TRAIN_STRIDE))
        perm = perm[perm <= train_starts[-1]]
        tot = 0.0
        for i in range(0, len(perm), BATCH):
            X, Y = windows(perm[i:i + BATCH], feats_n, target_t)
            opt.zero_grad()
            mu, sigma = model(X)
            loss = neg_elbo(mu, sigma, Y, model.kl_loss(), n_train)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            tot += loss.item() * len(X) / len(perm)
        sched.step()
        if ep % 10 == 0 or ep == EPOCHS - 1:
            print(f"epoch {ep:3d}  -ELBO/N {tot / n_train:8.4f}  KL {model.kl_loss().item():8.0f}"
                  f"  ({time.time() - t0:.0f} s)")


@torch.no_grad()
def predict(model, starts, feats_n, target_t, n_mc=N_MC):
    X, _ = windows(starts, feats_n, target_t)
    mus, var_a = [], []
    for _ in range(n_mc):
        m, s = model(X, sample=True)
        mus.append(m)
        var_a.append(s ** 2)
    mus = torch.stack(mus)
    mean = mus.mean(0)
    var_e = mus.var(0, unbiased=False)
    var_a = torch.stack(var_a).mean(0)
    return [a.numpy() for a in (mean, var_e.sqrt(), var_a.sqrt(), (var_e + var_a).sqrt())]


def rmse(p, y, ax=None):
    return np.sqrt(np.mean((p - y) ** 2, axis=ax))


def plot_forecast_blocks(model, n_split, t_all, target, clean, feats_n, target_t):
    seg = np.arange(n_split, n_split + 16 * OUTPUT_LEN, OUTPUT_LEN) - INPUT_LEN
    m_s, e_s, a_s, s_s = predict(model, seg, feats_n, target_t)
    ts = t_all[seg[:, None] + INPUT_LEN + AR_OUT]
    tt = ts.ravel()
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    ax[0].plot(tt, target[seg[:, None] + INPUT_LEN + AR_OUT].ravel(), ".", color=GREY, ms=2.5,
               label="measured ecg_mV")
    ax[0].plot(tt, clean[seg[:, None] + INPUT_LEN + AR_OUT].ravel(), color=INK, lw=1,
               label="true clean ECG")
    for k in range(len(seg)):
        ax[0].fill_between(ts[k], m_s[k] - 1.96 * s_s[k], m_s[k] + 1.96 * s_s[k],
                           color=BLUE, alpha=0.25, lw=0, label="95% band" if k == 0 else None)
        ax[0].plot(ts[k], m_s[k], color=BLUE, lw=1.4, label="forecast mean" if k == 0 else None)
        ax[0].axvline(ts[k, 0], color=GREY, lw=0.5, ls=":")
    ax[0].set(ylabel="ECG [mV]",
              title="Each 0.25 s block is forecast from the 1 s before it (dotted = forecast origin)")
    ax[0].legend(loc="upper right", frameon=False, ncol=4)
    ax[1].plot(tt, a_s.ravel(), color=BLUE, lw=1.3, label="aleatoric (learned noise)")
    ax[1].plot(tt, e_s.ravel(), color=ORANGE, lw=1.3, label="epistemic (weights)")
    for k in range(len(seg)):
        ax[1].axvline(ts[k, 0], color=GREY, lw=0.5, ls=":")
    ax[1].set(xlabel="time [s]", ylabel="std [mV]", title="Predictive uncertainty")
    ax[1].legend(loc="upper right", frameon=False)
    fig.savefig("bayes_lstm_forecast.png", dpi=150)


def plot_rmse_vs_lead(mean, persist, last_beat, y_clean):
    lead_ms = 1000 * (AR_OUT + 1) / FS
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(lead_ms, rmse(persist, y_clean, 0), color=ORANGE, lw=2, label="persistence")
    ax.plot(lead_ms, rmse(last_beat, y_clean, 0), color=AQUA, lw=2, label="repeat last beat")
    ax.plot(lead_ms, rmse(mean, y_clean, 0), color=BLUE, lw=2, label="Bayesian LSTM")
    ax.set(xlabel="lead time [ms]", ylabel="RMSE vs clean ECG [mV]",
           title="Forecast error vs lead time (test set)")
    ax.legend(frameon=False)
    fig.savefig("bayes_lstm_rmse_vs_lead.png", dpi=150)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bayesian LSTM forecast of the next 0.25 s of ECG from the last 1 s of measurable signals.")
    parser.add_argument("--csv", default="patients_basic/ecg_patient_06.csv",
                        help="patient CSV written by ecg_patients.m")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    df = pd.read_csv(args.csv)
    df = df.iloc[np.flatnonzero(df[INPUTS].notna().all(axis=1))[0]:].reset_index(drop=True)
    feats = df[INPUTS].to_numpy(np.float32)
    target = df[TARGET].to_numpy(np.float32)
    clean = df["ecg_clean_mV"].to_numpy(np.float32)
    t_all = df["time_s"].to_numpy()
    n = len(df)
    n_split = int(TRAIN_FRAC * n)

    mu_x, sd_x = feats[:n_split].mean(0), feats[:n_split].std(0)
    feats_n = torch.tensor((feats - mu_x) / sd_x)
    target_t = torch.tensor(target)

    train_starts = np.arange(0, n_split - INPUT_LEN - OUTPUT_LEN + 1)
    test_starts = np.arange(n_split - INPUT_LEN, n - INPUT_LEN - OUTPUT_LEN + 1, TEST_STRIDE)
    n_train = len(train_starts)
    print(f"X window [{INPUT_LEN} x {len(INPUTS)}] {INPUTS} -> Y [{OUTPUT_LEN}] {TARGET}")
    print(f"train windows {n_train} (targets {t_all[INPUT_LEN]:.1f}-{t_all[n_split - 1]:.1f} s), "
          f"test windows {len(test_starts)} (targets {t_all[n_split]:.1f}-{t_all[-1]:.1f} s)")

    model = BayesLSTM(len(INPUTS), OUTPUT_LEN)
    n_w = sum(p.numel() for name, p in model.named_parameters() if name.endswith("mu"))
    print(f"Bayesian LSTM ({N_LAYERS} layer(s) x {HIDDEN} units): {n_w} weights (x2 variational parameters)")

    train(model, train_starts, feats_n, target_t)

    model.eval()
    torch.save(model.state_dict(), "bayes_lstm_ecg.pt")
    mean, sd_epi, sd_ale, sd_tot = predict(model, test_starts, feats_n, target_t)
    idx_y = test_starts[:, None] + INPUT_LEN + AR_OUT
    y_noisy, y_clean = target[idx_y], clean[idx_y]

    last = test_starts + INPUT_LEN - 1
    persist = np.repeat(target[last][:, None], OUTPUT_LEN, axis=1)
    rr_samp = np.round(df["rr_last_s"].to_numpy()[last] * FS).astype(int)
    last_beat = target[idx_y - rr_samp[:, None]]

    print("\n--- test set (last 20 %, unseen), all 64 lead times ---")
    print(f"{'':22s}{'RMSE vs clean':>15s}{'RMSE vs noisy':>15s}")
    for name, p in [("Bayesian LSTM mean", mean), ("repeat last beat", last_beat), ("persistence", persist)]:
        print(f"{name:22s}{rmse(p, y_clean):13.4f} mV{rmse(p, y_noisy):12.4f} mV")
    print(f"skill vs repeat-last-beat : {1 - rmse(mean, y_clean) / rmse(last_beat, y_clean):.2f}")
    for lvl, zq in [(50, 0.674), (68, 0.994), (95, 1.960)]:
        cov = 100 * np.mean(np.abs(y_noisy - mean) <= zq * sd_tot)
        print(f"{lvl}% band coverage         : {cov:.1f} %")
    print(f"mean std  aleatoric {sd_ale.mean():.4f} mV, epistemic {sd_epi.mean():.4f} mV")
    np.savez("bayes_lstm_test.npz", mean=mean, sd_epi=sd_epi, sd_ale=sd_ale, sd_tot=sd_tot,
             y_noisy=y_noisy, y_clean=y_clean, persist=persist, last_beat=last_beat)

    pd.DataFrame({
        "window_id": np.repeat(np.arange(1, len(test_starts) + 1), OUTPUT_LEN),
        "forecast_origin_time_s": np.repeat(t_all[last], OUTPUT_LEN),
        "lead_step": np.tile(AR_OUT + 1, len(test_starts)),
        "lead_time_s": np.tile((AR_OUT + 1) / FS, len(test_starts)),
        "time_s": t_all[idx_y].ravel(),
        "actual_ecg_mV": y_noisy.ravel(),
        "clean_ecg_mV": y_clean.ravel(),
        "predicted_mean_mV": mean.ravel(),
        "epistemic_std_mV": sd_epi.ravel(),
        "aleatoric_std_mV": sd_ale.ravel(),
        "total_std_mV": sd_tot.ravel(),
        "lower_95_mV": (mean - 1.96 * sd_tot).ravel(),
        "upper_95_mV": (mean + 1.96 * sd_tot).ravel(),
    }).to_csv("bayes_lstm_test_predictions.csv", index=False)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    plot_forecast_blocks(model, n_split, t_all, target, clean, feats_n, target_t)
    plot_rmse_vs_lead(mean, persist, last_beat, y_clean)
    print("\nsaved: bayes_lstm_forecast.png, bayes_lstm_rmse_vs_lead.png, bayes_lstm_test.npz, "
          "bayes_lstm_test_predictions.csv")


if __name__ == "__main__":
    main()
