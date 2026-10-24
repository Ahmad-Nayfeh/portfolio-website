"""Bearing faults on CWRU: FFT features + logistic regression against a 1D CNN,
with less training data and more noise.

Run:  python experiment.py [data_dir]   (numpy, scipy, scikit-learn, torch, matplotlib)
Downloads 16 CWRU .mat files (about 70 MB) into data_dir on the first run.
Writes the figures to ../../public/images/blog/bearing-fft-vs-cnn/ and the numbers to results.json.
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
from scipy.signal import resample_poly
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import nn

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "public/images/blog/bearing-fft-vs-cnn"
OUT.mkdir(parents=True, exist_ok=True)
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "data"
DATA.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)

# --- 1. Data: drive-end accelerometer, 0.007" faults, motor loads 0-3 HP.
CLASSES = ["normal", "inner race", "ball", "outer race"]
FILES = {  # class -> file number per load (0, 1, 2, 3 HP)
    "normal": [97, 98, 99, 100],
    "inner race": [105, 106, 107, 108],
    "ball": [118, 119, 120, 121],
    "outer race": [130, 131, 132, 133],
}
WIN, HOP = 2048, 1024


def load(num):
    path = DATA / f"{num}.mat"
    if not path.exists():
        urllib.request.urlretrieve(f"https://engineering.case.edu/sites/default/files/{num}.mat", path)
    m = sio.loadmat(path)
    x = next(v for k, v in m.items() if k.endswith("DE_time")).ravel().astype(np.float32)
    # The normal recordings are sampled at 48 kHz, the fault recordings at 12 kHz.
    # Without this, a model can learn the sampling rate instead of the fault.
    if num in FILES["normal"]:
        x = resample_poly(x, 1, 4).astype(np.float32)
    return x


def windows(x):
    n = (x.size - WIN) // HOP + 1
    return np.stack([x[i * HOP: i * HOP + WIN] for i in range(n)])


def split(loads):
    X, y = [], []
    for c, name in enumerate(CLASSES):
        for load_hp in loads:
            w = windows(load(FILES[name][load_hp]))
            X.append(w); y.append(np.full(len(w), c))
    return np.concatenate(X), np.concatenate(y)


Xtr_pool, ytr_pool = split([0, 1])      # train on 0 and 1 HP
Xte, yte = split([2, 3])                # test on 2 and 3 HP, never seen in training
print("train pool", Xtr_pool.shape, np.bincount(ytr_pool), "test", Xte.shape, np.bincount(yte))


def add_noise(X, snr_db, rng):
    if snr_db is None:
        return X
    p = (X**2).mean(axis=1, keepdims=True)
    return X + rng.normal(size=X.shape).astype(np.float32) * np.sqrt(p / 10 ** (snr_db / 10))


# --- 2. Method A: FFT band energies + logistic regression.
HANN = np.hanning(WIN).astype(np.float32)


def fft_features(X, bands=64):
    spec = np.abs(np.fft.rfft(X * HANN, axis=1))[:, 1:1025]             # 0-6 kHz, 1024 bins
    return np.log1p(spec.reshape(len(X), bands, -1).mean(axis=2))       # 64 bands of ~94 Hz


def lowpass(X, hz=4000):
    """Keeps 0-4 kHz only. The normal recordings carry energy up to 6 kHz; the fault recordings were
    filtered by the recording equipment above about 5.5 kHz. That band says how a file was recorded,
    not what is wrong with the bearing, so the "0-4 kHz" methods never see it."""
    S = np.fft.rfft(X, axis=1)
    S[:, int(hz / 12000 * WIN):] = 0
    return np.fft.irfft(S, n=WIN, axis=1).astype(np.float32)


def fft_features_4k(X, bands=64):
    spec = np.abs(np.fft.rfft(X * HANN, axis=1))[:, 1:1 + 640]          # 0-3.75 kHz, 640 bins
    return np.log1p(spec.reshape(len(X), bands, -1).mean(axis=2))


def run_fft(Xtr, ytr, test_sets, feats=fft_features):
    scaler = StandardScaler().fit(feats(Xtr))
    clf = LogisticRegression(C=1.0, max_iter=3000).fit(scaler.transform(feats(Xtr)), ytr)
    out = {}
    for k, X in test_sets.items():
        pred = clf.predict(scaler.transform(feats(X)))
        out[k] = float((pred == yte).mean())
        out[k + "_pred_counts"] = np.bincount(pred, minlength=4).tolist()
    return out


# --- 3. Method B: a small 1D CNN on the raw window (wide first kernel, as in WDCNN).
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        def block(i, o, k, s):
            return nn.Sequential(nn.Conv1d(i, o, k, s, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU(), nn.MaxPool1d(2))
        self.f = nn.Sequential(block(1, 16, 64, 8), block(16, 32, 3, 1), block(32, 64, 3, 1), block(64, 64, 3, 1))
        self.head = nn.Linear(64, 4)

    def forward(self, x):
        return self.head(self.f(x).mean(dim=2))


def run_cnn(Xtr, ytr, test_sets, seed, steps=400):
    torch.manual_seed(seed)
    scale = float(Xtr.std())
    model = CNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    xt = torch.from_numpy(Xtr / scale).unsqueeze(1)
    yt = torch.from_numpy(ytr).long()
    g = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, len(xt), (min(32, len(xt)),), generator=g)
        loss = nn.functional.cross_entropy(model(xt[idx]), yt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    out = {}
    with torch.no_grad():
        for k, X in test_sets.items():
            pred = torch.cat([model(torch.from_numpy(X[i:i + 512] / scale).unsqueeze(1)).argmax(1) for i in range(0, len(X), 512)])
            out[k] = float((pred.numpy() == yte).mean())
            out[k + "_recall"] = [float((pred.numpy()[yte == c] == c).mean()) for c in range(4)]
    return out


# --- 4. The grid: windows per class in training x noise in the test set.
SIZES = [4, 8, 16, 32, 64, 160]          # 160: about all the normal windows there are
NOISE = [None, 10, 5, 0, -5]          # test SNR in dB (None = as recorded)
rng = np.random.default_rng(0)
test_sets = {str(s): add_noise(Xte, s, rng) for s in NOISE}
test_sets_4k = {k: lowpass(X) for k, X in test_sets.items()}

results = {"train_pool": np.bincount(ytr_pool).tolist(), "test": np.bincount(yte).tolist(), "runs": []}
t0 = time.time()
for size in SIZES:
    for seed in range(5 if size <= 32 else 3):
        r = np.random.default_rng(100 + seed)
        idx = np.concatenate([r.choice(np.nonzero(ytr_pool == c)[0], size, replace=False) for c in range(4)])
        Xtr, ytr = Xtr_pool[idx], ytr_pool[idx]
        results["runs"].append({
            "size": size, "seed": seed, "fft": run_fft(Xtr, ytr, test_sets), "fft4k": run_fft(Xtr, ytr, test_sets, fft_features_4k),
            "cnn": run_cnn(Xtr, ytr, test_sets, seed), "cnn4k": run_cnn(lowpass(Xtr), ytr, test_sets_4k, seed),
        })
    print(f"size {size} done, {time.time() - t0:.0f} s", flush=True)


def summary(method, size, noise):
    acc = [r[method][str(noise)] for r in results["runs"] if r["size"] == size]
    return float(np.mean(acc)), float(np.std(acc))


METHODS = ["fft", "fft4k", "cnn", "cnn4k"]
STYLE = [("fft", "tab:green", "-", "FFT bands + LR, 0-6 kHz"), ("fft4k", "tab:green", "--", "FFT bands + LR, 0-4 kHz"),
         ("cnn", "tab:purple", "-", "1D CNN, 0-6 kHz"), ("cnn4k", "tab:purple", "--", "1D CNN, 0-4 kHz")]
spec = np.abs(np.fft.rfft(Xte * HANN, axis=1)) ** 2
freqs = np.fft.rfftfreq(WIN, 1 / 12000)
results["energy_above_5500hz_pct_median"] = {CLASSES[c]: float(np.median((spec[:, freqs > 5500].sum(1) / spec.sum(1))[yte == c]) * 100) for c in range(4)}
share = results["energy_above_5500hz_pct_median"]
results["normal_vs_fault_energy_ratio_above_5500hz"] = {"min": share["normal"] / max(v for k, v in share.items() if k != "normal"),
                                                       "max": share["normal"] / min(v for k, v in share.items() if k != "normal")}
results["cnn_parameters"] = sum(p.numel() for p in CNN().parameters())


def top_peaks_hz(num, fs, lo=100, hi=200, k=3):
    """The strongest spectral peaks of a whole recording in [lo, hi] Hz, read at sampling rate fs.
    Used to check the normal files' true rate: their peaks match the fault files' only at 48 kHz."""
    m = sio.loadmat(DATA / f"{num}.mat")
    x = next(v for kk, v in m.items() if kk.endswith("DE_time")).ravel()
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    f = np.fft.rfftfreq(x.size, 1 / fs)
    band = (f > lo) & (f < hi)
    peaks = []
    for i in np.argsort(spec[band])[::-1]:
        fr = float(f[band][i])
        if all(abs(fr - q) > 3 for q in peaks):
            peaks.append(fr)
        if len(peaks) == k:
            break
    return sorted(round(q) for q in peaks)


results["peaks_hz"] = {"normal_97_read_at_48k": top_peaks_hz(97, 48000), "inner_105_read_at_12k": top_peaks_hz(105, 12000),
                       "normal_97_read_at_12k": top_peaks_hz(97, 12000)}
results["summary"] = {m: {str(n): {str(s): summary(m, s, n) for s in SIZES} for n in NOISE} for m in METHODS}
(HERE / "results.json").write_text(json.dumps(results, indent=2))

# --- 5. Figures.
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(8, 4))
for m, col, ls, lab in STYLE:
    mean = [summary(m, s, 0)[0] * 100 for s in SIZES]
    sd = [summary(m, s, 0)[1] * 100 for s in SIZES]
    ax.errorbar(SIZES, mean, yerr=sd, marker="o", capsize=3, color=col, ls=ls, label=lab)
ax.set_xscale("log", base=2); ax.set_xticks(SIZES); ax.set_xticklabels(SIZES)
ax.set_xlabel("training windows per class"); ax.set_ylabel("accuracy at 0 dB test noise (%)")
ax.axhline(25, color="0.7", ls=":", lw=1); ax.legend(frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "accuracy-vs-data.png", dpi=150); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
labels = ["clean", "10 dB", "5 dB", "0 dB", "-5 dB"]
for a, size in zip(axs, [16, 160]):
    for m, col, ls, lab in STYLE:
        a.plot(labels, [summary(m, size, n)[0] * 100 for n in NOISE], marker="o", ls=ls, color=col, label=lab)
    a.set_title(f"{size} training windows per class"); a.set_xlabel("noise added to the test signal (SNR)")
    a.axhline(25, color="0.7", ls=":", lw=1)
axs[0].set_ylabel("test accuracy (%)"); axs[1].legend(frameon=False, fontsize=8.5, loc="center left")
fig.tight_layout(); fig.savefig(OUT / "accuracy-vs-noise.png", dpi=150); plt.close(fig)

# One window of each class, and its spectrum: what the FFT features see.
fig, axs = plt.subplots(4, 2, figsize=(9, 6.5), sharex="col")
f = np.fft.rfftfreq(WIN, 1 / 12000)
for c in range(4):
    w = Xte[np.nonzero(yte == c)[0][10]]
    axs[c, 0].plot(np.arange(WIN) / 12, w, lw=0.6, color="0.25"); axs[c, 0].set_ylabel(CLASSES[c], rotation=0, ha="right")
    axs[c, 1].semilogy(f, np.abs(np.fft.rfft(w * HANN)) + 1e-3, lw=0.6, color="tab:green")
axs[3, 0].set_xlabel("time (ms)"); axs[3, 1].set_xlabel("frequency (Hz)")
axs[0, 0].set_title("raw vibration window"); axs[0, 1].set_title("its spectrum (log scale)")
fig.tight_layout(); fig.savefig(OUT / "classes.png", dpi=150); plt.close(fig)

print(json.dumps(results["summary"], indent=1))
