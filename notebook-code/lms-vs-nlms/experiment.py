"""LMS against NLMS for removing mains interference from a real ECG.

Run:  python experiment.py   (numpy, scipy, matplotlib, pooch)
Writes the figures to ../../public/images/blog/lms-vs-nlms/ and the numbers to results.json.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import datasets, signal

OUT = Path(__file__).resolve().parents[2] / "public/images/blog/lms-vs-nlms"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)

# --- 1. The data: 60 s of a real ECG (MIT-BIH record 208, 360 Hz), in millivolts.
fs = 360
ecg = datasets.electrocardiogram()[: 60 * fs]
ecg = ecg - np.median(ecg)
t = np.arange(ecg.size) / fs

# --- 2. The interference: 60 Hz mains plus its 3rd harmonic, with a slowly drifting phase.
# At t = 30 s it becomes four times stronger (a motor or a charger switched on nearby).
phase = np.cumsum(rng.normal(0, 0.002, ecg.size))           # slow random phase drift
amp = np.where(t < 30, 0.5, 2.0)                             # mV
mains = amp * (np.sin(2 * np.pi * 60 * t + phase) + 0.3 * np.sin(2 * np.pi * 180 * t + 3 * phase))
primary = ecg + mains                                        # what the ECG electrode records

# The reference pickup sees the same source, with its own gain and phase shift, and no ECG.
ref = 1.7 * amp * (np.sin(2 * np.pi * 60 * t + phase + 0.9) + 0.3 * np.sin(2 * np.pi * 180 * t + 3 * phase + 2.1))
ref = ref + rng.normal(0, 0.01, ecg.size)                    # a little sensor noise


def lms(x, d, M, mu, normalized=False, eps=1e-6):
    """Adaptive FIR filter: predicts the interference in d from the reference x; returns the error e = d - y."""
    w = np.zeros(M)
    buf = np.zeros(M)
    e = np.empty_like(d)
    for n in range(d.size):
        buf = np.roll(buf, 1)
        buf[0] = x[n]
        y = w @ buf
        e[n] = d[n] - y
        step = mu / (eps + buf @ buf) if normalized else mu
        w = w + step * e[n] * buf
        if not np.isfinite(w).all():                         # diverged: report it instead of crashing
            e[n:] = np.nan
            break
    return e


def snr_db(clean, est):
    err = est - clean
    if not np.isfinite(err).all():
        return float("nan")
    return 10 * np.log10(np.sum(clean**2) / np.sum(err**2))


def snr_track(clean, est, win=fs):
    return np.array([snr_db(clean[i:i + win], est[i:i + win]) for i in range(0, clean.size - win + 1, win)])


M = 8
first, second = slice(0, 30 * fs), slice(30 * fs, 60 * fs)
results = {"input_snr_db": {"first_half": snr_db(ecg[first], primary[first]), "second_half": snr_db(ecg[second], primary[second])}}

# --- 3. Step-size sweep: SNR in each half for both algorithms.
lms_mus = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
nlms_mus = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
sweep = {"lms": [], "nlms": []}
for name, mus, norm in [("lms", lms_mus, False), ("nlms", nlms_mus, True)]:
    for mu in mus:
        e = lms(ref, primary, M, mu, normalized=norm)
        sweep[name].append({"mu": mu, "first_half": snr_db(ecg[first], e[first]), "second_half": snr_db(ecg[second], e[second])})
results["sweep"] = sweep

# The fair pick: the best step size judged on the first 30 s only (all an engineer would have
# before the interference changed), then we watch what happens after t = 30 s.
best_lms = max(sweep["lms"], key=lambda r: r["first_half"] if np.isfinite(r["first_half"]) else -1e9)["mu"]
best_nlms = max(sweep["nlms"], key=lambda r: r["first_half"] if np.isfinite(r["first_half"]) else -1e9)["mu"]
e_lms = lms(ref, primary, M, best_lms)
e_nlms = lms(ref, primary, M, best_nlms, normalized=True)

# Baseline: fixed notch filters at 60 and 180 Hz (causal, as a real-time device would run them).
b1, a1 = signal.iirnotch(60, 30, fs)
b3, a3 = signal.iirnotch(180, 30, fs)
e_notch = signal.lfilter(b3, a3, signal.lfilter(b1, a1, primary))

tracks = {k: snr_track(ecg, v) for k, v in {"LMS": e_lms, "NLMS": e_nlms, "Notch": e_notch, "No filter": primary}.items()}
results["picked"] = {"lms_mu": best_lms, "nlms_mu": best_nlms}
for k, est in {"lms": e_lms, "nlms": e_nlms, "notch": e_notch}.items():
    results[k] = {
        "first_half_db": snr_db(ecg[first], est[first]),
        "second_half_db": snr_db(ecg[second], est[second]),
    }
# The notch also removes the ECG's own content near 60 and 180 Hz: how much, on the clean ECG alone?
clean_through_notch = signal.lfilter(b3, a3, signal.lfilter(b1, a1, ecg))
results["notch"]["distortion_snr_on_clean_ecg_db"] = snr_db(ecg, clean_through_notch)
for k in ["lms", "nlms", "notch"]:
    results[k]["improvement_second_half_db"] = results[k]["second_half_db"] - results["input_snr_db"]["second_half"]
results["nlms_minus_lms_second_half_db"] = results["nlms"]["second_half_db"] - results["lms"]["second_half_db"]

(Path(__file__).parent / "results.json").write_text(json.dumps(results, indent=2))

# --- 4. Figures.
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
z = slice(int(29.0 * fs), int(32.0 * fs))
fig, ax = plt.subplots(3, 1, figsize=(9, 6.5), sharex=True)
ax[0].plot(t[z], primary[z], lw=0.8, color="0.4"); ax[0].set_title("Recorded: ECG + mains (interference x4 at t = 30 s)")
ax[1].plot(t[z], e_lms[z], lw=0.8, color="tab:red"); ax[1].set_title(f"LMS output (mu = {best_lms})")
ax[2].plot(t[z], e_nlms[z], lw=0.8, color="tab:blue"); ax[2].set_title(f"NLMS output (mu = {best_nlms})")
for a in ax:
    a.plot(t[z], ecg[z], lw=0.6, color="k", alpha=0.35)
    a.axvline(30, color="0.6", ls="--", lw=0.8); a.set_ylabel("mV")
ax[2].set_xlabel("time (s)")
fig.tight_layout(); fig.savefig(OUT / "waveforms.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 3.8))
sec = np.arange(len(tracks["LMS"])) + 0.5
for k, c in [("No filter", "0.6"), ("Notch", "tab:green"), ("LMS", "tab:red"), ("NLMS", "tab:blue")]:
    ax.plot(sec, tracks[k], label=k, color=c, lw=1.6)
ax.axvline(30, color="0.6", ls="--", lw=0.8)
ax.set_xlabel("time (s)"); ax.set_ylabel("output SNR per second (dB)"); ax.legend(ncol=4, frameon=False)
fig.tight_layout(); fig.savefig(OUT / "snr-over-time.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for a, name, col in [(ax[0], "lms", "tab:red"), (ax[1], "nlms", "tab:blue")]:
    mus = [r["mu"] for r in sweep[name]]
    a.semilogx(mus, [r["first_half"] for r in sweep[name]], "o-", color=col, label="first 30 s")
    a.semilogx(mus, [r["second_half"] for r in sweep[name]], "s--", color=col, alpha=0.6, label="last 30 s (x4)")
    a.set_title(name.upper()); a.set_xlabel("step size mu")
ax[0].set_ylabel("output SNR (dB)"); ax[0].legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT / "step-size-sweep.png", dpi=150); plt.close(fig)

print(json.dumps(results, indent=2))
