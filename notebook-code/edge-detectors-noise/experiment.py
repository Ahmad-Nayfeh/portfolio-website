"""Sobel, Scharr and Canny under increasing noise, scored against edges we know exactly.

Run:  python experiment.py   (numpy, opencv-python-headless, matplotlib)
Writes the figures to ../../public/images/blog/edge-detectors-noise/ and the numbers to results.json.
"""
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "public/images/blog/edge-detectors-noise"
OUT.mkdir(parents=True, exist_ok=True)
SIZE = 256
TOL = 2          # a detected edge pixel counts if it is within 2 px of a true edge


def scene(seed):
    """Random filled shapes (circles, rectangles, triangles) at random grey levels on a gradient background.
    Rendered at 4x and downsampled, so the edges are anti-aliased like in a real photo.
    Returns the image (0-255 float) and the true edge map (boundaries between regions)."""
    r = np.random.default_rng(seed)
    big = SIZE * 4
    label = np.zeros((big, big), np.int32)
    img = np.tile(np.linspace(70, 130, big, dtype=np.float32), (big, 1))   # smooth background: no edges
    for k in range(1, 9):
        level = float(r.uniform(20, 235))
        mask = np.zeros((big, big), np.uint8)
        kind = r.integers(3)
        if kind == 0:
            cv2.circle(mask, tuple(int(v) for v in r.integers(150, big - 150, 2)), int(r.integers(60, 200)), 1, -1)
        elif kind == 1:
            p = r.integers(80, big - 80, 4)
            cv2.rectangle(mask, (int(min(p[0], p[1])), int(min(p[2], p[3]))), (int(max(p[0], p[1])), int(max(p[2], p[3]))), 1, -1)
        else:
            cv2.fillPoly(mask, [r.integers(60, big - 60, (3, 2)).astype(np.int32)], 1)
        img[mask == 1] = level
        label[mask == 1] = k
    img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    lab = label[::4, ::4]
    edge = np.zeros_like(lab, bool)
    edge[:-1, :] |= lab[:-1, :] != lab[1:, :]
    edge[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    return img, edge


def grad_mag(img, op, blur):
    x = cv2.GaussianBlur(img, (0, 0), blur) if blur else img
    if op == "sobel":
        gx, gy = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
    else:
        gx, gy = cv2.Scharr(x, cv2.CV_32F, 1, 0), cv2.Scharr(x, cv2.CV_32F, 0, 1)
    m = np.hypot(gx, gy)
    return m / (m.max() + 1e-9)


def f1(pred, truth):
    """Precision: share of detected pixels near a true edge. Recall: share of true edge pixels near a detection."""
    if pred.sum() == 0:
        return 0.0, 0.0, 0.0
    near_truth = cv2.distanceTransform((~truth).astype(np.uint8), cv2.DIST_L2, 3) <= TOL
    near_pred = cv2.distanceTransform((~pred).astype(np.uint8), cv2.DIST_L2, 3) <= TOL
    p = (pred & near_truth).sum() / pred.sum()
    rcl = (truth & near_pred).sum() / truth.sum()
    return (0.0 if p + rcl == 0 else 2 * p * rcl / (p + rcl)), p, rcl


METHODS = {
    "Sobel": lambda img, t: grad_mag(img, "sobel", 0) > t,
    "Scharr": lambda img, t: grad_mag(img, "scharr", 0) > t,
    "Gaussian blur + Sobel": lambda img, t: grad_mag(img, "sobel", 1.5) > t,
    "Canny": lambda img, t: cv2.Canny(cv2.GaussianBlur(img, (0, 0), 1.5).clip(0, 255).astype(np.uint8), 255 * t * 0.5, 255 * t, L2gradient=True) > 0,
}
THRESH = np.linspace(0.02, 0.9, 45)
NOISE = [0, 5, 10, 20, 30, 45, 60]          # std of Gaussian noise, grey levels (image range 0-255)


def scores(method, imgs, edges, t):
    return [f1(METHODS[method](im, t), e) for im, e in zip(imgs, edges)]


rng = np.random.default_rng(1)
tune = [scene(s) for s in range(100, 110)]     # threshold chosen on these 10 scenes
test = [scene(s) for s in range(200, 230)]     # scored on these 30 different ones
results = {"tolerance_px": TOL, "noise": NOISE, "methods": {}}
for name in METHODS:
    rows = []
    for sd in NOISE:
        noisy = lambda img: img + rng.normal(0, sd, img.shape).astype(np.float32)
        tune_imgs = [noisy(i) for i, _ in tune]
        best_t = max(THRESH, key=lambda t: np.mean([s[0] for s in scores(name, tune_imgs, [e for _, e in tune], t)]))
        test_imgs = [noisy(i) for i, _ in test]
        s = np.array(scores(name, test_imgs, [e for _, e in test], best_t))
        rows.append({"noise_sd": sd, "threshold": float(best_t), "f1": float(s[:, 0].mean()), "f1_sd": float(s[:, 0].std()),
                     "precision": float(s[:, 1].mean()), "recall": float(s[:, 2].mean())})
    results["methods"][name] = rows
    print(name, [round(r["f1"], 3) for r in rows], flush=True)

# How different are Sobel and Scharr really? Their responses to the same clean scene.
img0, _ = test[0]
a, b = grad_mag(img0, "sobel", 0), grad_mag(img0, "scharr", 0)
results["sobel_vs_scharr_magnitude_correlation"] = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
f1s = {m: [r["f1"] for r in rows] for m, rows in results["methods"].items()}
results["max_gap_sobel_vs_scharr"] = float(max(abs(a - b) for a, b in zip(f1s["Sobel"], f1s["Scharr"])))
results["max_gain_canny_vs_sobel"] = float(max(b - a for a, b in zip(f1s["Sobel"], f1s["Canny"])))
(HERE / "results.json").write_text(json.dumps(results, indent=2))

# --- Figures.
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(8, 4))
for name, col, ls in [("Sobel", "tab:red", "-"), ("Scharr", "tab:orange", "--"), ("Gaussian blur + Sobel", "tab:blue", "-"), ("Canny", "tab:green", "-")]:
    rows = results["methods"][name]
    ax.errorbar(NOISE, [r["f1"] for r in rows], yerr=[r["f1_sd"] for r in rows], marker="o", capsize=3, color=col, ls=ls, label=name)
ax.set_xlabel("noise standard deviation (grey levels, out of 255)"); ax.set_ylabel(f"edge F1 (within {TOL} px)")
ax.set_ylim(0, 1); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT / "f1-vs-noise.png", dpi=150); plt.close(fig)

img, edge = test[3]
noisy = img + np.random.default_rng(7).normal(0, 30, img.shape).astype(np.float32)
fig, axs = plt.subplots(2, 3, figsize=(9, 6.2))
panels = [("clean image", img, "gray"), ("true edges", edge, "gray_r"), ("noise sd = 30", noisy, "gray")]
for name in ["Sobel", "Gaussian blur + Sobel", "Canny"]:
    t = next(r["threshold"] for r in results["methods"][name] if r["noise_sd"] == 30)
    panels.append((name, METHODS[name](noisy, t), "gray_r"))
for a, (title, im, cmap) in zip(axs.ravel(), panels):
    a.imshow(im, cmap=cmap, vmin=0, vmax=255 if im.dtype != bool else 1); a.set_title(title); a.axis("off")
fig.tight_layout(); fig.savefig(OUT / "examples.png", dpi=150); plt.close(fig)
print(json.dumps({k: results[k] for k in ["sobel_vs_scharr_magnitude_correlation"]}))
