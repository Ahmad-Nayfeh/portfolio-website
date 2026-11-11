import os
import json
import numpy as np
import scipy.ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

# Set fixed random seeds for reproducibility
np.random.seed(42)

OUT_DIR = os.environ.get("OUT_DIR", ".")
os.makedirs(OUT_DIR, exist_ok=True)

def generate_synthetic_image(seed, size=128):
    """Generates a synthetic image with structured gradients, shapes, and textures."""
    rng = np.random.RandomState(seed)
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)
    
    # Smooth background gradient
    img = 0.5 + 0.3 * np.sin(2 * np.pi * rng.uniform(0.5, 2.0) * xx + rng.uniform(0, 2*np.pi))
    img += 0.2 * np.cos(2 * np.pi * rng.uniform(0.5, 2.0) * yy)
    
    # Geometric objects
    num_shapes = rng.randint(2, 5)
    for _ in range(num_shapes):
        cx, cy = rng.uniform(-0.6, 0.6, size=2)
        rad = rng.uniform(0.15, 0.35)
        val = rng.uniform(-0.25, 0.25)
        mask = ((xx - cx)**2 + (yy - cy)**2) < rad**2
        img[mask] += val
        
    # Fine texture noise
    texture = scipy.ndimage.gaussian_filter(rng.randn(size, size), sigma=1.0)
    img += 0.05 * texture
    
    return np.clip(img, 0.0, 1.0)

def compute_ssim(img1, img2):
    """Computes Structural Similarity Index (SSIM) between two single-channel images."""
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2
    mu1 = scipy.ndimage.gaussian_filter(img1, 1.5)
    mu2 = scipy.ndimage.gaussian_filter(img2, 1.5)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = scipy.ndimage.gaussian_filter(img1 ** 2, 1.5) - mu1_sq
    sigma2_sq = scipy.ndimage.gaussian_filter(img2 ** 2, 1.5) - mu2_sq
    sigma12 = scipy.ndimage.gaussian_filter(img1 * img2, 1.5) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(np.mean(ssim_map))

def compute_psnr(img1, img2):
    """Computes Peak Signal-to-Noise Ratio (PSNR) in dB."""
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-12:
        return 100.0
    return float(10.0 * np.log10(1.0 / mse))

# Create frequency masks
size = 128
u = np.fft.fftfreq(size)
v = np.fft.fftfreq(size)
uu, vv = np.meshgrid(np.fft.fftshift(u), np.fft.fftshift(v))
radius = np.sqrt(uu**2 + vv**2)

# Low-pass mask (SpectralCanary), high-pass mask, and mid-pass mask
sigma_lp = 0.10
sigma_hp = 0.15
mask_lp = np.exp(- (radius**2) / (2 * sigma_lp**2))
mask_hp = 1.0 - np.exp(- (radius**2) / (2 * sigma_hp**2))

# Generate fixed canary pattern C
canary_rng = np.random.RandomState(999)
canary_spatial = canary_rng.randn(size, size)
canary_spatial = scipy.ndimage.gaussian_filter(canary_spatial, sigma=0.8)
canary_spatial = canary_spatial - np.mean(canary_spatial)
canary_spatial = canary_spatial / np.std(canary_spatial)
F_canary = np.fft.fftshift(np.fft.fft2(canary_spatial))

def embed_lowpass(img, alpha):
    F_img = np.fft.fftshift(np.fft.fft2(img))
    F_marked = F_img * (1.0 - alpha * mask_lp) + (alpha * mask_lp * F_canary)
    marked = np.real(np.fft.ifft2(np.fft.ifftshift(F_marked)))
    return np.clip(marked, 0.0, 1.0)

def embed_spatial(img, beta):
    marked = img + beta * canary_spatial
    return np.clip(marked, 0.0, 1.0)

def embed_highpass(img, gamma):
    F_img = np.fft.fftshift(np.fft.fft2(img))
    F_marked = F_img * (1.0 - gamma * mask_hp) + (gamma * mask_hp * F_canary)
    marked = np.real(np.fft.ifft2(np.fft.ifftshift(F_marked)))
    return np.clip(marked, 0.0, 1.0)

# Calibrate embedding strengths for target PSNR ~ 38.0 dB
target_psnr = 38.0
test_sample = generate_synthetic_image(0, size=size)

def binary_search_strength(embed_fn, low=0.0001, high=2.0, target=38.0):
    for _ in range(20):
        mid = (low + high) / 2.0
        m = embed_fn(test_sample, mid)
        p = compute_psnr(test_sample, m)
        if p > target:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0

alpha_opt = binary_search_strength(embed_lowpass, 0.001, 1.5, target_psnr)
beta_opt = binary_search_strength(embed_spatial, 0.001, 0.5, target_psnr)
gamma_opt = binary_search_strength(embed_highpass, 0.001, 1.5, target_psnr)

# Detectors (Normalized Matched Filter / Correlation)
def detect_lowpass(img):
    F_img = np.fft.fftshift(np.fft.fft2(img))
    sub_y = F_img * mask_lp
    sub_c = F_canary * mask_lp
    y_sp = np.real(np.fft.ifft2(np.fft.ifftshift(sub_y)))
    c_sp = np.real(np.fft.ifft2(np.fft.ifftshift(sub_c)))
    y_flat = y_sp.flatten() - np.mean(y_sp)
    c_flat = c_sp.flatten() - np.mean(c_sp)
    norm = (np.linalg.norm(y_flat) * np.linalg.norm(c_flat) + 1e-12)
    return float(np.dot(y_flat, c_flat) / norm)

def detect_spatial(img):
    y_flat = img.flatten() - np.mean(img)
    c_flat = canary_spatial.flatten() - np.mean(canary_spatial)
    norm = (np.linalg.norm(y_flat) * np.linalg.norm(c_flat) + 1e-12)
    return float(np.dot(y_flat, c_flat) / norm)

def detect_highpass(img):
    F_img = np.fft.fftshift(np.fft.fft2(img))
    sub_y = F_img * mask_hp
    sub_c = F_canary * mask_hp
    y_sp = np.real(np.fft.ifft2(np.fft.ifftshift(sub_y)))
    c_sp = np.real(np.fft.ifft2(np.fft.ifftshift(sub_c)))
    y_flat = y_sp.flatten() - np.mean(y_sp)
    c_flat = c_sp.flatten() - np.mean(c_sp)
    norm = (np.linalg.norm(y_flat) * np.linalg.norm(c_flat) + 1e-12)
    return float(np.dot(y_flat, c_flat) / norm)

# Generate dataset: 100 positive (marked), 100 negative (control)
num_samples = 100
originals = [generate_synthetic_image(i + 100, size=size) for i in range(num_samples)]
controls = [generate_synthetic_image(i + 2000, size=size) for i in range(num_samples)]

marked_lp = [embed_lowpass(img, alpha_opt) for img in originals]
marked_sp = [embed_spatial(img, beta_opt) for img in originals]
marked_hp = [embed_highpass(img, gamma_opt) for img in originals]

# Compute baseline PSNR / SSIM averages
psnr_lp_val = float(np.mean([compute_psnr(o, m) for o, m in zip(originals, marked_lp)]))
psnr_sp_val = float(np.mean([compute_psnr(o, m) for o, m in zip(originals, marked_sp)]))
psnr_hp_val = float(np.mean([compute_psnr(o, m) for o, m in zip(originals, marked_hp)]))

ssim_lp_val = float(np.mean([compute_ssim(o, m) for o, m in zip(originals, marked_lp)]))
ssim_sp_val = float(np.mean([compute_ssim(o, m) for o, m in zip(originals, marked_sp)]))
ssim_hp_val = float(np.mean([compute_ssim(o, m) for o, m in zip(originals, marked_hp)]))

# Perturbation Attack Functions
def apply_blur(img, sigma=2.0):
    return scipy.ndimage.gaussian_filter(img, sigma=sigma)

def apply_noise(img, sigma=0.12):
    rng = np.random.RandomState(123)
    return np.clip(img + rng.randn(*img.shape) * sigma, 0.0, 1.0)

def apply_resample(img, factor=0.5):
    h, w = img.shape
    small = scipy.ndimage.zoom(img, factor, order=1)
    back = scipy.ndimage.zoom(small, 1.0 / factor, order=1)
    return np.clip(back[:h, :w], 0.0, 1.0)

regimes = {
    "Clean": lambda img: img,
    "Gaussian Blur (s=2.0)": lambda img: apply_blur(img, sigma=2.0),
    "Gaussian Noise (s=0.12)": lambda img: apply_noise(img, sigma=0.12),
    "Resampling (2x Down/Up)": lambda img: apply_resample(img, factor=0.5)
}

auc_results = {"Low-Pass (SpectralCanary)": {}, "Spatial Additive": {}, "High-Pass Spectral": {}}
roc_data = {}

for regime_name, attack_fn in regimes.items():
    for method_name, marked_list, detect_fn in [
        ("Low-Pass (SpectralCanary)", marked_lp, detect_lowpass),
        ("Spatial Additive", marked_sp, detect_spatial),
        ("High-Pass Spectral", marked_hp, detect_highpass)
    ]:
        scores_pos = [detect_fn(attack_fn(img)) for img in marked_list]
        scores_neg = [detect_fn(attack_fn(img)) for img in controls]
        
        y_true = np.array([1] * num_samples + [0] * num_samples)
        y_scores = np.array(scores_pos + scores_neg)
        
        auc = float(roc_auc_score(y_true, y_scores))
        auc_results[method_name][regime_name] = round(auc, 4)
        
        if regime_name == "Gaussian Blur (s=2.0)":
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_data[method_name] = (fpr, tpr, auc)

# Figure 1: Visual and Spectral Comparison
fig, axes = plt.subplots(3, 4, figsize=(12, 9))
sample_orig = originals[0]
methods_vis = [
    ("Original / Reference", sample_orig),
    ("Low-Pass (SpectralCanary)", marked_lp[0]),
    ("Spatial Additive", marked_sp[0]),
    ("High-Pass Spectral", marked_hp[0])
]

for col_idx, (title, img) in enumerate(methods_vis):
    # Row 0: Spatial Image
    axes[0, col_idx].imshow(img, cmap='gray', vmin=0, vmax=1)
    axes[0, col_idx].set_title(title, fontsize=10, fontweight='bold')
    axes[0, col_idx].axis('off')
    
    # Row 1: Fourier Magnitude Spectrum
    F_mag = np.log(1 + np.abs(np.fft.fftshift(np.fft.fft2(img))))
    axes[1, col_idx].imshow(F_mag, cmap='magma')
    axes[1, col_idx].set_title(f"{title}\nLog Magnitude Spectrum", fontsize=9)
    axes[1, col_idx].axis('off')
    
    # Row 2: Residual Difference Map vs Original
    diff = np.abs(img - sample_orig)
    im_diff = axes[2, col_idx].imshow(diff, cmap='inferno', vmin=0, vmax=0.15)
    axes[2, col_idx].set_title(f"Abs Residual (|I_m - I_0|)", fontsize=9)
    axes[2, col_idx].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "watermark_fidelity_vis.png"), dpi=200)
plt.close()

# Figure 2: ROC Curves under Gaussian Blur Attack
plt.figure(figsize=(7, 6))
for method_name, (fpr, tpr, auc) in roc_data.items():
    plt.plot(fpr, tpr, lw=2, label=f"{method_name} (AUC = {auc:.4f})")
plt.plot([0, 1], [0, 1], 'k--', lw=1, label="Random Chance (AUC = 0.5000)")
plt.xlabel("False Positive Rate (FPR)", fontsize=11)
plt.ylabel("True Positive Rate (TPR)", fontsize=11)
plt.title("Canary Detection ROC Curve under Gaussian Blur (sigma = 2.0)", fontsize=11, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower right', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "detection_roc_curves.png"), dpi=200)
plt.close()

# Figure 3: Robustness Comparison Across Perturbation Regimes
plt.figure(figsize=(9, 5.5))
regime_keys = list(regimes.keys())
x_indices = np.arange(len(regime_keys))
bar_width = 0.25

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for i, (method_name, color) in enumerate(zip(auc_results.keys(), colors)):
    aucs = [auc_results[method_name][rk] for rk in regime_keys]
    plt.bar(x_indices + i * bar_width, aucs, bar_width, label=method_name, color=color, alpha=0.85, edgecolor='black')

plt.xlabel("Perturbation Attack Regime", fontsize=11)
plt.ylabel("Detection ROC-AUC", fontsize=11)
plt.title("Canary Detection Robustness Across Image Perturbations", fontsize=12, fontweight='bold')
plt.xticks(x_indices + bar_width, regime_keys, rotation=15, fontsize=10)
plt.ylim(0.4, 1.05)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend(loc='lower left', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "robustness_across_perturbations.png"), dpi=200)
plt.close()

# Prepare and output results.json
results = {
    "psnr_lowpass": psnr_lp_val,
    "psnr_spatial": psnr_sp_val,
    "psnr_highpass": psnr_hp_val,
    "ssim_lowpass": ssim_lp_val,
    "ssim_spatial": ssim_sp_val,
    "ssim_highpass": ssim_hp_val,
    "auc_clean_lowpass": auc_results["Low-Pass (SpectralCanary)"]["Clean"],
    "auc_clean_spatial": auc_results["Spatial Additive"]["Clean"],
    "auc_clean_highpass": auc_results["High-Pass Spectral"]["Clean"],
    "auc_blur_lowpass": auc_results["Low-Pass (SpectralCanary)"]["Gaussian Blur (s=2.0)"],
    "auc_blur_spatial": auc_results["Spatial Additive"]["Gaussian Blur (s=2.0)"],
    "auc_blur_highpass": auc_results["High-Pass Spectral"]["Gaussian Blur (s=2.0)"],
    "auc_noise_lowpass": auc_results["Low-Pass (SpectralCanary)"]["Gaussian Noise (s=0.12)"],
    "auc_noise_spatial": auc_results["Spatial Additive"]["Gaussian Noise (s=0.12)"],
    "auc_noise_highpass": auc_results["High-Pass Spectral"]["Gaussian Noise (s=0.12)"],
    "auc_resample_lowpass": auc_results["Low-Pass (SpectralCanary)"]["Resampling (2x Down/Up)"],
    "auc_resample_spatial": auc_results["Spatial Additive"]["Resampling (2x Down/Up)"],
    "auc_resample_highpass": auc_results["High-Pass Spectral"]["Resampling (2x Down/Up)"]
}

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Done. Results written to results.json and figures saved to OUT_DIR.")
