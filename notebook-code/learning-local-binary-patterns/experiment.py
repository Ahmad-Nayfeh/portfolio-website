import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

out_dir = os.environ.get('OUT_DIR', '.')
os.makedirs(out_dir, exist_ok=True)

def generate_synthetic_textures(n_per_class=100, size=48):
    H, W = size, size
    y, x = np.mgrid[0:H, 0:W]
    images = []
    labels = []
    
    for c in range(6):
        for _ in range(n_per_class):
            noise = np.random.normal(0, 0.08, (H, W))
            if c == 0:
                freq = np.random.uniform(3.0, 5.0)
                phase = np.random.uniform(0, 2 * np.pi)
                img = np.sin(2 * np.pi * freq * y / H + phase) + noise
            elif c == 1:
                freq = np.random.uniform(3.0, 5.0)
                phase = np.random.uniform(0, 2 * np.pi)
                img = np.sin(2 * np.pi * freq * x / W + phase) + noise
            elif c == 2:
                freq = np.random.uniform(3.0, 5.0)
                phase = np.random.uniform(0, 2 * np.pi)
                img = np.sin(2 * np.pi * freq * (x + y) / (np.sqrt(2) * H) + phase) + noise
            elif c == 3:
                fx = np.random.uniform(2.5, 4.0)
                fy = np.random.uniform(2.5, 4.0)
                img = np.sign(np.sin(2 * np.pi * fx * x / W)) * np.sign(np.sin(2 * np.pi * fy * y / H)) + noise
            elif c == 4:
                img = np.zeros((H, W))
                n_spots = np.random.randint(15, 25)
                for _ in range(n_spots):
                    cx, cy = np.random.uniform(4, W - 4), np.random.uniform(4, H - 4)
                    sig_x, sig_y = np.random.uniform(2.0, 4.0), np.random.uniform(1.0, 2.5)
                    blob = np.exp(-(((x - cx) ** 2) / (2 * sig_x ** 2) + ((y - cy) ** 2) / (2 * sig_y ** 2)))
                    img += blob
                img += noise
            elif c == 5:
                kx, ky = np.meshgrid(np.fft.fftfreq(W), np.fft.fftfreq(H))
                dist = np.sqrt(kx ** 2 + ky ** 2)
                dist[0, 0] = 1.0
                white = np.random.normal(0, 1, (H, W))
                f_white = np.fft.fft2(white)
                f_colored = f_white / (dist ** 0.8)
                f_colored[0, 0] = 0.0
                img = np.real(np.fft.ifft2(f_colored)) + noise
            
            img = (img - np.mean(img)) / (np.std(img) + 1e-6)
            images.append(img)
            labels.append(c)
            
    images = np.array(images, dtype=np.float32)[:, None, :, :]
    labels = np.array(labels, dtype=np.int64)
    
    indices = np.arange(len(labels))
    np.random.shuffle(indices)
    train_idx = indices[:360]
    test_idx = indices[360:]
    
    return (
        torch.tensor(images[train_idx]), torch.tensor(labels[train_idx]),
        torch.tensor(images[test_idx]), torch.tensor(labels[test_idx])
    )

X_train, y_train, X_test, y_test = generate_synthetic_textures(n_per_class=100, size=48)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)

class FlexLBPModule(nn.Module):
    def __init__(self, init_radius=1.5, learn_offsets=True, learn_thresholds=True):
        super().__init__()
        angles = [2.0 * np.pi * p / 8.0 for p in range(8)]
        init_offsets = [[init_radius * np.cos(a), init_radius * np.sin(a)] for a in angles]
        
        self.learn_offsets = learn_offsets
        self.learn_thresholds = learn_thresholds
        
        if learn_offsets:
            self.offsets = nn.Parameter(torch.tensor(init_offsets, dtype=torch.float32))
        else:
            self.register_buffer('offsets', torch.tensor(init_offsets, dtype=torch.float32))
            
        if learn_thresholds:
            self.thresholds = nn.Parameter(torch.zeros(8, dtype=torch.float32))
        else:
            self.register_buffer('thresholds', torch.zeros(8, dtype=torch.float32))
            
    def forward(self, x, beta=10.0):
        B, C, H, W = x.shape
        H_out, W_out = H - 8, W - 8
        
        base_y, base_x = torch.meshgrid(
            torch.linspace(-1.0 + 8.0 / H, 1.0 - 8.0 / H, H_out, device=x.device),
            torch.linspace(-1.0 + 8.0 / W, 1.0 - 8.0 / W, W_out, device=x.device),
            indexing='ij'
        )
        base_grid = torch.stack([base_x, base_y], dim=-1)
        
        I_c = F.grid_sample(x, base_grid.unsqueeze(0).expand(B, -1, -1, -1), 
                            mode='bilinear', padding_mode='reflection', align_corners=True)
        
        b_bits = []
        for p in range(8):
            dx = self.offsets[p, 0] * (2.0 / W)
            dy = self.offsets[p, 1] * (2.0 / H)
            sample_grid = base_grid + torch.stack([dx, dy])
            I_p = F.grid_sample(x, sample_grid.unsqueeze(0).expand(B, -1, -1, -1),
                                mode='bilinear', padding_mode='reflection', align_corners=True)
            diff = (I_p - I_c).squeeze(1) - self.thresholds[p]
            b_hard = (diff >= 0).float()
            b_soft = torch.sigmoid(beta * diff)
            b_ste = b_hard - b_soft.detach() + b_soft
            b_bits.append(b_ste)
            
        b = torch.stack(b_bits, dim=1)
        
        A_terms = []
        for i in range(16):
            t0 = b[:, 0] if (i & 1) else (1.0 - b[:, 0])
            t1 = b[:, 1] if (i & 2) else (1.0 - b[:, 1])
            t2 = b[:, 2] if (i & 4) else (1.0 - b[:, 2])
            t3 = b[:, 3] if (i & 8) else (1.0 - b[:, 3])
            A_terms.append(t0 * t1 * t2 * t3)
        A = torch.stack(A_terms, dim=1).view(B, 16, -1)
        
        B_terms = []
        for j in range(16):
            t4 = b[:, 4] if (j & 1) else (1.0 - b[:, 4])
            t5 = b[:, 5] if (j & 2) else (1.0 - b[:, 5])
            t6 = b[:, 6] if (j & 4) else (1.0 - b[:, 6])
            t7 = b[:, 7] if (j & 8) else (1.0 - b[:, 7])
            B_terms.append(t4 * t5 * t6 * t7)
        B_mat = torch.stack(B_terms, dim=1).view(B, 16, -1)
        
        hist = torch.bmm(A, B_mat.transpose(1, 2)) / float(H_out * W_out)
        hist = hist.view(B, 256)
        return hist

class TextureClassifier(nn.Module):
    def __init__(self, extractor, num_classes=6):
        super().__init__()
        self.extractor = extractor
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x):
        feat = self.extractor(x)
        out = self.fc(feat)
        return out, feat

def train_model(model, train_loader, test_loader, epochs=40, lr_extractor=0.04, lr_clf=0.02):
    params = []
    if any(p.requires_grad for p in model.extractor.parameters()):
        params.append({'params': [p for p in model.extractor.parameters() if p.requires_grad], 'lr': lr_extractor})
    params.append({'params': model.fc.parameters(), 'lr': lr_clf})
    
    optimizer = torch.optim.Adam(params)
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
    
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            logits, _ = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(y_b)
            correct += (logits.argmax(dim=-1) == y_b).sum().item()
            total += len(y_b)
        
        train_loss = total_loss / total
        train_acc = correct / total
        
        model.eval()
        t_loss, t_correct, t_total = 0.0, 0, 0
        with torch.no_grad():
            for X_b, y_b in test_loader:
                logits, _ = model(X_b)
                loss = criterion(logits, y_b)
                t_loss += loss.item() * len(y_b)
                t_correct += (logits.argmax(dim=-1) == y_b).sum().item()
                t_total += len(y_b)
        
        test_loss = t_loss / t_total
        test_acc = t_correct / t_total
        
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
    return history

configs = {
    'fixed_r1': {'radius': 1.0, 'learn_offsets': False, 'learn_thresholds': False},
    'fixed_r2': {'radius': 2.0, 'learn_offsets': False, 'learn_thresholds': False},
    'flex_offsets_only': {'radius': 1.5, 'learn_offsets': True, 'learn_thresholds': False},
    'flex_thresholds_only': {'radius': 1.5, 'learn_offsets': False, 'learn_thresholds': True},
    'flex_full': {'radius': 1.5, 'learn_offsets': True, 'learn_thresholds': True}
}

results = {}
models = {}
histories = {}

for name, cfg in configs.items():
    set_seed(42)
    ext = FlexLBPModule(init_radius=cfg['radius'], learn_offsets=cfg['learn_offsets'], learn_thresholds=cfg['learn_thresholds'])
    clf = TextureClassifier(ext, num_classes=6)
    hist = train_model(clf, train_loader, test_loader, epochs=45, lr_extractor=0.03, lr_clf=0.015)
    models[name] = clf
    histories[name] = hist
    results[name + '_acc'] = round(hist['test_acc'][-1] * 100, 2)
    results[name + '_loss'] = round(hist['test_loss'][-1], 4)

class MultiscaleFixedLBP(nn.Module):
    def __init__(self):
        super().__init__()
        self.ext1 = FlexLBPModule(init_radius=1.0, learn_offsets=False, learn_thresholds=False)
        self.ext2 = FlexLBPModule(init_radius=2.0, learn_offsets=False, learn_thresholds=False)
        self.fc = nn.Linear(512, 6)
    def forward(self, x):
        h1 = self.ext1(x)
        h2 = self.ext2(x)
        h = torch.cat([h1, h2], dim=-1)
        return self.fc(h), h

set_seed(42)
ms_model = MultiscaleFixedLBP()
optimizer_ms = torch.optim.Adam(ms_model.fc.parameters(), lr=0.015)
criterion = nn.CrossEntropyLoss()
for epoch in range(45):
    ms_model.train()
    for X_b, y_b in train_loader:
        optimizer_ms.zero_grad()
        logits, _ = ms_model(X_b)
        loss = criterion(logits, y_b)
        loss.backward()
        optimizer_ms.step()

ms_model.eval()
ms_correct, ms_total = 0, 0
with torch.no_grad():
    for X_b, y_b in test_loader:
        logits, _ = ms_model(X_b)
        ms_correct += (logits.argmax(dim=-1) == y_b).sum().item()
        ms_total += len(y_b)
results['multiscale_fixed_acc'] = round(ms_correct / ms_total * 100, 2)

models['fixed_r1'].eval()
start_time = time.perf_counter()
with torch.no_grad():
    for _ in range(10):
        for X_b, _ in test_loader:
            _ = models['fixed_r1'](X_b)
fixed_inf_time = (time.perf_counter() - start_time) / (10 * len(X_test)) * 1000.0

models['flex_full'].eval()
start_time = time.perf_counter()
with torch.no_grad():
    for _ in range(10):
        for X_b, _ in test_loader:
            _ = models['flex_full'](X_b)
flex_inf_time = (time.perf_counter() - start_time) / (10 * len(X_test)) * 1000.0

results['inference_fixed_ms_per_image'] = round(fixed_inf_time, 3)
results['inference_flex_ms_per_image'] = round(flex_inf_time, 3)

init_offsets = models['flex_full'].extractor.offsets.detach().cpu().numpy().copy()
learned_offsets = models['flex_full'].extractor.offsets.detach().cpu().numpy().tolist()
learned_thresholds = [round(float(v), 4) for v in models['flex_full'].extractor.thresholds.detach().cpu().numpy()]

angles = [2.0 * np.pi * p / 8.0 for p in range(8)]
initial_offsets_circle = [[round(1.5 * np.cos(a), 4), round(1.5 * np.sin(a), 4)] for a in angles]

results['initial_offsets'] = initial_offsets_circle
results['learned_offsets'] = [[round(v[0], 4), round(v[1], 4)] for v in learned_offsets]
results['learned_thresholds'] = learned_thresholds
results['num_extractor_params'] = sum(p.numel() for p in models['flex_full'].extractor.parameters() if p.requires_grad)

fig1, ax = plt.subplots(figsize=(6, 6))
init_pts = np.array(initial_offsets_circle)
learn_pts = np.array(results['learned_offsets'])

circle = plt.Circle((0, 0), 1.5, color='gray', fill=False, linestyle='--', linewidth=1.2, label='Initial Circle (R=1.5)')
ax.add_patch(circle)
ax.scatter(0, 0, color='black', marker='+', s=100, label='Center Pixel (0, 0)')
ax.scatter(init_pts[:, 0], init_pts[:, 1], color='#3b82f6', marker='o', s=60, label='Initial Circular Offsets')
ax.scatter(learn_pts[:, 0], learn_pts[:, 1], color='#ef4444', marker='^', s=80, label='FLEX-LBP Learned Offsets')

for i in range(8):
    ax.annotate(f'p{i}', (init_pts[i, 0] + 0.08, init_pts[i, 1] + 0.08), fontsize=8, color='#1e3a8a')
    ax.annotate(f'p{i}', (learn_pts[i, 0] + 0.08, learn_pts[i, 1] + 0.08), fontsize=8, color='#991b1b')
    ax.arrow(init_pts[i, 0], init_pts[i, 1], learn_pts[i, 0] - init_pts[i, 0], learn_pts[i, 1] - init_pts[i, 1],
             head_width=0.08, head_length=0.08, fc='#64748b', ec='#64748b', length_includes_head=True, alpha=0.7)

ax.set_xlim(-2.8, 2.8)
ax.set_ylim(-2.8, 2.8)
ax.set_xlabel('Sampling Offset dx (pixels)', fontsize=11)
ax.set_ylabel('Sampling Offset dy (pixels)', fontsize=11)
ax.set_title('Sampling Offsets: Fixed Circle vs. Learned FLEX-LBP', fontsize=12)
ax.grid(True, linestyle=':', alpha=0.6)
ax.set_aspect('equal')
ax.legend(loc='lower left', fontsize=9)
fig1.tight_layout()
fig1.savefig(os.path.join(out_dir, 'learned_offsets.png'), dpi=150)
plt.close(fig1)

fig2, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4.5))
epochs_arr = np.arange(1, 46)

ax_loss.plot(epochs_arr, histories['fixed_r1']['test_loss'], label='Fixed LBP (R=1.0)', color='#64748b', linestyle='--')
ax_loss.plot(epochs_arr, histories['fixed_r2']['test_loss'], label='Fixed LBP (R=2.0)', color='#0284c7', linestyle='--')
ax_loss.plot(epochs_arr, histories['flex_full']['test_loss'], label='FLEX-LBP (Full)', color='#ef4444', linewidth=2.0)
ax_loss.set_xlabel('Epoch', fontsize=11)
ax_loss.set_ylabel('Test Cross-Entropy Loss', fontsize=11)
ax_loss.set_title('Test Loss Trajectory', fontsize=12)
ax_loss.grid(True, linestyle=':', alpha=0.6)
ax_loss.legend(fontsize=9)

ax_acc.plot(epochs_arr, [v * 100 for v in histories['fixed_r1']['test_acc']], label='Fixed LBP (R=1.0)', color='#64748b', linestyle='--')
ax_acc.plot(epochs_arr, [v * 100 for v in histories['fixed_r2']['test_acc']], label='Fixed LBP (R=2.0)', color='#0284c7', linestyle='--')
ax_acc.plot(epochs_arr, [v * 100 for v in histories['flex_full']['test_acc']], label='FLEX-LBP (Full)', color='#ef4444', linewidth=2.0)
ax_acc.set_xlabel('Epoch', fontsize=11)
ax_acc.set_ylabel('Test Accuracy (%)', fontsize=11)
ax_acc.set_title('Test Accuracy Trajectory', fontsize=12)
ax_acc.grid(True, linestyle=':', alpha=0.6)
ax_acc.legend(fontsize=9)

fig2.tight_layout()
fig2.savefig(os.path.join(out_dir, 'training_curves.png'), dpi=150)
plt.close(fig2)

all_preds_fixed = []
all_preds_flex = []
all_targets = []

models['fixed_r2'].eval()
models['flex_full'].eval()
with torch.no_grad():
    for X_b, y_b in test_loader:
        p_fix, _ = models['fixed_r2'](X_b)
        p_flx, _ = models['flex_full'](X_b)
        all_preds_fixed.extend(p_fix.argmax(dim=-1).cpu().numpy())
        all_preds_flex.extend(p_flx.argmax(dim=-1).cpu().numpy())
        all_targets.extend(y_b.cpu().numpy())

cm_fixed = confusion_matrix(all_targets, all_preds_fixed)
cm_flex = confusion_matrix(all_targets, all_preds_flex)
class_names = ['Horiz', 'Vert', 'Diag', 'Check', 'Blobs', 'PinkN']

fig3, (ax_cm1, ax_cm2) = plt.subplots(1, 2, figsize=(11, 4.8))
im1 = ax_cm1.imshow(cm_fixed, interpolation='nearest', cmap=plt.cm.Blues)
ax_cm1.set_title(f'Fixed LBP R=2.0 (Acc: {results["fixed_r2_acc"]}%)', fontsize=11)
ax_cm1.set_xticks(range(6))
ax_cm1.set_yticks(range(6))
ax_cm1.set_xticklabels(class_names, rotation=45)
ax_cm1.set_yticklabels(class_names)
ax_cm1.set_xlabel('Predicted Label', fontsize=10)
ax_cm1.set_ylabel('True Label', fontsize=10)
for i in range(6):
    for j in range(6):
        ax_cm1.text(j, i, format(cm_fixed[i, j], 'd'), ha='center', va='center', 
                    color='white' if cm_fixed[i, j] > cm_fixed.max() / 2 else 'black')

im2 = ax_cm2.imshow(cm_flex, interpolation='nearest', cmap=plt.cm.Blues)
ax_cm2.set_title(f'FLEX-LBP Full (Acc: {results["flex_full_acc"]}%)', fontsize=11)
ax_cm2.set_xticks(range(6))
ax_cm2.set_yticks(range(6))
ax_cm2.set_xticklabels(class_names, rotation=45)
ax_cm2.set_yticklabels(class_names)
ax_cm2.set_xlabel('Predicted Label', fontsize=10)
ax_cm2.set_ylabel('True Label', fontsize=10)
for i in range(6):
    for j in range(6):
        ax_cm2.text(j, i, format(cm_flex[i, j], 'd'), ha='center', va='center',
                    color='white' if cm_flex[i, j] > cm_flex.max() / 2 else 'black')

fig3.colorbar(im2, ax=[ax_cm1, ax_cm2], fraction=0.03, pad=0.04)
fig3.tight_layout()
fig3.savefig(os.path.join(out_dir, 'confusion_matrices.png'), dpi=150)
plt.close(fig3)

fig4, ax_bar = plt.subplots(figsize=(8, 4.5))
methods = ['Fixed LBP\n(R=1.0)', 'Fixed LBP\n(R=2.0)', 'Fixed Multi\n(R=1+2)', 'FLEX\n(Offsets)', 'FLEX\n(Thresh)', 'FLEX\n(Full)']
accuracies = [
    results['fixed_r1_acc'],
    results['fixed_r2_acc'],
    results['multiscale_fixed_acc'],
    results['flex_offsets_only_acc'],
    results['flex_thresholds_only_acc'],
    results['flex_full_acc']
]
bar_colors = ['#94a3b8', '#64748b', '#475569', '#38bdf8', '#818cf8', '#ef4444']
bars = ax_bar.bar(methods, accuracies, color=bar_colors, width=0.6, edgecolor='#1e293b')
ax_bar.set_ylabel('Test Accuracy (%)', fontsize=11)
ax_bar.set_ylim(min(accuracies) - 10.0, 105.0)
ax_bar.set_title('Texture Classification Accuracy: Fixed vs. Learned LBP Descriptors', fontsize=12)
ax_bar.grid(axis='y', linestyle=':', alpha=0.6)

for bar in bars:
    yval = bar.get_height()
    ax_bar.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.2, f'{yval:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

fig4.tight_layout()
fig4.savefig(os.path.join(out_dir, 'ablation_accuracy.png'), dpi=150)
plt.close(fig4)

with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)
