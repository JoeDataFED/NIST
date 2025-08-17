#%%
from __future__ import annotations
import os, math, json, random, pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint as ckpt

BASE_DIR = Path(os.getcwd()).resolve()
SAVE_DIR = BASE_DIR / "checkpoints_t2"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

T2CFG = {
    "seed": 42,
    "img_h": 139, "img_w": 250, "img_c": 3,
    "time_dim": 512,
    "base_ch": 32,
    "ch_mults": (1, 2, 4),
    "groups": 8,
    "timesteps": 1000,
    "beta_start": 1e-4, "beta_end": 2e-2,
    "use_checkpoint": True,
    "epochs": 50,
    "batch": 2,
    "accum_steps": 4,
    "lr": 2e-4,
    "wd": 1e-4,
    "grad_clip": 1.0,
    "amp": True,
    "channels_last": True,
    "num_workers": 0,
    "num_per_cond": 100,
    "sample_bs": 10
}

def seed_everything(s: int):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_everything(T2CFG["seed"])

def safe_load_dataset(filename: str) -> Optional[Dict]:
    p = BASE_DIR / filename
    if not p.exists():
        print(f"[WARN] Miss dataset: {p}")
        return None
    print(f"[INFO] Load: {p}")
    with open(p, "rb") as f:
        return pickle.load(f)

def _key_endswith(images: Dict[str, np.ndarray], suffix: str) -> str:
    for k in images.keys():
        if k.lower().endswith(suffix.lower()):
            return k
    raise KeyError(f"Not found suffix '{suffix}' in keys: {list(images.keys())}")

def extract_triplet(images: Dict[str, np.ndarray]) -> Tuple[np.ndarray,np.ndarray,np.ndarray,Tuple[str,str,str]]:
    ka = _key_endswith(images, "a")
    kb = _key_endswith(images, "b")
    kc = _key_endswith(images, "c")
    return images[ka], images[kb], images[kc], (ka,kb,kc)

LABELED = safe_load_dataset("labeled_training_set.pkl")
UNLABELED = safe_load_dataset("unlabeled_training_set.pkl")

def collect_images_by_condition(ds: Optional[Dict], cond: str) -> List[np.ndarray]:
    out = []
    if not ds: return out
    for _, arr in ds.items():
        for it in arr:
            xa, xb, xc, _ = extract_triplet(it["images"])
            x = {"a": xa, "b": xb, "c": xc}[cond]
            out.append(x)
    return out

def to_float01(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32)
    if x.max() > 1.5: x = x / 255.0
    return np.clip(x, 0.0, 1.0)

class Task2ROIDataset(Dataset):
    def __init__(self, conds=("a","b","c")):
        self.samples = []
        self.cond_map = {"a":0, "b":1, "c":2}
        H, W = T2CFG["img_h"], T2CFG["img_w"]
        for cond in conds:
            imgs = collect_images_by_condition(LABELED, cond) + collect_images_by_condition(UNLABELED, cond)
            for im in imgs:
                x = torch.from_numpy(to_float01(im)).permute(2,0,1).unsqueeze(0)
                x = F.interpolate(x, size=(H,W), mode="bilinear", align_corners=False)[0]
                self.samples.append((x, self.cond_map[cond]))
        print(f"[DATA] Total samples: {len(self.samples)}")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x, cid = self.samples[idx]
        if random.random() < 0.5:
            b = 1.0 + (random.random()-0.5)*0.1
            c = 1.0 + (random.random()-0.5)*0.2
            mean = x.mean(dim=(1,2), keepdim=True)
            x = (x*b - mean)*c + mean
            x = x.clamp(0.0, 1.0)
        return x, cid

train_ds = Task2ROIDataset(("a","b","c"))
train_loader = DataLoader(
    train_ds, batch_size=T2CFG["batch"], shuffle=True,
    num_workers=T2CFG["num_workers"], pin_memory=True, drop_last=True
)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32)
                          * -(math.log(10000.0) / (half - 1)))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=1)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, groups=8, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        g1 = min(groups, in_ch) if (in_ch % groups) != 0 else groups
        g2 = min(groups, out_ch) if (out_ch % groups) != 0 else groups
        self.norm1 = nn.GroupNorm(g1, in_ch)
        self.act1  = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.norm2 = nn.GroupNorm(g2, out_ch)
        self.act2  = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.res = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def _forward_impl(self, x, t_emb):
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.res(x)

    def forward(self, x, t_emb):
        if self.use_checkpoint and self.training:
            return ckpt(self._forward_impl, x, t_emb)
        else:
            return self._forward_impl(x, t_emb)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, groups=8, use_checkpoint=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResBlock(in_ch, out_ch, time_dim, groups, use_checkpoint),
            ResBlock(out_ch, out_ch, time_dim, groups, use_checkpoint)
        ])
        self.pool = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, time_dim, groups=8, use_checkpoint=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResBlock(in_ch + skip_ch, out_ch, time_dim, groups, use_checkpoint),
            ResBlock(out_ch, out_ch, time_dim, groups, use_checkpoint)
        ])
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

class CondUNet(nn.Module):
    def __init__(self, img_c=3, base_ch=32, ch_mults=(1,2,4), time_dim=512, groups=8, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.cond_emb = nn.Embedding(3, time_dim)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        self.in_conv = nn.Conv2d(img_c, base_ch, 3, padding=1)
        chs = [base_ch * m for m in ch_mults]
        self.downs = nn.ModuleList([
            Down(base_ch, chs[0], time_dim, groups, use_checkpoint),
            Down(chs[0], chs[1], time_dim, groups, use_checkpoint),
            Down(chs[1], chs[2], time_dim, groups, use_checkpoint)
        ])
        mid_ch = chs[-1]
        self.mid1 = ResBlock(mid_ch, mid_ch, time_dim, groups, use_checkpoint)
        self.mid2 = ResBlock(mid_ch, mid_ch, time_dim, groups, use_checkpoint)
        self.ups = nn.ModuleList([
            Up(chs[2], chs[2], chs[1], time_dim, groups, use_checkpoint),
            Up(chs[1], chs[1], chs[0], time_dim, groups, use_checkpoint),
            Up(chs[0], chs[0], base_ch, time_dim, groups, use_checkpoint)
        ])
        self.out_norm = nn.GroupNorm(min(groups, base_ch), base_ch)
        self.out_act  = nn.SiLU()
        self.out_conv = nn.Conv2d(base_ch, img_c, 3, padding=1)

    def forward(self, x, t: torch.Tensor, cond_id: torch.Tensor):
        orig_size = x.shape[-2:]
        t_emb = self.time_mlp(t) + self.cond_emb(cond_id)
        h = self.in_conv(x)
        hs = []
        for d in self.downs:
            for blk in d.blocks:
                h = blk(h, t_emb)
            hs.append(h)
            h = d.pool(h)
        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)
        for up, skip in zip(self.ups, reversed(hs)):
            h = up.up(h)
            if skip.shape[-2:] != h.shape[-2:]:
                skip = F.interpolate(skip, size=h.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            for blk in up.blocks:
                h = blk(h, t_emb)
        out = self.out_conv(self.out_act(self.out_norm(h)))
        if out.shape[-2:] != orig_size:
            out = F.interpolate(out, size=orig_size, mode="bilinear", align_corners=False)
        return out

class DiffSchedule:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=2e-2, device="cpu"):
        self.T = T
        betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32, device=device)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - self.alphas_bar)
        self.alphas_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas[:-1]], dim=0)
        self.posterior_variance = betas * (1.0 - self.alphas_prev) / (1.0 - self.alphas_bar)
        self.posterior_logvar_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = betas * torch.sqrt(self.alphas_prev) / (1.0 - self.alphas_bar)
        self.posterior_mean_coef2 = (1.0 - self.alphas_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_bar)

SCHED = DiffSchedule(T2CFG["timesteps"], T2CFG["beta_start"], T2CFG["beta_end"], device=device)

def q_sample(x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor]=None) -> torch.Tensor:
    if noise is None: noise = torch.randn_like(x0)
    sqrt_ab = SCHED.sqrt_alphas_bar[t].view(-1,1,1,1)
    sqrt_om = SCHED.sqrt_one_minus_alphas_bar[t].view(-1,1,1,1)
    return sqrt_ab * x0 + sqrt_om * noise

def p_losses(model: nn.Module, x0: torch.Tensor, t: torch.Tensor, cond_id: torch.Tensor) -> torch.Tensor:
    noise = torch.randn_like(x0)
    xt = q_sample(x0, t, noise)
    pred_eps = model(xt, t, cond_id)
    return F.mse_loss(pred_eps, noise)

@torch.no_grad()
def p_sample(model: nn.Module, xt: torch.Tensor, t: torch.Tensor, cond_id: torch.Tensor) -> torch.Tensor:
    eps = model(xt, t, cond_id)
    a_t = SCHED.alphas[t].view(-1,1,1,1)
    ab_t = SCHED.alphas_bar[t].view(-1,1,1,1)
    sqrt_one_minus_ab = torch.sqrt(1.0 - ab_t)
    x0 = (xt - sqrt_one_minus_ab * eps) / torch.sqrt(ab_t).clamp(min=1e-8)
    mean = (SCHED.posterior_mean_coef1[t].view(-1,1,1,1) * x0 +
            SCHED.posterior_mean_coef2[t].view(-1,1,1,1) * xt)
    noise = torch.randn_like(xt) if (t > 0).any() else torch.zeros_like(xt)
    logvar = SCHED.posterior_logvar_clipped[t].view(-1,1,1,1)
    return mean + torch.exp(0.5*logvar) * noise

@torch.no_grad()
def p_sample_loop(model: nn.Module, shape: Tuple[int,int,int,int], cond_id: int) -> torch.Tensor:
    b, c, h, w = shape
    x = torch.randn(shape, device=device)
    cond = torch.full((b,), cond_id, dtype=torch.long, device=device)
    for i in reversed(range(SCHED.T)):
        t = torch.full((b,), i, dtype=torch.long, device=device)
        x = p_sample(model, x, t, cond)
        x = x.clamp(-1.5, 1.5)
    return x

class EMA:
    def __init__(self, m: nn.Module, decay=0.999):
        self.ema = CondUNet(
            img_c=T2CFG["img_c"], base_ch=T2CFG["base_ch"], ch_mults=T2CFG["ch_mults"],
            time_dim=T2CFG["time_dim"], groups=T2CFG["groups"], use_checkpoint=False
        ).to(device)
        self.ema.load_state_dict(m.state_dict())
        if T2CFG["channels_last"]:
            self.ema = self.ema.to(memory_format=torch.channels_last)
        for p in self.ema.parameters(): p.requires_grad_(False)
        self.decay = decay
    @torch.no_grad()
    def update(self, m: nn.Module):
        for (k,v),(_,nv) in zip(self.ema.state_dict().items(), m.state_dict().items()):
            v.copy_(v * self.decay + (1.0 - self.decay) * nv)

model = CondUNet(
    img_c=T2CFG["img_c"], base_ch=T2CFG["base_ch"], ch_mults=T2CFG["ch_mults"],
    time_dim=T2CFG["time_dim"], groups=T2CFG["groups"], use_checkpoint=T2CFG["use_checkpoint"]
).to(device)
if T2CFG["channels_last"]:
    model = model.to(memory_format=torch.channels_last)

ema = EMA(model, decay=0.999)
optimizer = torch.optim.AdamW(model.parameters(), lr=T2CFG["lr"], weight_decay=T2CFG["wd"])
scaler = torch.amp.GradScaler('cuda', enabled=(T2CFG["amp"] and device.type=="cuda"))

def train_one_epoch(epoch_idx: int) -> float:
    model.train()
    total_loss, n = 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    for step, (x, cid) in enumerate(train_loader, 1):
        x = x.to(device, non_blocking=True)
        cid = cid.to(device, non_blocking=True)
        if T2CFG["channels_last"]:
            x = x.contiguous(memory_format=torch.channels_last)
        t = torch.randint(0, SCHED.T, (x.size(0),), device=device, dtype=torch.long)
        try:
            with torch.amp.autocast('cuda', enabled=scaler.is_enabled()):
                loss = p_losses(model, x, t, cid) / T2CFG["accum_steps"]
            scaler.scale(loss).backward()
            if step % T2CFG["accum_steps"] == 0:
                if T2CFG["grad_clip"] and T2CFG["grad_clip"] > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), T2CFG["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
            total_loss += float(loss.detach().cpu().item()) * x.size(0)
            n += x.size(0)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[OOM] skipped step {step}, clearing cache.")
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue
            else:
                raise e
        if step % 20 == 0 or step == len(train_loader):
            avg = total_loss / max(n,1)
            print(f"[E{epoch_idx:02d}] step {step:03d}/{len(train_loader)} | loss={avg:.4f}")
    return total_loss / max(n,1)

print(f"[TRAIN] Start training: epochs={T2CFG['epochs']}, steps_per_epoch≈{len(train_loader)}")
train_losses = []
for ep in range(1, T2CFG["epochs"]+1):
    l = train_one_epoch(ep)
    train_losses.append((ep, l))
    torch.save({"ema": ema.ema.state_dict(), "cfg": T2CFG}, SAVE_DIR / f"t2_ema_ep{ep:03d}.pt")
    print(f"[SAVE] EMA @ epoch {ep} saved.")
torch.save({"ema": ema.ema.state_dict(), "cfg": T2CFG}, SAVE_DIR / "t2_ema_latest.pt")
print("[TRAIN] Finished.")

@torch.no_grad()
def sample_condition_arrays(model_ema: nn.Module, cond_id: int, n: int, bs: int=10) -> np.ndarray:
    model_ema.eval()
    H, W, C = T2CFG["img_h"], T2CFG["img_w"], T2CFG["img_c"]
    out_list, remain = [], n
    while remain > 0:
        b = min(bs, remain)
        x = p_sample_loop(model_ema, (b, C, H, W), cond_id)
        x = x.clamp(0,1)
        x = (x * 255.0).round().byte().cpu().permute(0,2,3,1).numpy()
        out_list.append(x)
        remain -= b
    arr = np.concatenate(out_list, axis=0)
    assert arr.shape == (n, H, W, C)
    return arr

ck = torch.load(SAVE_DIR / "t2_ema_latest.pt", map_location=device)
ema.ema.load_state_dict(ck["ema"])

arr_a = sample_condition_arrays(ema.ema, cond_id=0, n=T2CFG["num_per_cond"], bs=T2CFG["sample_bs"])
arr_b = sample_condition_arrays(ema.ema, cond_id=1, n=T2CFG["num_per_cond"], bs=T2CFG["sample_bs"])
arr_c = sample_condition_arrays(ema.ema, cond_id=2, n=T2CFG["num_per_cond"], bs=T2CFG["sample_bs"])

with open(BASE_DIR / "NIST_Task2_a.pkl", "wb") as f:
    pickle.dump(arr_a, f)
with open(BASE_DIR / "NIST_Task2_b.pkl", "wb") as f:
    pickle.dump(arr_b, f)
with open(BASE_DIR / "NIST_Task2_c.pkl", "wb") as f:
    pickle.dump(arr_c, f)

print("[SUBMIT] Saved:",
      BASE_DIR / "NIST_Task2_a.pkl",
      BASE_DIR / "NIST_Task2_b.pkl",
      BASE_DIR / "NIST_Task2_c.pkl")

import matplotlib.pyplot as plt
if train_losses:
    ep = [e for e,_ in train_losses]
    ls = [l for _,l in train_losses]
    plt.figure(figsize=(6,4))
    plt.plot(ep, ls, marker="o")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Task2 — Diffusion Train Loss")
    plt.grid(True, ls="--", alpha=0.3)
    out_png = SAVE_DIR / "t2_train_loss.png"
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.show()
    print(f"[PLOT] Train loss curve saved to {out_png}")
else:
    print("[PLOT] No losses to plot.")

import os, json, math, random, pickle, warnings
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

EVAL_DIR = SAVE_DIR / "eval_task2"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

gen_paths = {
    "a": BASE_DIR / "NIST_Task2_a.pkl",
    "b": BASE_DIR / "NIST_Task2_b.pkl",
    "c": BASE_DIR / "NIST_Task2_c.pkl",
}
GENERATED: Dict[str, Optional[np.ndarray]] = {}
for k,p in gen_paths.items():
    if p.exists():
        with open(p, "rb") as f:
            arr = pickle.load(f)
        assert isinstance(arr, np.ndarray) and arr.ndim == 4 and arr.shape[-1] == 3, f"{k} bad shape: {type(arr)}, {getattr(arr, 'shape', None)}"
        GENERATED[k] = arr
        print(f"[EVAL] Loaded generated {k}: {arr.shape}, dtype={arr.dtype}")
    else:
        GENERATED[k] = None
        print(f"[EVAL][WARN] Missing generated file for {k}: {p}")

def _extract_real_by_cond(cond: str, max_n: int = 1000) -> np.ndarray:
    H, W = T2CFG["img_h"], T2CFG["img_w"]
    imgs = collect_images_by_condition(LABELED, cond) + collect_images_by_condition(UNLABELED, cond)
    out = []
    rng = random.Random(T2CFG["seed"])
    rng.shuffle(imgs)
    for im in imgs[:max_n]:
        x = torch.from_numpy(to_float01(im)).permute(2,0,1).unsqueeze(0)
        x = F.interpolate(x, size=(H,W), mode="bilinear", align_corners=False)[0]
        x = (x.clamp(0,1) * 255.0).round().byte().permute(1,2,0).numpy()
        out.append(x)
    if not out:
        return np.zeros((0, H, W, 3), np.uint8)
    arr = np.stack(out, axis=0)
    print(f"[EVAL] Real {cond}: {arr.shape}")
    return arr

REAL = {k: _extract_real_by_cond(k, max_n=1000) for k in ("a","b","c")}

def _grid(axs_imgs: List[np.ndarray], cols: int = 8, title: str = "", out_path: Optional[Path]=None):
    n = len(axs_imgs)
    rows = math.ceil(n/cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*1.6, rows*1.6))
    if rows == 1: axes = np.array([axes])
    for i in range(rows*cols):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.axis("off")
        if i < n:
            ax.imshow(axs_imgs[i])
    if title: fig.suptitle(title)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=160)
        print(f"[PLOT] Saved {out_path}")
    plt.show()

for cond in ("a","b","c"):
    gen = GENERATED.get(cond)
    real = REAL.get(cond)
    if gen is None or gen.shape[0] == 0 or real.shape[0] == 0:
        print(f"[PLOT][SKIP] {cond}")
        continue
    idx_g = np.linspace(0, gen.shape[0]-1, 16, dtype=int)
    idx_r = np.linspace(0, real.shape[0]-1, 16, dtype=int)
    _grid([real[i] for i in idx_r], title=f"Real {cond} (sample)", out_path=EVAL_DIR / f"real_{cond}_grid.png")
    _grid([gen[i]  for i in idx_g], title=f"Generated {cond} (sample)", out_path=EVAL_DIR / f"gen_{cond}_grid.png")

def _load_inception():
    try:
        import torchvision
        from torchvision.models import inception_v3
        try:
            weights = torchvision.models.Inception_V3_Weights.IMAGENET1K_V1
            model = inception_v3(weights=weights, aux_logits=False).eval().to(device)
        except Exception:
            model = inception_v3(pretrained=True, aux_logits=False).eval().to(device)
        model.fc = nn.Identity()
        return model, True
    except Exception as e:
        warnings.warn(f"InceptionV3 unavailable ({e}); fallback to LightCNN for pseudo-FID.")
        class LightCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
                    nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                    nn.Conv2d(64,128, 3, stride=2, padding=1), nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(128, 256), nn.ReLU(),
                    nn.Linear(256, 128)
                )
            def forward(self, x): return self.net(x)
        return LightCNN().eval().to(device), False

INCEP, IS_INCEP = _load_inception()

def _preproc_for_inception(x: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.float32:
        x = x.float()
    x = x / 255.0 if x.max() > 1.0 else x.clamp(0,1)
    x = x.permute(0,3,1,2).contiguous()
    sz = 299 if IS_INCEP else 128
    x = F.interpolate(x, size=(sz,sz), mode="bilinear", align_corners=False)
    if IS_INCEP:
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
        x = (x - mean) / std
    return x

@torch.no_grad()
def get_activations(arr: np.ndarray, batch_size: int = 32) -> np.ndarray:
    feats = []
    n = arr.shape[0]
    for i in range(0, n, batch_size):
        b = torch.from_numpy(arr[i:i+batch_size]).to(device)
        b = _preproc_for_inception(b)
        f = INCEP(b)
        feats.append(f.detach().cpu().numpy())
    return np.concatenate(feats, axis=0)

def _sqrtm_psd(mat: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals)).dot(vecs.T)

def compute_fid(real: np.ndarray, fake: np.ndarray, max_real: int = 1000) -> float:
    if real.shape[0] > max_real:
        idx = np.random.RandomState(123).choice(real.shape[0], max_real, replace=False)
        real = real[idx]
    fr = get_activations(real)
    ff = get_activations(fake)
    m1, m2 = fr.mean(axis=0), ff.mean(axis=0)
    C1, C2 = np.cov(fr, rowvar=False), np.cov(ff, rowvar=False)
    diff = m1 - m2
    covmean = _sqrtm_psd(C1.dot(C2))
    fid = diff.dot(diff) + np.trace(C1 + C2 - 2.0*covmean)
    return float(np.real(fid))

def _load_lpips():
    try:
        import lpips
        net = lpips.LPIPS(net='alex').to(device).eval()
        return net, True
    except Exception as e:
        warnings.warn(f"lpips not available ({e}); using VGG16 feature L2 as proxy.")
        try:
            import torchvision
            from torchvision.models import vgg16
            try:
                weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1
                vgg = vgg16(weights=weights).features[:16].eval().to(device)
            except Exception:
                vgg = vgg16(pretrained=True).features[:16].eval().to(device)
            for p in vgg.parameters(): p.requires_grad_(False)
            return vgg, False
        except Exception as e2:
            warnings.warn(f"VGG16 also unavailable ({e2}); fallback to pixel-space L2.")
            return None, False

LPIPS_NET, IS_TRUE_LPIPS = _load_lpips()

@torch.no_grad()
def lpips_diversity(imgs: np.ndarray, npairs: int = 200) -> float:
    n = imgs.shape[0]
    if n < 2: return 0.0
    rng = np.random.RandomState(1234)
    pairs = rng.randint(0, n, size=(npairs, 2))
    dists = []
    for i,j in pairs:
        x = torch.from_numpy(imgs[[i,j]]).to(device).float()/255.0
        x = x.permute(0,3,1,2).contiguous()
        if LPIPS_NET is None:
            d = torch.mean((x[0]-x[1])**2).item()
        elif IS_TRUE_LPIPS:
            import lpips
            d = LPIPS_NET(x[0:1]*2-1, x[1:2]*2-1).item()
        else:
            mean = torch.tensor([0.485,0.456,0.406], device=x.device).view(1,3,1,1)
            std  = torch.tensor([0.229,0.224,0.225], device=x.device).view(1,3,1,1)
            xf = LPIPS_NET(((x-mean)/std))
            d = torch.mean((xf[0]-xf[1])**2).item()
        dists.append(d)
    return float(np.mean(dists))

results = {}
for cond in ("a","b","c"):
    gen = GENERATED.get(cond)
    real = REAL.get(cond)
    if gen is None or gen.shape[0] == 0 or real.shape[0] == 0:
        print(f"[EVAL][SKIP] {cond}")
        continue
    fid = compute_fid(real, gen, max_real=1000)
    ldiv = lpips_diversity(gen, npairs=min(500, gen.shape[0]*(gen.shape[0]-1)//2))
    results[cond] = {"FID": fid, "LPIPS_diversity": ldiv}
    print(f"[EVAL][{cond}] FID={fid:.3f} | LPIPS(diversity)={ldiv:.4f}")

with open(EVAL_DIR / "metrics_task2.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"[EVAL] Metrics saved → {EVAL_DIR / 'metrics_task2.json'}")

def side_by_side(real_arr: np.ndarray, gen_arr: np.ndarray, k: int = 8, title: str = "", out_path: Optional[Path]=None):
    k = min(k, real_arr.shape[0], gen_arr.shape[0])
    idx_r = np.random.RandomState(0).choice(real_arr.shape[0], k, replace=False)
    idx_g = np.random.RandomState(1).choice(gen_arr.shape[0], k, replace=False)
    fig, axes = plt.subplots(k, 2, figsize=(4.2, 1.8*k))
    for i,(ir,ig) in enumerate(zip(idx_r, idx_g)):
        axes[i,0].imshow(real_arr[ir]); axes[i,0].axis("off"); axes[i,0].set_title("Real")
        axes[i,1].imshow(gen_arr[ig]);  axes[i,1].axis("off"); axes[i,1].set_title("Gen")
    if title: fig.suptitle(title)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=160); print(f"[PLOT] Saved {out_path}")
    plt.show()

for cond in ("a","b","c"):
    gen = GENERATED.get(cond)
    real = REAL.get(cond)
    if gen is None or gen.shape[0] == 0 or real.shape[0] == 0:
        continue
    side_by_side(real, gen, k=8, title=f"{cond}: Real vs Generated", out_path=EVAL_DIR / f"{cond}_real_vs_gen.png")

import os, re, glob, pickle, random
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

BASE_DIR = Path(os.getcwd())
H, W, C = 139, 250, 3
N_PER_COND = 100
RNG = np.random.default_rng(123)

def load_real_from_disk() -> dict:
    patts = sorted(glob.glob(str(BASE_DIR / "Cropped_Part_*_Figure/**/*.png"), recursive=True))
    buckets = {"a": [], "b": [], "c": []}
    rx = re.compile(r".*([a-cA-C])\.png$")
    for p in patts:
        m = rx.match(p)
        if not m:
            continue
        key = m.group(1).lower()
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        buckets[key].append(img.astype(np.uint8))
    out = {}
    for k in "abc":
        if len(buckets[k])>0:
            out[k] = np.stack(buckets[k], 0)
            print(f"[REAL] loaded {k}: {out[k].shape}, min={out[k].min()} max={out[k].max()} mean={out[k].mean():.2f}")
    return out

try:
    REAL
except NameError:
    REAL = load_real_from_disk()

assert all(k in REAL and REAL[k].ndim==4 for k in "abc"), 

def ssd(a, b, mask=None):
    diff = (a.astype(np.int32)-b.astype(np.int32))**2
    if mask is None:
        return diff.mean()
    return (diff*mask).sum()/np.maximum(mask.sum(),1)

def min_error_boundary_cut(overlap_left, overlap_right, axis=1):
    err = np.mean((overlap_left.astype(np.float32)-overlap_right.astype(np.float32))**2, axis=2)
    h, w = err.shape
    cost = err.copy()
    path = np.zeros_like(err, dtype=np.int32)
    for i in range(1, h):
        for j in range(w):
            prev = cost[i-1, max(j-1,0):min(j+2,w)]
            idx = np.argmin(prev)
            cost[i,j] += prev[idx]
            path[i,j] = idx + max(j-1,0)
    cut = np.zeros(h, np.int32)
    cut[-1] = int(np.argmin(cost[-1]))
    for i in range(h-2, -1, -1):
        cut[i] = path[i+1, cut[i+1]]
    mask = np.zeros((h,w), np.uint8)
    for i in range(h):
        mask[i,:cut[i]] = 1
    return mask

def quilt_from_bank(bank: np.ndarray, tile=32, overlap=8, bias_bands=False) -> np.ndarray:
    out = np.zeros((H,W,3), np.uint8)
    vert_energy = None
    if bias_bands:
        vert_energy = []
        ky = cv2.getDerivKernels(2,0,ksize=3)[0]
        for i in range(len(bank)):
            g = cv2.Sobel(cv2.cvtColor(bank[i], cv2.COLOR_RGB2GRAY), cv2.CV_32F, 0, 1, ksize=3)
            vert_energy.append(float(np.mean(np.abs(g))))
        vert_energy = np.array(vert_energy, np.float32)
        vert_energy = (vert_energy - vert_energy.min())/(vert_energy.ptp()+1e-6)
    for y in range(0, H, tile-overlap):
        for x in range(0, W, tile-overlap):
            h_ = min(tile, H-y); w_ = min(tile, W-x)
            ref = np.zeros((h_, w_, 3), np.uint8)
            mask = np.zeros((h_, w_), np.uint8)
            if y>0:
                ref_top = out[y:y+overlap, x:x+w_]
                ref[:overlap, :w_] = ref_top
                mask[:overlap, :w_] = 255
            if x>0:
                ref_left = out[y:y+h_, x:x+overlap]
                ref[:h_, :overlap] = ref_left
                mask[:h_, :overlap] = 255
            cand_idx = RNG.choice(len(bank), size=min(500, len(bank)), replace=False)
            best_cost = 1e18; best_patch=None
            for idx in cand_idx:
                src = bank[idx]
                sy = RNG.integers(0, max(1, src.shape[0]-h_+1))
                sx = RNG.integers(0, max(1, src.shape[1]-w_+1))
                patch = src[sy:sy+h_, sx:sx+w_]
                cost = ssd(patch, ref, (mask>0)[...,None])
                if bias_bands:
                    w_bias = 0.2 + 0.8*vert_energy[idx]
                    cost *= (1.0 - 0.3*w_bias)
                if cost < best_cost:
                    best_cost = cost
                    best_patch = patch
            placed = best_patch.copy()
            if y>0:
                ovl_h = min(overlap, h_)
                mask_top = min_error_boundary_cut(placed[:ovl_h,:], ref[:ovl_h,:], axis=0)
                top_sel = (mask_top==1)[:,:,None]
                placed[:ovl_h,:] = top_sel*ref[:ovl_h,:] + (1-top_sel)*placed[:ovl_h,:]
            if x>0:
                ovl_w = min(overlap, w_)
                mask_left = min_error_boundary_cut(placed[:,:ovl_w], ref[:,:ovl_w], axis=1)
                left_sel = (mask_left==1)[:,:,None]
                placed[:,:ovl_w] = left_sel*ref[:,:ovl_w] + (1-left_sel)*placed[:,:ovl_w]
            out[y:y+h_, x:x+w_] = placed
    return out

def hist_match_to_real(x: np.ndarray, real_bank: np.ndarray) -> np.ndarray:
    y = x.copy()
    for ch in range(3):
        idx = RNG.choice(len(real_bank), size=min(200, len(real_bank)), replace=False)
        ref = real_bank[idx, :, :, ch].reshape(-1)
        hist_ref, bins = np.histogram(ref, bins=256, range=(0,255), density=True)
        cdf_ref = np.cumsum(hist_ref); cdf_ref /= cdf_ref[-1]
        src = y[:,:,ch].reshape(-1)
        hist_src, _ = np.histogram(src, bins=256, range=(0,255), density=True)
        cdf_src = np.cumsum(hist_src); cdf_src /= cdf_src[-1]
        lut = np.interp(cdf_src, cdf_ref, np.arange(256))
        y[:,:,ch] = np.interp(src, np.arange(256), lut).reshape(H,W).astype(np.uint8)
    return y

def synthesize_set(real_bank: np.ndarray, bias_bands=False) -> np.ndarray:
    out = np.zeros((N_PER_COND, H, W, C), np.uint8)
    for i in range(N_PER_COND):
        img = quilt_from_bank(real_bank, tile=32, overlap=8, bias_bands=bias_bands)
        img = hist_match_to_real(img, real_bank)
        out[i] = img
    return out

print("[SYN] start quilting…")
gen_a = synthesize_set(REAL['a'], bias_bands=False)
gen_b = synthesize_set(REAL['b'], bias_bands=True)
gen_c = synthesize_set(REAL['c'], bias_bands=False)
print("[SYN] done.", gen_a.shape, gen_b.shape, gen_c.shape)

with open(BASE_DIR/"NIST_Task2_a.pkl","wb") as f: pickle.dump(gen_a, f)
with open(BASE_DIR/"NIST_Task2_b.pkl","wb") as f: pickle.dump(gen_b, f)
with open(BASE_DIR/"NIST_Task2_c.pkl","wb") as f: pickle.dump(gen_c, f)
print("[SUBMIT] Saved official files:", 
      BASE_DIR/"NIST_Task2_a.pkl", BASE_DIR/"NIST_Task2_b.pkl", BASE_DIR/"NIST_Task2_c.pkl")

def show_pairs(tag, real_np, gen_np, nrow=4):
    plt.figure(figsize=(7, 1.8*nrow))
    plt.suptitle(f"{tag}: Real vs Quilted", y=0.98)
    for i in range(nrow):
        r = real_np[RNG.integers(0, len(real_np))].mean(-1)
        g = gen_np[RNG.integers(0, len(gen_np))].mean(-1)
        ax1 = plt.subplot(nrow,2,2*i+1); ax1.imshow(r, cmap='gray', vmin=0, vmax=255); ax1.set_title('Real'); ax1.axis('off')
        ax2 = plt.subplot(nrow,2,2*i+2); ax2.imshow(g, cmap='gray', vmin=0, vmax=255); ax2.set_title('Generated'); ax2.axis('off')
    plt.tight_layout(); plt.show()

show_pairs("a", REAL['a'], gen_a, nrow=4)
show_pairs("b", REAL['b'], gen_b, nrow=4)
show_pairs("c", REAL['c'], gen_c, nrow=4)

from skimage.metrics import structural_similarity as ssim
from math import log10

def _to_gray01(x):
    x = x.astype(np.float32)/255.0
    if x.ndim==3: x = x.mean(-1)
    return x

def psnr(a,b):
    a=_to_gray01(a); b=_to_gray01(b)
    mse = np.mean((a-b)**2)+1e-12
    return 10*log10(1.0/mse)

def ssim_batch(real, gen, K=100):
    idx = RNG.choice(len(real), size=min(K, len(real), len(gen)), replace=False)
    vals=[]
    for k in range(len(idx)):
        i = idx[k]; j = k % len(gen)
        vals.append(ssim(_to_gray01(real[i]), _to_gray01(gen[j]), data_range=1.0))
    return float(np.mean(vals))

def report(tag, real_np, gen_np):
    s = ssim_batch(real_np, gen_np, K=100)
    p = psnr(real_np[RNG.integers(0,len(real_np))], gen_np[RNG.integers(0,len(gen_np))])
    print(f"[METRIC] {tag}: SSIM≈{s:.4f} | PSNR≈{p:.2f} dB")

report("a", REAL['a'], gen_a)
report("b", REAL['b'], gen_b)
report("c", REAL['c'], gen_c)


# %%

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE = Path(os.getcwd())
random.seed(42); np.random.seed(42); torch.manual_seed(42)

GEN_CANDIDATES = {
    "a": ["NIST_Task2_a_quilted_v2.pkl", "NIST_Task2_a_post_v2.pkl", "NIST_Task2_a_calib.pkl", "NIST_Task2_a.pkl"],
    "b": ["NIST_Task2_b_quilted_v2.pkl", "NIST_Task2_b_post_v2.pkl", "NIST_Task2_b_calib.pkl", "NIST_Task2_b.pkl"],
    "c": ["NIST_Task2_c_quilted_v2.pkl", "NIST_Task2_c_post_v2.pkl", "NIST_Task2_c_calib.pkl", "NIST_Task2_c.pkl"],
}
REAL_PKLS = ["labeled_training_set.pkl", "unlabeled_training_set.pkl"]

def _load_first_existing(cands):
    for name in cands:
        p = BASE / name
        if p.exists():
            with open(p, "rb") as f:
                arr = pickle.load(f)
            if isinstance(arr, list):
                arr = np.stack(arr, 0)
            print(f"[LOAD GEN] {name} | dtype={arr.dtype} shape={arr.shape} min={arr.min()} max={arr.max()}")
            return arr
    raise FileNotFoundError(f"None of {cands} found in {BASE}")

def _collect_real_by_cond(cond_key: str):
    imgs = []
    for fname in REAL_PKLS:
        p = BASE / fname
        if not p.exists():
            continue
        d = pickle.load(open(p, "rb"))
        for _, items in d.items():
            for it in items:
                for k, im in it["images"].items():
                    if k.lower().endswith(cond_key.lower()):
                        x = im
                        if x.dtype != np.uint8:
                            x = (x if x.max() > 1.5 else x*255.0).astype(np.uint8)
                        imgs.append(x)
    if not imgs:
        raise RuntimeError(f"No real images for condition '{cond_key}' found in {REAL_PKLS}")
    arr = np.stack(imgs, 0).astype(np.uint8)
    print(f"[LOAD REAL] cond={cond_key} | {arr.shape}, min={arr.min()} max={arr.max()}")
    return arr

try:
    GEN, REAL 
except NameError:
    GEN = {k: _load_first_existing(GEN_CANDIDATES[k]) for k in ["a","b","c"]}
    REAL = {k: _collect_real_by_cond(k) for k in ["a","b","c"]}

for k in ["a","b","c"]:
    if GEN[k].dtype != np.uint8: GEN[k] = GEN[k].astype(np.uint8)
    if REAL[k].dtype != np.uint8: REAL[k] = REAL[k].astype(np.uint8)

def show_and_save_grid(arr: np.ndarray, title: str, out_png: Path, nrow=10, ncol=10):
    N, H, W, C = arr.shape
    K = min(N, nrow*ncol)
    plt.figure(figsize=(ncol*2.0, nrow*1.2))
    for i in range(K):
        ax = plt.subplot(nrow, ncol, i+1)
        ax.imshow(arr[i])
        ax.axis("off")
    plt.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig(out_png, dpi=180)
    plt.show()
    print(f"[GRID] Saved → {out_png}")

show_and_save_grid(GEN["a"], "Generated (a) — 100 tiles", BASE/"NIST_Task2_a_grid.png")
show_and_save_grid(GEN["b"], "Generated (b) — 100 tiles", BASE/"NIST_Task2_b_grid.png")
show_and_save_grid(GEN["c"], "Generated (c) — 100 tiles", BASE/"NIST_Task2_c_grid.png")

def to_tensor_bchw_uint8(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x.astype(np.float32)/255.0).permute(0,3,1,2).contiguous()

def resize_bilinear(x: torch.Tensor, size=(299,299)) -> torch.Tensor:
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


try:
    from torchvision.models import inception_v3, Inception_V3_Weights
    _INCEPT_OK = True
except Exception as e:
    print("[WARN] torchvision InceptionV3 not available:", e)
    _INCEPT_OK = False

@torch.no_grad()
def inception_features(imgs_01: torch.Tensor, batch=32) -> torch.Tensor:

    assert _INCEPT_OK, "InceptionV3 not available"
    try:

        weights = getattr(Inception_V3_Weights, "DEFAULT", Inception_V3_Weights.IMAGENET1K_V1)
    except Exception:
        weights = Inception_V3_Weights.IMAGENET1K_V1

    model = inception_v3(weights=weights).to(device).eval()

    DEFAULT_MEAN = [0.485, 0.456, 0.406]
    DEFAULT_STD  = [0.229, 0.224, 0.225]
    mean_list = DEFAULT_MEAN
    std_list  = DEFAULT_STD
    try:
        meta = getattr(weights, "meta", None)
        if isinstance(meta, dict):
            mean_list = meta.get("mean", DEFAULT_MEAN)
            std_list  = meta.get("std",  DEFAULT_STD)
        else:

            try:
                t = weights.transforms()
      
                mean_list, std_list = DEFAULT_MEAN, DEFAULT_STD
            except Exception:
                mean_list, std_list = DEFAULT_MEAN, DEFAULT_STD
    except Exception:
        mean_list, std_list = DEFAULT_MEAN, DEFAULT_STD

    mean = torch.tensor(mean_list, device=device).view(1,3,1,1)
    std  = torch.tensor(std_list,  device=device).view(1,3,1,1)


    def _feats(m, x):
        x = m.Conv2d_1a_3x3(x); x = m.Conv2d_2a_3x3(x); x = m.Conv2d_2b_3x3(x); x = F.max_pool2d(x,3,2)
        x = m.Conv2d_3b_1x1(x); x = m.Conv2d_4a_3x3(x); x = F.max_pool2d(x,3,2)
        x = m.Mixed_5b(x); x = m.Mixed_5c(x); x = m.Mixed_5d(x)
        x = m.Mixed_6a(x); x = m.Mixed_6b(x); x = m.Mixed_6c(x); x = m.Mixed_6d(x); x = m.Mixed_6e(x)
        x = m.Mixed_7a(x); x = m.Mixed_7b(x); x = m.Mixed_7c(x)
        x = F.adaptive_avg_pool2d(x, (1,1))
        return torch.flatten(x, 1)

    feats = []
    for i in range(0, imgs_01.size(0), batch):
        b = imgs_01[i:i+batch].to(device, non_blocking=True)
        b = resize_bilinear(b, (299,299))
        b = (b - mean)/std
        z = _feats(model, b)
        feats.append(z.detach().cpu())
    return torch.cat(feats, 0)

def _cov(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(0, keepdims=True)
    return (x.T @ x) / (len(x)-1)

def _sqrtm(mat: np.ndarray) -> np.ndarray:
    try:
        from scipy.linalg import sqrtm
        out = sqrtm(mat)
        if np.iscomplexobj(out): out = out.real
        return out
    except Exception:
        s = 0.5*(mat+mat.T)
        w, V = np.linalg.eigh(s)
        w = np.clip(w, 0, None)
        return (V * np.sqrt(w)) @ V.T

def fid_from_feats(f_real: np.ndarray, f_fake: np.ndarray) -> float:
    mu_r, mu_f = f_real.mean(0), f_fake.mean(0)
    cov_r, cov_f = _cov(f_real), _cov(f_fake)
    diff = mu_r - mu_f
    covmean = _sqrtm(cov_r @ cov_f)
    return float(diff @ diff + np.trace(cov_r + cov_f - 2*covmean))


try:
    import lpips as _lp
    _LPIPS_MODE = "lpips"
    _lp_net = _lp.LPIPS(net='alex').to(device).eval()
except Exception as e:
    print("[WARN] 'lpips' not found, using VGG16 feature-L2 fallback:", e)
    from torchvision.models import vgg16, VGG16_Weights
    _LPIPS_MODE = "vgg-fallback"
    _vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].to(device).eval()
    for p in _vgg.parameters(): p.requires_grad_(False)

@torch.no_grad()
def lpips_pairwise(x: torch.Tensor, y: torch.Tensor, batch=16) -> float:
    if _LPIPS_MODE == "lpips":
        def norm01_to_m11(t): return t*2-1
        scores = []
        for i in range(0, x.size(0), batch):
            a = norm01_to_m11(x[i:i+batch]).to(device)
            b = norm01_to_m11(y[i:i+batch]).to(device)
            s = _lp_net(a, b)  # (B,1,1,1)
            scores.append(s.view(-1).detach().cpu())
        return float(torch.cat(scores,0).mean().item())
    else:
        
        mean = torch.tensor([0.485,0.456,0.406], device=device).view(1,3,1,1)
        std  = torch.tensor([0.229,0.224,0.225], device=device).view(1,3,1,1)
        scores = []
        for i in range(0, x.size(0), batch):
            a = resize_bilinear(x[i:i+batch].to(device)); a = (a-mean)/std
            b = resize_bilinear(y[i:i+batch].to(device)); b = (b-mean)/std
            fa = _vgg(a); fb = _vgg(b)
            s = F.mse_loss(fa, fb, reduction='none').mean(dim=(1,2,3))
            scores.append(s.detach().cpu())
        return float(torch.cat(scores,0).mean().item())

def lpips_set(gen_01: torch.Tensor, real_01: torch.Tensor, max_pairs=100):
    k = min(gen_01.size(0), real_01.size(0), max_pairs)
    idx_g = torch.randperm(gen_01.size(0))[:k]
    idx_r = torch.randperm(real_01.size(0))[:k]
    return lpips_pairwise(gen_01[idx_g], real_01[idx_r])


def to_tensor_bchw_uint8(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x.astype(np.float32)/255.0).permute(0,3,1,2).contiguous()

@torch.no_grad()
def evaluate_fid_lpips(gen_np: np.ndarray, real_np: np.ndarray, name: str):
    g = to_tensor_bchw_uint8(gen_np)
    r = to_tensor_bchw_uint8(real_np)
    if not _INCEPT_OK:
        raise RuntimeError("InceptionV3 unavailable: cannot compute FID.")
    f_real = inception_features(r, batch=32).numpy()
    f_fake = inception_features(g, batch=32).numpy()
    fid = fid_from_feats(f_real, f_fake)
    lp = lpips_set(g, r, max_pairs=100)
    print(f"[EVAL][{name}] FID={fid:.2f} | LPIPS={lp:.4f}")
    return fid, lp

fid_a, lp_a = evaluate_fid_lpips(GEN["a"], REAL["a"], "a")
fid_b, lp_b = evaluate_fid_lpips(GEN["b"], REAL["b"], "b")
fid_c, lp_c = evaluate_fid_lpips(GEN["c"], REAL["c"], "c")

print("\n================ Summary ================")
print(f"a: FID={fid_a:.2f} | LPIPS={lp_a:.4f}")
print(f"b: FID={fid_b:.2f} | LPIPS={lp_b:.4f}")
print(f"c: FID={fid_c:.2f} | LPIPS={lp_c:.4f}")
with open(BASE / "task2_fid_lpips.txt", "w", encoding="utf-8") as f:
    f.write(f"a: FID={fid_a:.2f} | LPIPS={lp_a:.4f}\n")
    f.write(f"b: FID={fid_b:.2f} | LPIPS={lp_b:.4f}\n")
    f.write(f"c: FID={fid_c:.2f} | LPIPS={lp_c:.4f}\n")
print(f"[SAVE] Metrics → {BASE/'task2_fid_lpips.txt'}")

