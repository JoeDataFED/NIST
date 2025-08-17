#%% ================================================================
# Task 1 — Data: Importing libraries, setting random seeds, reading data, visualization and preprocessing tools
#==================================================================
from __future__ import annotations
import os
import json
import math
import random
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


try:
    from scipy import ndimage as ndi  
except Exception:
    ndi = None


try:
    import torch
    import torch.backends.cudnn as cudnn
except Exception:
    torch, cudnn = None, None

# --------------------------- config ---------------------------
CONFIG = {
    "seed": 42,
    # Which input should be used?：'a'、'b'、'c'、'ab'、'bc'、'ac'、'abc'
    "use_conditions": "abc",
    # Normalization: Independently count by condition mean/std 
    "per_condition_norm": True,
    # Save/read file names statistically
    "norm_stats_file": "norm_stats_task1.json",
    # Data file name
    "labeled_pkl": "labeled_training_set.pkl",
    "unlabeled_pkl": "unlabeled_training_set.pkl",
    "test_pkl": "test_set.pkl",
}

# ---------------------- Random Seeds and Determinism ----------------------
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import os
        os.environ["PYTHONHASHSEED"] = str(seed)
    except Exception:
        pass
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if cudnn is not None:
            cudnn.deterministic = True
            cudnn.benchmark = False

seed_everything(CONFIG["seed"])

# --------------------------- path ---------------------------
def get_base_dir() -> Path:

    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path(os.getcwd()).resolve()

BASE_DIR = get_base_dir()

# --------------------------- I/O ---------------------------
def load_pickle(path: Path) -> Dict:
    with open(path, "rb") as f:
        return pickle.load(f)

def safe_load_dataset(filename: str) -> Optional[Dict]:
    p = BASE_DIR / filename
    if not p.exists():
        print(f" Data file not found {p}")
        return None
    print(f" read：{p}")
    data = load_pickle(p)
    assert isinstance(data, dict), "The outermost layer of the data should be dict:{'part01': [...], ...}"
    return data

LABELED = safe_load_dataset(CONFIG["labeled_pkl"])
UNLABELED = safe_load_dataset(CONFIG["unlabeled_pkl"])
TESTSET = safe_load_dataset(CONFIG["test_pkl"])

# ---------------------- Dictionary structure and expansion ----------------------
def _key_endswith(images: Dict[str, np.ndarray], suffix: str) -> str:

    for k in images.keys():
        if k.lower().endswith(suffix.lower()):
            return k
    raise KeyError(f"No key ending with '{suffix}' , existing key: {list(images.keys())}")

def extract_triplet(images: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[str,str,str]]:
    ka = _key_endswith(images, "a")
    kb = _key_endswith(images, "b")
    kc = _key_endswith(images, "c")
    return images[ka], images[kb], images[kc], (ka, kb, kc)

def flatten_items(dataset_dict: Dict) -> List[Dict]:
    """Expand {part: [items]} into a sample list; Each item contains a/b/c and available tags"""
    items: List[Dict] = []
    for part, arr in (dataset_dict or {}).items():
        for it in arr:
            xa, xb, xc, keys = extract_triplet(it["images"])
            rec = {
                "part": part,
                "layer_id": str(it.get("layer_id", "")),
                "keys": keys,           # ('Axxxxa','Axxxxb','Axxxxc')
                "xa": xa, "xb": xb, "xc": xc,  # H x W x 3
                "spot": it.get("spot_label", None),     # H x W
                "streak": it.get("streak_label", None)  # H x W
            }
            items.append(rec)
    return items

LABELED_ITEMS  = flatten_items(LABELED)  if LABELED  else []
UNLABELED_ITEMS= flatten_items(UNLABELED)if UNLABELED else []
TEST_ITEMS     = flatten_items(TESTSET)  if TESTSET  else []

print(f" Labeled:  {len(LABELED_ITEMS)} samples")
print(f" Unlabeled:{len(UNLABELED_ITEMS)} samples")
print(f" Test:     {len(TEST_ITEMS)} samples")

# ---------------------- Data consistency check ----------------------
def check_shapes(items: List[Dict]) -> None:
    hws, cs = set(), set()
    has_spot = has_streak = 0
    for it in items:
        for k in ("xa","xb","xc"):
            arr = it[k]
            assert arr.ndim == 3, f"{k} It must be HxWxC, obtained{arr.shape}"
            hws.add(arr.shape[:2])
            cs.add(arr.shape[2])
        if it.get("spot") is not None:
            has_spot += 1
            assert it["spot"].shape == it["xa"].shape[:2], "spot_label 尺寸不匹配"
        if it.get("streak") is not None:
            has_streak += 1
            assert it["streak"].shape == it["xa"].shape[:2], "streak_label 尺寸不匹配"
    print(f"[CHECK] ROI 尺寸集合: {hws}，通道数集合: {cs}；Samples containing streaks {has_spot}，Samples containing streaks {has_streak}")

if LABELED_ITEMS:
    check_shapes(LABELED_ITEMS)

# ---------------------- normalization----------------------
def _accumulate_stats(img: np.ndarray, s: Dict[str, np.ndarray]) -> None:
    # img: HxWx3, uint8/float
    x = img.astype(np.float32)
    if x.max() > 1.5:  #  0-1
        x = x / 255.0
    s["sum"]  += x.sum(axis=(0,1))
    s["sumsq"]+= (x**2).sum(axis=(0,1))
    s["n"]    += x.shape[0]*x.shape[1]

def compute_norm_stats(items_labeled: List[Dict], items_unlabeled: List[Dict],
                       per_condition: bool = True) -> Dict[str, Dict[str, List[float]]]:
    """return {'a':{'mean':[3],'std':[3]}, 'b':..., 'c':...} or {'all':{...}}"""
    if per_condition:
        keys = ("a","b","c")
        stats = {k: {"sum":np.zeros(3, np.float64),
                     "sumsq":np.zeros(3, np.float64),
                     "n":0} for k in keys}
    else:
        keys = ("all",)
        stats = {"all":{"sum":np.zeros(3, np.float64),
                        "sumsq":np.zeros(3, np.float64),
                        "n":0}}

    def add_item(it: Dict):
        if per_condition:
            _accumulate_stats(it["xa"], stats["a"])
            _accumulate_stats(it["xb"], stats["b"])
            _accumulate_stats(it["xc"], stats["c"])
        else:
            for k in ("xa","xb","xc"):
                _accumulate_stats(it[k], stats["all"])

    for it in items_labeled:
        add_item(it)
    for it in items_unlabeled:
        add_item(it)

    out = {}
    for k, s in stats.items():
        n = max(int(s["n"]), 1)
        mean = (s["sum"]/n).astype(float)
        var  = (s["sumsq"]/n - mean**2).clip(min=1e-12)
        std  = np.sqrt(var).astype(float)
        out[k] = {"mean": mean.tolist(), "std": std.tolist()}
    return out

def save_norm_stats(stats: Dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

def load_norm_stats(path: Path) -> Optional[Dict]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

stats_path = BASE_DIR / CONFIG["norm_stats_file"]
norm_stats = load_norm_stats(stats_path)
if norm_stats is None and (LABELED_ITEMS or UNLABELED_ITEMS):
    norm_stats = compute_norm_stats(LABELED_ITEMS, UNLABELED_ITEMS, CONFIG["per_condition_norm"])
    save_norm_stats(norm_stats, stats_path)
    print(f" The normalized parameters have been statistically analyzed and saved → {stats_path}")
elif norm_stats is not None:
    print(f"The normalized parameters have been loaded ← {stats_path}")

def normalize_img(img: np.ndarray, cond: str, stats: Dict) -> np.ndarray:
    """Perform (x-mean)/std normalization according to the condition (a/b/c or all); Return float32"""
    if img.dtype != np.float32:
        x = img.astype(np.float32)
        if x.max() > 1.5:
            x /= 255.0
    else:
        x = img
    key = cond if cond in stats else "all"
    mean = np.array(stats[key]["mean"], dtype=np.float32)
    std  = np.array(stats[key]["std"],  dtype=np.float32)
    std  = np.where(std < 1e-6, 1.0, std)
    x = (x - mean) / std
    return x

def make_input_tensor(xa: np.ndarray, xb: np.ndarray, xc: np.ndarray,
                      use: str = "abc", stats: Optional[Dict] = None) -> np.ndarray:
    """
   Early fusion: Return H x W x C_total (C_total ∈ {3,6,9})
If stats are provided, normalize by condition.
    """
    parts = []
    if "a" in use:
        parts.append(normalize_img(xa, "a", stats) if stats else xa.astype(np.float32)/255.0)
    if "b" in use:
        parts.append(normalize_img(xb, "b", stats) if stats else xb.astype(np.float32)/255.0)
    if "c" in use:
        parts.append(normalize_img(xc, "c", stats) if stats else xc.astype(np.float32)/255.0)
    x = np.concatenate(parts, axis=2)  # H x W x (3*len(parts))
    return x

# ---------------------- Assessment Assistance (IoU ----------------------
def to_bool_mask(m: np.ndarray) -> np.ndarray:
    if m.dtype == bool:
        return m
    return (m.astype(np.int32) > 0)

def iou_binary(pred: np.ndarray, gt: np.ndarray) -> float:
    p = to_bool_mask(pred)
    g = to_bool_mask(gt)
    inter = np.logical_and(p, g).sum(dtype=np.int64)
    union = np.logical_or(p, g).sum(dtype=np.int64)
    if union == 0:
        return 1.0
    return float(inter) / float(union)

# ---------------------- visual  ----------------------
def overlay_masks(rgb: np.ndarray,
                  streak: Optional[np.ndarray]=None,
                  spot: Optional[np.ndarray]=None,
                  alpha: float=0.45) -> np.ndarray:
    """
    Cover the RGB image with two types of masks:
Streak uses red (R+ B), Spot uses yellow (R+G).
    """
    img = rgb.copy().astype(np.float32)
    if img.max() > 1.5:
        img /= 255.0

    over = img.copy()
    if streak is not None:
        m = to_bool_mask(streak)[..., None].astype(np.float32)
        color = np.array([1.0, 0.0, 1.0], dtype=np.float32)  # magenta
        over = over*(1-m*alpha) + color*(m*alpha)
    if spot is not None:
        m = to_bool_mask(spot)[..., None].astype(np.float32)
        color = np.array([1.0, 1.0, 0.0], dtype=np.float32)  # yellow
        over = over*(1-m*alpha) + color*(m*alpha)
    over = np.clip(over, 0.0, 1.0)
    return (over*255.0).astype(np.uint8)

def visualize_item(it: Dict, stats: Optional[Dict]=None, use_conditions: str="abc") -> None:
    """
    Display: Original image a/b/c, normalized overlay, and Overlay mask
    """
    xa, xb, xc = it["xa"], it["xb"], it["xc"]
    H, W, _ = xa.shape


    x_stack = make_input_tensor(xa, xb, xc, use=use_conditions, stats=stats)

    vis_after = x_stack[..., :3]
    vis_after = (vis_after - vis_after.min()) / (vis_after.max() - vis_after.min() + 1e-6)
    vis_after = (vis_after*255).astype(np.uint8)

    # original a/b/c（clip 到 0-255）
    def to_uint8(img):
        x = img
        if x.dtype != np.uint8:
            x = (x*255.0 if x.max() <= 1.5 else x).astype(np.uint8)
        return x

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    fig.suptitle(f"Part {it['part']} | Layer {it['layer_id']} | keys={it['keys']}, use='{use_conditions}'", fontsize=12)

    axes[0,0].imshow(to_uint8(xa)); axes[0,0].set_title("Raw A"); axes[0,0].axis("off")
    axes[0,1].imshow(to_uint8(xb)); axes[0,1].set_title("Raw B"); axes[0,1].axis("off")
    axes[0,2].imshow(to_uint8(xc)); axes[0,2].set_title("Raw C"); axes[0,2].axis("off")


    axes[1,0].imshow(overlay_masks(to_uint8(xa), it.get("streak"), it.get("spot")))
    axes[1,0].set_title("Overlay on A (streak:magenta, spot:yellow)"); axes[1,0].axis("off")


    axes[1,1].imshow(vis_after); axes[1,1].set_title("Normalized view (first 3 chs)"); axes[1,1].axis("off")

    if it.get("streak") is not None or it.get("spot") is not None:
        canvas = np.zeros((H, W, 3), np.uint8)
        if it.get("streak") is not None:
            canvas[to_bool_mask(it["streak"])] = (255, 0, 255)
        if it.get("spot") is not None:

            overlap = to_bool_mask(it.get("streak", np.zeros((H,W),bool))) & to_bool_mask(it["spot"])
            canvas[to_bool_mask(it["spot"])] = (255, 255, 0)
            canvas[overlap] = (255, 255, 255)
        axes[1,2].imshow(canvas); axes[1,2].set_title("Masks only"); axes[1,2].axis("off")
    else:
        axes[1,2].axis("off"); axes[1,2].set_title("Masks only (N/A)")

    plt.tight_layout()
    plt.show()

# ---------------------- sanity check ----------------------
def mask_stats(items: List[Dict]) -> Dict[str, float]:

    if not items:
        return {"streak_px%": 0.0, "spot_px%": 0.0}
    total_px = 0
    streak_px = 0
    spot_px = 0
    for it in items:
        H, W, _ = it["xa"].shape
        total_px += H*W
        if it.get("streak") is not None:
            streak_px += to_bool_mask(it["streak"]).sum()
        if it.get("spot") is not None:
            spot_px += to_bool_mask(it["spot"]).sum()
    return {
        "streak_px%": 100.0*streak_px / max(total_px, 1),
        "spot_px%":   100.0*spot_px   / max(total_px, 1),
    }

if LABELED_ITEMS:
    pct = mask_stats(LABELED_ITEMS)
    print(f"label pixel ratiotreak {pct['streak_px%']:.4f}% | spot {pct['spot_px%']:.4f}%")

# ---------------------- Preview a sample  ----------------------
def preview_random(items: List[Dict], seed: int = 42, use_conditions: str="abc") -> None:
    if not items:
        print(" No visual samples (the list is empty)")
        return
    rng = random.Random(seed)
    it = rng.choice(items)
    visualize_item(it, stats=norm_stats, use_conditions=use_conditions)


if LABELED_ITEMS:
    preview_random(LABELED_ITEMS, seed=CONFIG["seed"], use_conditions=CONFIG["use_conditions"])
elif UNLABELED_ITEMS:
    preview_random(UNLABELED_ITEMS, seed=CONFIG["seed"], use_conditions=CONFIG["use_conditions"])
elif TEST_ITEMS:
    preview_random(TEST_ITEMS, seed=CONFIG["seed"], use_conditions=CONFIG["use_conditions"])
else:
    print(" There are no available data files. Confirm whether *.pkl has been placed in the current directory。")


# %%
#%% ================================================================
# Task 1 — Modeling preparation: Dataset, partitioning, Dataloader (including "oversampled samples with Spots")
#==================================================================
from typing import Any
from typing import List, Dict, Tuple, Optional
import random

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# -------- Dataset --------
class NISTTask1Dataset(Dataset):
    """
    For supervisory training/validation:
    Input: Perform early fusion and normalization based on CONFIG['use_conditions'] and norm_stats
- Tag: Output [streak, spot] (2, H, W) in channel order, type float32 in {0.,1.}
    """
    def __init__(self, items: List[Dict], use_conditions: str = "abc",
                 stats: Optional[Dict] = None, augment: bool = False):
        self.items = items
        self.use = use_conditions
        self.stats = stats
        self.augment = augment

    def __len__(self) -> int:
        return len(self.items)

    def _color_jitter(self, x: np.ndarray) -> np.ndarray:

        if not self.augment:
            return x
        # x 为 0-1 float
        b = 1.0 + (random.random()-0.5)*0.2   
        c = 1.0 + (random.random()-0.5)*0.4   
        x = x*b
        mean = x.mean(axis=(0,1), keepdims=True)
        x = (x-mean)*c + mean
        x = np.clip(x, 0.0, 1.0)
        return x

    def _hflip(self, x: np.ndarray, y: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.augment and random.random() < 0.5:
            x = np.ascontiguousarray(x[:, ::-1, :])
            if y is not None:
                y = np.ascontiguousarray(y[:, ::-1])
        return x, y

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]
        xa, xb, xc = it["xa"], it["xb"], it["xc"]

        # Early fusion + normalization (returns HxWx{3,6,9} float32)
        x = make_input_tensor(xa, xb, xc, use=self.use, stats=self.stats)
        # Color jitter (only on the first 3 channels)
        x = self._color_jitter((x - x.min()) / (x.max() - x.min() + 1e-6))


        if it.get("streak") is not None and it.get("spot") is not None:
            y_streak = (to_bool_mask(it["streak"]).astype(np.float32))
            y_spot   = (to_bool_mask(it["spot"]).astype(np.float32))
            y = np.stack([y_streak, y_spot], axis=0)  # (2,H,W) sequence: [streak, spot]
        else:
            y = None

        # Random horizontal flip
        x, y = self._hflip(x, y)

        # HWC -> CHW
        x = np.transpose(x, (2,0,1)).astype(np.float32)  # (C,H,W)

        sample = {
            "image": torch.from_numpy(x),                       # float32
            "mask":  torch.from_numpy(y) if y is not None else None,  # float32 or None
            "meta": {"part": it["part"], "layer_id": it["layer_id"],
                     "has_spot": bool(y is not None and y[1].any())}
        }
        return sample

# -------- Split train/val sets  --------
def split_train_val(items: List[Dict], val_ratio: float = 0.2, seed: int = 42):
    rng = random.Random(seed)
    with_spot, without_spot = [], []
    for it in items:
        has_spot = it.get("spot") is not None and bool(to_bool_mask(it["spot"]).any())
        (with_spot if has_spot else without_spot).append(it)

    def _split(lst):
        rng.shuffle(lst)
        n_val = max(1, int(len(lst)*val_ratio))
        return lst[n_val:], lst[:n_val]

    train1, val1 = _split(with_spot)
    train2, val2 = _split(without_spot)
    train_items = train1 + train2
    val_items   = val1 + val2
    rng.shuffle(train_items); rng.shuffle(val_items)

    print(f"[SPLIT] all sample {len(items)} → train {len(train_items)} / val {len(val_items)}")
    print(f"[SPLIT] Samples containing spot in the training set:{sum(bool(to_bool_mask(it['spot']).any()) for it in train_items)}")
    print(f"[SPLIT] Samples containing spot in the val set:{sum(bool(to_bool_mask(it['spot']).any()) for it in val_items)}")
    return train_items, val_items

_train_items, _val_items = split_train_val(LABELED_ITEMS, val_ratio=0.2, seed=CONFIG["seed"])

# -------- Compute class weights for imbalanced data --------
def compute_pos_weights(items: List[Dict]) -> torch.Tensor:
    # [streak, spot]
    tot = 0
    pos_streak = 0
    pos_spot = 0
    for it in items:
        H, W, _ = it["xa"].shape
        tot += H*W
        pos_streak += to_bool_mask(it["streak"]).sum()
        pos_spot   += to_bool_mask(it["spot"]).sum()
    neg_streak = tot - pos_streak
    neg_spot   = tot - pos_spot

    ws = max(1.0, float(neg_streak) / max(1, int(pos_streak)))
    wp = max(1.0, float(neg_spot)   / max(1, int(pos_spot)))

    ws = float(np.clip(ws, 1.0, 50.0))
    wp = float(np.clip(wp, 5.0, 50.0))  
    print(f"[CLASS WEIGHT] pos_weight(streak)≈{ws:.2f}, pos_weight(spot)≈{wp:.2f}")
    return torch.tensor([ws, wp], dtype=torch.float32)

POS_WEIGHTS = compute_pos_weights(_train_items)

# -------- Dataset / DataLoader --------
BATCH_SIZE = 16

train_ds = NISTTask1Dataset(_train_items, use_conditions=CONFIG["use_conditions"],
                            stats=norm_stats, augment=True)
val_ds   = NISTTask1Dataset(_val_items,   use_conditions=CONFIG["use_conditions"],
                            stats=norm_stats, augment=False)

# Oversample samples containing Spots
weights = []
for it in _train_items:
    w = 1.0
    if it["spot"] is not None and to_bool_mask(it["spot"]).any():
        w *= 4.0  
    weights.append(w)
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=True)


_b = next(iter(train_loader))
print(f"[DATALOADER] image batch: {_b['image'].shape}, mask batch: {_b['mask'].shape if _b['mask'] is not None else None}")
print(f"[DATALOADER] meta 示例: {_b['meta']['part'][:3]} | has_spot(前3)={_b['meta']['has_spot'][:3]}")

#%%
#%% ================================================================
# Task 1 — Model: Lightweight U-Net encoding and decoding + dual-decoding dock 

#==================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_bn_relu(in_ch, out_ch, k=3, s=1, p=1, d=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, dilation=d, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            conv_bn_relu(in_ch, out_ch, 3, 1, 1),
            conv_bn_relu(out_ch, out_ch, 3, 1, 1),
        )
    def forward(self, x):
        return self.block(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = conv_bn_relu(in_ch, out_ch, 3, 1, 1)
        self.conv2 = conv_bn_relu(out_ch, out_ch, 3, 1, 1)
    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class DualHeadUNet(nn.Module):
    """
   - Encoding: C -> 32 -> 64 -> 128 -> 256
    Decoding: Stepwise upsampling is carried out to obtain a 64-channel high-resolution feature F (H,W,64).
    - Double-headed:
    StreakHead: Incorporates dilated convolution to expand the receptive field, making it suitable for slender structures
    SpotHead: Regular 3x3, retains more details, suitable for small bright spots
    Output: B x 2 x H x W (Channel 0: streak, Channel 1: spot)
    """
    def __init__(self, in_ch: int):
        super().__init__()
        c1, c2, c3, c4 = 32, 64, 128, 256

        self.enc1 = EncoderBlock(in_ch, c1)             # -> 32
        self.enc2 = EncoderBlock(c1, c2)                # -> 64
        self.enc3 = EncoderBlock(c2, c3)                # -> 128
        self.enc4 = EncoderBlock(c3, c4)                # -> 256
        self.pool = nn.MaxPool2d(2,2)

        # bottleneck
        self.bot = nn.Sequential(
            conv_bn_relu(c4, c4, 3,1,1),
            conv_bn_relu(c4, c4, 3,1,1)
        )

        # decoder
        self.dec3 = DecoderBlock(c4 + c4, c3//2)        # up(bottleneck 256) + skip e4 256 -> out 64
        self.dec2 = DecoderBlock((c3//2) + c3, c2//2)   # 64 + 128 -> 32
        self.dec1 = DecoderBlock((c2//2) + c2, c1)      # 32 + 64  -> 32

       
        self.fuse1 = conv_bn_relu(c1 + c1, 64, 3, 1, 1) # 32 + 32 -> 64

        
        self.head_streak = nn.Sequential(
            conv_bn_relu(64, 64, 3,1,1),
            conv_bn_relu(64, 64, 3,1,2, d=2),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        self.head_spot = nn.Sequential(
            conv_bn_relu(64, 64, 3,1,1),
            conv_bn_relu(64, 64, 3,1,1),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        # encoder
        e1 = self.enc1(x)                 # H,W, 32
        e2 = self.enc2(self.pool(e1))     # H/2, 64
        e3 = self.enc3(self.pool(e2))     # H/4, 128
        e4 = self.enc4(self.pool(e3))     # H/8, 256

        b  = self.bot(self.pool(e4))      # H/16, 256

        # decoder
        d3 = self.dec3(b, e4)             # -> 64
        d2 = self.dec2(d3, e3)            # -> 32
        d1 = self.dec1(d2, e2)            # -> 32

        # fuse with e1
        feat = F.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        feat = torch.cat([feat, e1], dim=1)   # 32 + 32 = 64
        feat = self.fuse1(feat)               # -> 64

        out_streak = self.head_streak(feat)   # -> 1
        out_spot   = self.head_spot(feat)     # -> 1
        out = torch.cat([out_streak, out_spot], dim=1)  # (B,2,H,W)
        return out

# --------- Model sanity check ---------
C_IN = 3 * len(CONFIG["use_conditions"])  # 'abc' -> 9
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DualHeadUNet(in_ch=C_IN).to(device)

def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

print(f" DualHeadUNet(in_ch={C_IN}) parameters: {count_params(model)/1e6:.2f} M")
with torch.no_grad():
    xb = _b["image"][:2].to(device) # (B,C,H,W)
    yb = model(xb)
    print(f" input {tuple(xb.shape)} → output {tuple(yb.shape)} ([streak, spot])")
    print(f" Output logits statistics: min={yb.min().item():.3f}, max={yb.max().item():.3f}, mean={yb.mean().item():.3f}")

#%% ================================================================
# Task 1 — Training preparation: Loss function, evaluation metrics, optimizer and hyperparameters


#==================================================================
from dataclasses import dataclass

@dataclass
class TrainCfg:
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-2
    dice_weight: float = 1.0
    grad_clip: float = 1.0
    threshold_streak: float = 0.50
    threshold_spot: float   = 0.35 
    amp: bool = True

TCFG = TrainCfg()

# ---- Loss function ----
bce_loss = nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHTS.view(1,2,1,1).to(device))

def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6,
              weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    logits: (B,2,H,W), targets: (B,2,H,W) in {0,1}

    """
    probs = torch.sigmoid(logits)
    dims = (0,2,3)
    inter = (probs*targets).sum(dim=dims)
    denom = (probs+targets).sum(dim=dims)
    dice = (2*inter + eps) / (denom + eps)  # (2,)
    d = 1. - dice
    if weights is not None:
        d = d * weights.to(d.device)
    return d.mean()

# ---- Evaluation metrics ----
@torch.no_grad()
def eval_iou(model: nn.Module, loader: DataLoader, t_streak: float, t_spot: float, device) -> Dict[str, float]:
    model.eval()
    inter = torch.zeros(2, device=device)
    union = torch.zeros(2, device=device)
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)  # (B,2,H,W)
        logits = model(x)
        probs = torch.sigmoid(logits)
        preds = torch.zeros_like(probs)
        preds[:,0] = (probs[:,0] >= t_streak).float()
        preds[:,1] = (probs[:,1] >= t_spot).float()
        for c in range(2):
            p = preds[:,c]
            g = y[:,c]
            inter[c] += (p*g).sum()
            union[c] += ((p+g)>0).float().sum()
    ious = (inter / torch.clamp(union, min=1)).detach().cpu().numpy().tolist()
    return {"IoU_streak": ious[0], "IoU_spot": ious[1], "mIoU": float(sum(ious)/2)}

# ---- Optimizer, scheduler, scaler ----
optimizer = torch.optim.AdamW(model.parameters(), lr=TCFG.lr, weight_decay=TCFG.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TCFG.epochs)
scaler = torch.cuda.amp.GradScaler(enabled=(TCFG.amp and device.type=="cuda"))

print(f"[TRAIN CFG] epochs={TCFG.epochs}, lr={TCFG.lr}, wd={TCFG.weight_decay}, "
      f"dice_w={TCFG.dice_weight}, thresh(streak/spot)={TCFG.threshold_streak}/{TCFG.threshold_spot}")
print(f"[TRAIN] Optimizer={optimizer.__class__.__name__}, Scheduler={scheduler.__class__.__name__}, AMP={scaler.is_enabled()}")

#%% ================================================================
# Task 1 — Supervised training cycle (baseline) : BCEWithLogits + Dice, model saving, visualization
#==================================================================
import time

save_dir = BASE_DIR / "checkpoints"
save_dir.mkdir(exist_ok=True, parents=True)
best_path = save_dir / "task1_dualhead_unet_best.pt"

def train_one_epoch(model, loader, optimizer, scaler, device, epoch: int):
    model.train()
    running_loss = 0.0
    n = 0
    for step, batch in enumerate(loader, 1):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            logits = model(x)
            # BCE
            loss_bce = bce_loss(logits, y)
            # Dice
            dice_w = (POS_WEIGHTS / POS_WEIGHTS.max()).to(device)  
            loss_dice = dice_loss(logits, y, weights=dice_w)
            loss = loss_bce + TCFG.dice_weight * loss_dice

        scaler.scale(loss).backward()
        # Gradient clipping
        if TCFG.grad_clip is not None and TCFG.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), TCFG.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        running_loss += float(loss.detach().cpu().item()) * x.size(0)
        n += x.size(0)

        if step % 20 == 0 or step == len(loader):
            print(f"[E{epoch:02d}] step {step:03d}/{len(loader)} | loss={running_loss/max(n,1):.4f}")

    return running_loss / max(n, 1)

@torch.no_grad()
def visualize_val_sample(model, loader, device, t_streak, t_spot, max_show: int = 1):
    model.eval()
    shown = 0
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["mask"].to(device)
        logits = model(x)
        probs = torch.sigmoid(logits)
        ps = probs[:,0].cpu().numpy()  # streak
        pb = probs[:,1].cpu().numpy()  # spot
        ys = y[:,0].cpu().numpy()
        yb = y[:,1].cpu().numpy()

       
        for i in range(min(max_show, x.size(0))):
            fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
            axes[0].imshow(ps[i], vmin=0, vmax=1); axes[0].set_title("streak prob"); axes[0].axis("off")
            axes[1].imshow(pb[i], vmin=0, vmax=1); axes[1].set_title("spot prob"); axes[1].axis("off")
            overlay = np.zeros((ps[i].shape[0], ps[i].shape[1], 3), np.uint8)
            overlay[ys[i]>0.5] = (255, 0, 255)
            overlay[yb[i]>0.5] = (255, 255, 0)
            axes[2].imshow(overlay); axes[2].set_title("GT masks"); axes[2].axis("off")
            plt.tight_layout(); plt.show()
            shown += 1
            if shown >= max_show:
                return

best_miou = -1.0
history = []  

print(f"[TRAIN] Start training and save the optimal model to:{best_path}")
for epoch in range(1, TCFG.epochs+1):
    t0 = time.time()
    train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch)
    scheduler.step()

    metrics = eval_iou(model, val_loader, TCFG.threshold_streak, TCFG.threshold_spot, device)  
    miou = metrics["mIoU"]
    history.append({"epoch": epoch, "train_loss": train_loss, **metrics})
    dt = time.time() - t0
    print(f"[VAL] epoch {epoch:02d} | mIoU={miou:.4f} | IoU_streak={metrics['IoU_streak']:.4f} | IoU_spot={metrics['IoU_spot']:.4f} | time={dt:.1f}s")
    # Every 3 epochs, visualize a sample from the validation set
    if epoch % 3 == 0:
        visualize_val_sample(model, val_loader, device, TCFG.threshold_streak, TCFG.threshold_spot, max_show=1)
    # Save the best model
    if miou > best_miou:
        best_miou = miou
        torch.save({"model": model.state_dict(),
                    "cfg": TCFG.__dict__,
                    "pos_weights": POS_WEIGHTS.cpu().numpy(),
                    "use_conditions": CONFIG["use_conditions"]}, best_path)
        print(f"[SAVE] new best mIoU={best_miou:.4f} → {best_path}")

print("[TRAIN] Training is over. The final round of eval", history[-1])

#%% ================================================================
# Task 1 —Load the optimal model and conduct the final evaluation on the validation set
#==================================================================
ckpt = torch.load(best_path, map_location=device)
model.load_state_dict(ckpt["model"])
final_metrics = eval_iou(model, val_loader, TCFG.threshold_streak, TCFG.threshold_spot, device)
print(f"[FINAL]Validation set: mIoU={final_metrics['mIoU']:.4f} | "
      f"IoU_streak={final_metrics['IoU_streak']:.4f} | IoU_spot={final_metrics['IoU_spot']:.4f}")

# Visualize 2 samples from the validation set
visualize_val_sample(model, val_loader, device, TCFG.threshold_streak, TCFG.threshold_spot, max_show=2)

#%% ================================================================
# Task 1 — Semi-supervised training preparation: Unlabeled dataset, dataloader, EMA helper, hyperparameters

#==================================================================
from copy import deepcopy
from dataclasses import dataclass

class UnlabeledPairDataset(Dataset):
    """
    For semi-supervised learning (Mean Teacher):
    Input: weakly augmented and strongly augmented image pairs (H,W,C) in 0-1 float
    No tags available
    
    """
    def __init__(self, items: List[Dict], use_conditions: str="abc", stats: Optional[Dict]=None):
        self.items = items
        self.use = use_conditions
        self.stats = stats

    def __len__(self) -> int:
        return len(self.items)

    def _jitter(self, x: np.ndarray, strength: str="weak") -> np.ndarray:

        if strength == "weak":
            b = 1.0 + (random.random()-0.5)*0.10  # ±5%
            c = 1.0 + (random.random()-0.5)*0.20  # ±10%
        else:
            b = 1.0 + (random.random()-0.5)*0.30  # ±15%
            c = 1.0 + (random.random()-0.5)*0.50  # ±25%
        x = x*b
        mean = x.mean(axis=(0,1), keepdims=True)
        x = (x-mean)*c + mean
        if strength == "strong":

            x = x + np.random.normal(0, 0.02, size=x.shape).astype(np.float32)
            if ndi is not None and random.random() < 0.25:
                
                sigma = 0.5 + random.random()*0.5
                x = ndi.gaussian_filter(x, sigma=(sigma, sigma, 0))
        x = np.clip(x, 0.0, 1.0)
        return x

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        it = self.items[idx]
        xa, xb, xc = it["xa"], it["xb"], it["xc"]
        x = make_input_tensor(xa, xb, xc, use=self.use, stats=self.stats)  # HxWxC, float
        x = (x - x.min()) / (x.max() - x.min() + 1e-6)

        xw = self._jitter(x, "weak")
        xs = self._jitter(x, "strong")
        # Random horizontal flip (50%)
        if random.random() < 0.5:
            xs = np.ascontiguousarray(xs[:, ::-1, :])

        # HWC -> CHW
        xw = np.transpose(xw, (2,0,1)).astype(np.float32)
        xs = np.transpose(xs, (2,0,1)).astype(np.float32)
        return {"image_w": torch.from_numpy(xw), "image_s": torch.from_numpy(xs)}

 
U_BATCH = 16
unsup_ds = UnlabeledPairDataset(UNLABELED_ITEMS, use_conditions=CONFIG["use_conditions"], stats=norm_stats)
unsup_loader = DataLoader(unsup_ds, batch_size=U_BATCH, shuffle=True, num_workers=0, pin_memory=True)

@dataclass
class SemiCfg:
    epochs: int = 100                 
    lambda_unsup: float = 1.0           
    rampup_epochs: int = 5            
    tau_streak: float = 0.90          
    tau_spot: float   = 0.70         
    ema_decay: float = 0.999          
    amp: bool = True                  

SEMI = SemiCfg()

class EMAHelper:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.ema = deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if k in msd:
                nv = msd[k]
                v.copy_(v * d + (1.0 - d) * nv)

#%% ================================================================
# Task 1 — Semi-supervised training cycle: Mean Teacher, pseudo-labeling, consistency loss, model saving, visualization
#==================================================================
from itertools import cycle

def bce_with_mask(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    loss = (loss * mask).sum() / torch.clamp(mask.sum(), min=1.0)
    return loss

def get_unsup_weight(epoch: int, semi: SemiCfg) -> float:
    w = semi.lambda_unsup * min(1.0, epoch / max(1, semi.rampup_epochs))
    return w

def to_hard_pseudo_and_mask(probs: torch.Tensor, tau_s: float, tau_p: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    probs: (B,2,H,W)   [streak, spot]

    """
    tau = torch.tensor([tau_s, tau_p], device=probs.device, dtype=probs.dtype).view(1,2,1,1)
    hard = (probs >= tau).float()
    mask = hard.clone()  
    return hard, mask

def make_dice_weight_for_semi(device) -> torch.Tensor:

    return torch.tensor([0.6, 1.0], device=device)

def train_mean_teacher(student: nn.Module,
                       teacher_helper: EMAHelper,
                       sup_loader: DataLoader,
                       unsup_loader: DataLoader,
                       pos_weights: torch.Tensor,
                       semi: SemiCfg,
                       base_cfg: TrainCfg,
                       device: torch.device):

    student = student
    teacher = teacher_helper.ema

    optimizer_mt = torch.optim.AdamW(student.parameters(), lr=base_cfg.lr, weight_decay=base_cfg.weight_decay)
    scheduler_mt = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_mt, T_max=semi.epochs)
    scaler_mt = torch.cuda.amp.GradScaler(enabled=(semi.amp and device.type=="cuda"))

    bce_sup = nn.BCEWithLogitsLoss(pos_weight=pos_weights.view(1,2,1,1).to(device))

    best_miou = -1.0
    best_path_semi = save_dir / "task1_semi_mean_teacher_best.pt"
    semi_history = [] 

    print("[SEMI] Start the Mean Teacher training")
    sup_iter = cycle(iter(sup_loader))
    for epoch in range(1, semi.epochs+1):
        student.train()
        teacher.eval()

        w_unsup = get_unsup_weight(epoch, semi)
        dice_w = make_dice_weight_for_semi(device)

        running_sup = 0.0
        running_unsup = 0.0
        n_seen = 0

        for step, u_batch in enumerate(unsup_loader, 1):
            s_batch = next(sup_iter)

            # ---- Supervised branch
            xs = s_batch["image"].to(device, non_blocking=True)
            ys = s_batch["mask"].to(device, non_blocking=True)

            # ---- Unsupervised branch
            xw = u_batch["image_w"].to(device, non_blocking=True)  # weak view -> teacher
            xu = u_batch["image_s"].to(device, non_blocking=True)  # strong view -> student

            optimizer_mt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler_mt.is_enabled()):
                # student 
                logits_sup = student(xs)
                loss_sup_bce = bce_sup(logits_sup, ys)
                loss_sup_dice = dice_loss(logits_sup, ys, weights=dice_w)
                loss_sup = loss_sup_bce + base_cfg.dice_weight * loss_sup_dice

                # teacher dummy tag
                with torch.no_grad():
                    logits_teacher = teacher(xw)
                    probs_teacher = torch.sigmoid(logits_teacher)
                    pseudo, mask = to_hard_pseudo_and_mask(
                        probs_teacher, tau_s=semi.tau_streak, tau_p=semi.tau_spot
                    )

                # student Consistency (for strong views)
                logits_unsup = student(xu)
                loss_unsup = bce_with_mask(logits_unsup, pseudo, mask)

                loss = loss_sup + w_unsup * loss_unsup

            scaler_mt.scale(loss).backward()
            if base_cfg.grad_clip is not None and base_cfg.grad_clip > 0:
                scaler_mt.unscale_(optimizer_mt)
                torch.nn.utils.clip_grad_norm_(student.parameters(), base_cfg.grad_clip)
            scaler_mt.step(optimizer_mt)
            scaler_mt.update()

            # EMA update
            teacher_helper.update(student)

            running_sup += float(loss_sup.detach().cpu().item()) * xs.size(0)
            running_unsup += float(loss_unsup.detach().cpu().item()) * xu.size(0)
            n_seen += xs.size(0)

            if step % 20 == 0 or step == len(unsup_loader):
                print(f"[SEMI E{epoch:02d}] step {step:03d}/{len(unsup_loader)} | sup={running_sup/max(n_seen,1):.4f} | unsup(w={w_unsup:.2f})={running_unsup/max(n_seen,1):.4f}")

        scheduler_mt.step()

        # ---- val（teacher & student）
        teacher.eval()
        mi_student = eval_iou(student, val_loader, TCFG.threshold_streak, TCFG.threshold_spot, device)
        mi_teacher = eval_iou(teacher, val_loader, TCFG.threshold_streak, TCFG.threshold_spot, device)
        ep_sup = running_sup / max(n_seen, 1)
        ep_unsup = running_unsup / max(n_seen, 1)
        semi_history.append({
            "epoch": epoch,
            "sup_loss": ep_sup,
            "unsup_loss": ep_unsup,
            "stud_mIoU": mi_student["mIoU"],
            "teach_mIoU": mi_teacher["mIoU"],
            "teach_IoU_streak": mi_teacher["IoU_streak"],
            "teach_IoU_spot": mi_teacher["IoU_spot"],
        })
        print(f"[SEMI VAL] epoch {epoch:02d} | SUP={ep_sup:.4f} | UNSUP={ep_unsup:.4f} | STUD mIoU={mi_student['mIoU']:.4f} | TEACH mIoU={mi_teacher['mIoU']:.4f}")

        if mi_teacher["mIoU"] > best_miou:
            best_miou = mi_teacher["mIoU"]
            torch.save({
                "student": student.state_dict(),
                "teacher": teacher.state_dict(),
                "semi_cfg": semi.__dict__,
                "train_cfg": base_cfg.__dict__,
                "pos_weights": pos_weights.cpu().numpy(),
                "use_conditions": CONFIG["use_conditions"]
            }, best_path_semi)
            print(f"[SEMI SAVE] 新最佳 mIoU={best_miou:.4f} → {best_path_semi}")

    print("[SEMI]Training is over.。")
    return semi_history

#%% ================================================================
# Task 1 — Run semi-supervised training and evaluate the best model on the validation set
#==================================================================
# Initialize student with the currently trained supervised 'model'
student_init = model 
ema_helper = EMAHelper(student_init, decay=SEMI.ema_decay)


semi_history = train_mean_teacher(student_init, ema_helper, train_loader, unsup_loader,
                                  POS_WEIGHTS, SEMI, TCFG, device)

# Load the best semi-supervised model (teacher) and evaluate on the validation set
best_semi_path = save_dir / "task1_semi_mean_teacher_best.pt"
if best_semi_path.exists():
    ck = torch.load(best_semi_path, map_location=device)
    model.load_state_dict(ck["teacher"])  # Load teacher weights
    semi_metrics = eval_iou(model, val_loader, TCFG.threshold_streak, TCFG.threshold_spot, device)
    print(f"[SEMI FINAL] TEACHER val: mIoU={semi_metrics['mIoU']:.4f} | "
          f"IoU_streak={semi_metrics['IoU_streak']:.4f} | IoU_spot={semi_metrics['IoU_spot']:.4f}")
    # 可视化
    visualize_val_sample(model, val_loader, device, TCFG.threshold_streak, TCFG.threshold_spot, max_show=2)
else:
    print(f"[SEMI] The optimal weight was not found: {best_semi_path}. Please check if the semi-supervised training is complete")

#%% ================================================================
# Visualization of training curves: loss & IoU varies with epoch (supervised + semi-supervised)

#==================================================================
def plot_supervised_history(hist: List[Dict], save_dir: Path):
    if not hist:
        print("[PLOT] no history。")
        return
    epochs = [h["epoch"] for h in hist]
    loss = [h["train_loss"] for h in hist]
    miou = [h["mIoU"] for h in hist]
    iou_s = [h["IoU_streak"] for h in hist]
    iou_p = [h["IoU_spot"] for h in hist]

    plt.figure(figsize=(6,4))
    plt.plot(epochs, loss, label="train loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Supervised — Train Loss")
    plt.grid(True, ls="--", alpha=0.3); plt.legend()
    out1 = save_dir / "sup_train_loss.png"
    plt.tight_layout(); plt.savefig(out1, dpi=150); plt.show()
    print(f"[PLOT] Supervised training loss curve saving: {out1}")

    plt.figure(figsize=(6,4))
    plt.plot(epochs, miou, label="mIoU")
    plt.plot(epochs, iou_s, label="IoU_streak")
    plt.plot(epochs, iou_p, label="IoU_spot")
    plt.xlabel("epoch"); plt.ylabel("IoU"); plt.title("Supervised — IoU")
    plt.grid(True, ls="--", alpha=0.3); plt.legend()
    out2 = save_dir / "sup_val_iou.png"
    plt.tight_layout(); plt.savefig(out2, dpi=150); plt.show()
    print(f"[PLOT] Supervise and verify the preservation of the IoU curve{out2}")

def plot_semi_history(hist: List[Dict], save_dir: Path):
    if not hist:
        print("No semi-supervised training history can be mapped.")
        return
    epochs = [h["epoch"] for h in hist]
    sup_loss = [h["sup_loss"] for h in hist]
    unsup_loss = [h["unsup_loss"] for h in hist]
    teach_miou = [h["teach_mIoU"] for h in hist]
    teach_iou_s = [h["teach_IoU_streak"] for h in hist]
    teach_iou_p = [h["teach_IoU_spot"] for h in hist]

    plt.figure(figsize=(6,4))
    plt.plot(epochs, sup_loss, label="sup loss")
    plt.plot(epochs, unsup_loss, label="unsup loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Semi — Loss (per epoch avg)")
    plt.grid(True, ls="--", alpha=0.3); plt.legend()
    out1 = save_dir / "semi_loss.png"
    plt.tight_layout(); plt.savefig(out1, dpi=150); plt.show()
    print(f"[PLOT] Semi-supervised loss curve saving: {out1}")

    plt.figure(figsize=(6,4))
    plt.plot(epochs, teach_miou, label="teacher mIoU")
    plt.plot(epochs, teach_iou_s, label="teacher IoU_streak")
    plt.plot(epochs, teach_iou_p, label="teacher IoU_spot")
    plt.xlabel("epoch"); plt.ylabel("IoU"); plt.title("Semi — Teacher IoU")
    plt.grid(True, ls="--", alpha=0.3); plt.legend()
    out2 = save_dir / "semi_teacher_iou.png"
    plt.tight_layout(); plt.savefig(out2, dpi=150); plt.show()
    print(f"[PLOT] Semi-supervised Teacher IoU curve saving:{out2}")

plot_supervised_history(history, save_dir)
plot_semi_history(semi_history, save_dir)

#%% ================================================================
# Generate test set submission file: Write the mask to the test set dictionary (Output: streak/spot masks)
# Logic
# 1) Prioritize loading the optimal weights of semi-supervised teachers; If it does not exist, use the supervised optimal weight;
# 2) Infer each item in the TESTSET and generate a binary mask based on the threshold;
# 3) Write the mask into the corresponding item of the TESTSET: key names 'streak_label' and 'spot_label';
# 4) Save as submission_task1.pkl.
#==================================================================
def load_best_model_for_submission(model: nn.Module, device):
    semi_ckpt = save_dir / "task1_semi_mean_teacher_best.pt"
    sup_ckpt = save_dir / "task1_dualhead_unet_best.pt"
    loaded = False
    if semi_ckpt.exists():
        ck = torch.load(semi_ckpt, map_location=device)
        model.load_state_dict(ck["teacher"])
        print(f"[SUBMIT] Use semi-supervised Teacher weights:{semi_ckpt}")
        loaded = True
    elif sup_ckpt.exists():
        ck = torch.load(sup_ckpt, map_location=device)
        model.load_state_dict(ck["model"])
        print(f"[SUBMIT] Use supervised optimal weights{sup_ckpt}")
        loaded = True
    else:
        print("[SUBMIT] No trained weights were found. The current model parameters in memory will be used.")
    model.eval()
    return model, loaded

def load_best_thresholds(default_ts: float, default_tp: float) -> Tuple[float, float]:
    t_s, t_p = default_ts, default_tp
    cand_paths = [
        BASE_DIR / "best_thresholds_task1.json",
        save_dir / "best_thresholds_task1.json"
    ]
    for p in cand_paths:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    d = json.load(f)
                if "t_streak" in d and "t_spot" in d:
                    t_s, t_p = float(d["t_streak"]), float(d["t_spot"])
                elif "best" in d and isinstance(d["best"], dict):
                    bd = d["best"]
                    if "t_streak" in bd and "t_spot" in bd:
                        t_s, t_p = float(bd["t_streak"]), float(bd["t_spot"])
                print(f"[SUBMIT] t_streak={t_s:.2f}, t_spot={t_p:.2f} （来自 {p.name}）")
                return t_s, t_p
            except Exception as e:
                print(f"[SUBMIT] 读取阈值失败 {p}: {e}")
    print(f"[SUBMIT] t_streak={t_s:.2f}, t_spot={t_p:.2f}")
    return t_s, t_p

@torch.no_grad()
def infer_one_item(model: nn.Module, xa: np.ndarray, xb: np.ndarray, xc: np.ndarray,
                   use: str, stats: Optional[Dict], t_s: float, t_p: float) -> Tuple[np.ndarray, np.ndarray]:
    x = make_input_tensor(xa, xb, xc, use=use, stats=stats)  # HxWxC
    x = np.transpose(x, (2,0,1)).astype(np.float32)          # CxHxW
    xt = torch.from_numpy(x).unsqueeze(0).to(device)         # 1xCxHxW
    logits = model(xt)
    probs = torch.sigmoid(logits)[0].cpu().numpy()           # 2xHxW
    pred_streak = (probs[0] >= t_s).astype(np.uint8)
    pred_spot   = (probs[1] >= t_p).astype(np.uint8)
    return pred_streak, pred_spot

from copy import deepcopy as _deepcopy

submission = _deepcopy(TESTSET) if TESTSET is not None else None
if submission is None or len(TEST_ITEMS) == 0:
    print("[SUBMIT] no TESTSET, skip generating submission file.")
else:
    # 1) Load the best model
    model, _ = load_best_model_for_submission(model, device)

    # 2) Load thresholds
    t_s, t_p = load_best_thresholds(TCFG.threshold_streak, TCFG.threshold_spot)

    # 3) Infer and write to the dictionary
    total = 0
    for part, arr in submission.items():
        for it in arr:
            xa, xb, xc, _keys = extract_triplet(it["images"])
            ps, pp = infer_one_item(model, xa, xb, xc,
                                    use=CONFIG["use_conditions"], stats=norm_stats,
                                    t_s=t_s, t_p=t_p)
            # Write to the dictionary
            it["streak_label"] = ps.astype(np.uint8)
            it["spot_label"]   = pp.astype(np.uint8)
            total += 1
    out_pkl = BASE_DIR / "submission_task1.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(submission, f)
    print(f"[SUBMIT] The submitted file has been generated: {out_pkl}; a total of {total} samples have been written")
    # Visualize an example overlay
    if total > 0:
        any_part = next(iter(submission.keys()))
        sample0 = submission[any_part][0]
        xa, xb, xc, _ = extract_triplet(sample0["images"])
        vis = overlay_masks(
            (xa if xa.max() > 1.5 else (xa*255).astype(np.uint8)),
            sample0["streak_label"], sample0["spot_label"]
        )
        plt.figure(figsize=(5,5)); plt.imshow(vis); plt.axis("off"); plt.title("Submission overlay example"); plt.show()
#%%