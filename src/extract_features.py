# ONE-NOTEBOOK FEATURE EXTRACTION for the RSNA knee competition.
# Paste this whole file into ONE cell of a NEW Kaggle notebook (nothing else in it), attach the
# competition data, set the accelerator to GPU, then use Save Version -> "Save & Run All".
# It runs on Kaggle's servers (you can close the tab) and the files in /kaggle/working are kept
# as the version's Output when it finishes.
import os, sys, time, subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg", "gdcm", "opencv-python-headless"])

import numpy as np
import pandas as pd
import pydicom
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm

cv2.setNumThreads(0)  # avoid fighting with the DataLoader workers
DATA_ROOT = "/kaggle/input/competitions/rsna-knee-abnormality-detection"
OUT_DIR = "/kaggle/working/feats"
os.makedirs(OUT_DIR, exist_ok=True)
PLANES = ["Axial", "Coronal", "Sagittal"]
CENTRAL = (0.2, 0.8)     # use the middle 60% of slices
CHUNK = 250              # studies per saved chunk
N_WORKERS = min(4, os.cpu_count() or 1)

train_series = pd.read_csv(os.path.join(DATA_ROOT, "train_series.csv"))
groups = {sid: g for sid, g in train_series.groupby("StudyInstanceUID")}
all_ids = list(train_series["StudyInstanceUID"].unique())

# ---------- same helpers as in your notebook ----------
def normalize_image(img, lo_p=1, hi_p=99):
    lo, hi = np.percentile(img, lo_p), np.percentile(img, hi_p)
    if hi - lo < 1e-6:
        return np.zeros_like(img, dtype=np.float32)
    return ((np.clip(img, lo, hi) - lo) / (hi - lo)).astype(np.float32)

def load_resize_normalize(path, size=256):
    px = pydicom.dcmread(path).pixel_array.astype(np.float32)
    return normalize_image(cv2.resize(px, (size, size), interpolation=cv2.INTER_AREA))

def get_best_series_for_plane(study_rows, plane, preferred_fluid=1):
    rows = study_rows[study_rows["Anatomical_Plane"] == plane]
    if len(rows) == 0:
        return None
    pref = rows[rows["Fluid_Sensitive"] == preferred_fluid]
    if len(pref) > 0:
        return pref.iloc[0]["SeriesInstanceUID"]
    rest = rows[rows["Fluid_Sensitive"] != preferred_fluid]
    return rest.iloc[0]["SeriesInstanceUID"] if len(rest) > 0 else None

# ---------- DICOM decoding in parallel worker processes ----------
class SeriesDS(torch.utils.data.Dataset):
    def __init__(self, jobs):
        self.jobs = jobs                      # list of (study_idx, plane_idx, series_dir or None)
    def __len__(self):
        return len(self.jobs)
    def __getitem__(self, k):
        si, p, sdir = self.jobs[k]
        if sdir is None:
            return si, p, None
        try:
            files = sorted(os.listdir(sdir))
            lo = int(len(files) * CENTRAL[0])
            hi = max(int(len(files) * CENTRAL[1]), lo + 1)
            sl = np.stack([load_resize_normalize(os.path.join(sdir, f)) for f in files[lo:hi]])
            return si, p, torch.from_numpy(sl)
        except Exception as e:
            print(f"failed {sdir}: {e}", flush=True)
            return si, p, None

# ---------- frozen ResNet18 on GPU ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, "| workers:", N_WORKERS, flush=True)
backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
backbone.fc = nn.Identity()
backbone = backbone.to(device).eval()
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

@torch.no_grad()
def pooled_feature(slices, bs=32):
    feats = []
    for i in range(0, len(slices), bs):
        xb = slices[i:i + bs].float().unsqueeze(1).repeat(1, 3, 1, 1)
        feats.append(backbone(((xb - mean) / std).to(device)).cpu().numpy())
    f = np.concatenate(feats)
    return np.concatenate([f.mean(0), f.max(0)])      # 1024 = 512 mean + 512 max

# ---------- chunked, resumable extraction ----------
t_start = time.time()
done_studies = 0
for c in range(0, len(all_ids), CHUNK):
    path = f"{OUT_DIR}/chunk_{c:05d}.npz"
    if os.path.exists(path):
        continue
    ids = all_ids[c:c + CHUNK]
    jobs = []
    for si, sid in enumerate(ids):
        for p, plane in enumerate(PLANES):
            ser = get_best_series_for_plane(groups[sid], plane)
            jobs.append((si, p, None if ser is None else os.path.join(DATA_ROOT, "train_series", sid, ser)))

    arr = np.zeros((len(ids), len(PLANES), 1024), dtype=np.float32)
    ok = np.zeros((len(ids), len(PLANES)), dtype=bool)
    loader = torch.utils.data.DataLoader(SeriesDS(jobs), batch_size=None, shuffle=False,
                                         num_workers=N_WORKERS, prefetch_factor=2 if N_WORKERS else None)
    for si, p, sl in tqdm(loader, total=len(jobs), desc=f"chunk {c // CHUNK}"):
        if sl is not None:
            arr[int(si), int(p)] = pooled_feature(sl)
            ok[int(si), int(p)] = True
    np.savez(path, ids=np.array(ids), feats=arr, ok=ok)

    done_studies += len(ids)
    elapsed = time.time() - t_start
    remaining = len(all_ids) - (c + len(ids))
    print(f"chunk done | {elapsed / 60:.1f} min elapsed | ~{elapsed / done_studies * remaining / 60:.0f} min left", flush=True)

# ---------- merge ----------
parts = [np.load(f"{OUT_DIR}/{f}", allow_pickle=True) for f in sorted(os.listdir(OUT_DIR)) if f.startswith("chunk_")]
ids_all = np.concatenate([p["ids"] for p in parts])
feats_all = np.concatenate([p["feats"] for p in parts])
ok_all = np.concatenate([p["ok"] for p in parts])
np.savez("/kaggle/working/all_slice_feats.npz", ids=ids_all, feats=feats_all, ok=ok_all)
print("Merged:", feats_all.shape, "| planes failed/missing:", int((~ok_all).sum()), flush=True)
