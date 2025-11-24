import os, glob, csv, random
import numpy as np
import torch
from torch.utils.data import Dataset

POSE_EDGES = [
    (11,12),(23,24),(11,23),(12,24),
    (11,13),(13,15),(12,14),(14,16),
    (23,25),(25,27),(27,29),(29,31),
    (24,26),(26,28),(28,30),(30,32),
    (0,2),(2,3),(3,7),(0,5),(5,6),(6,8)
]

def build_adj(J=33, edges=POSE_EDGES):
    A = np.zeros((J,J), dtype=np.float32)
    for a,b in edges:
        A[a,b]=A[b,a]=1.0
    for i in range(J):
        A[i,i]=1.0
    D = np.diag(1.0 / (A.sum(1)+1e-6))
    return (D @ A).astype(np.float32)  # simple normalized adjacency

def make_windows(T, win=96, hop=48):
    idx=[]
    if T<=win:
        idx.append((0,win))
        return idx
    s=0
    while s+win<=T:
        idx.append((s,s+win)); s+=hop
    if not idx: idx=[(0,win)]
    return idx

def load_npz_xyz(path):
    d = np.load(path)
    poses = d["poses"]  # [T,33,4] -> use xyz only
    xyz = poses[..., :3].astype(np.float32)    # [T,33,3]
    # replace NaNs with last valid (simple)
    if np.isnan(xyz).any():
        last=None
        for t in range(xyz.shape[0]):
            if np.isfinite(xyz[t]).all():
                last=xyz[t].copy()
            elif last is not None:
                xyz[t]=last
            else:
                xyz[t]=0.0
    return xyz

class PoseWindows(Dataset):
    """
    CSV format (no header):
    path_to_npz,label_name
    ...
    """
    def __init__(self, csv_path, class_names, win=96, hop=48, augment=False, center=True):
        self.items=[]   # (path,start,end,label_id)
        self.class_to_id = {c:i for i,c in enumerate(class_names)}
        with open(csv_path, newline='', encoding='utf-8') as f:
            for row in csv.reader(f):
                if not row: continue
                p, lab = row[0], row[1]
                if lab not in self.class_to_id: continue
                xyz = load_npz_xyz(p)
                for s,e in make_windows(xyz.shape[0], win, hop):
                    self.items.append((p,s,e,self.class_to_id[lab]))
        self.win=win
        self.augment=augment
        self.center=center

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        p,s,e,y = self.items[i]
        xyz = load_npz_xyz(p)           # [T,33,3]
        seg = xyz[s:e]                  # [win,33,3]
        if seg.shape[0] < self.win:     # pad if needed
            pad = np.repeat(seg[-1:], self.win - seg.shape[0], axis=0)
            seg = np.concatenate([seg, pad], 0)
        # light centering (again, in case extractor didn't)
        if self.center:
            pelvis = seg[:, [23,24], :].mean(1, keepdims=True)  # [T,1,3]
            seg = seg - pelvis
        # augmentation
        if self.augment:
            # small gaussian noise
            seg += np.random.normal(0, 0.01, seg.shape).astype(np.float32)
            # random horizontal flip (swap left/right joints)
            if random.random()<0.5:
                # swap L/R indices for symmetric joints
                swap_pairs = [(11,12),(13,14),(15,16),(23,24),(25,26),(27,28),(29,30),(31,32)]
                for a,b in swap_pairs:
                    seg[:, [a,b], :] = seg[:, [b,a], :]
                seg[...,0] *= -1  # flip X
        # to [C=3, T, J=33]
        seg = np.transpose(seg, (2,0,1)).astype(np.float32)
        x = torch.from_numpy(seg)   # [3,win,33]
        y = torch.tensor(y, dtype=torch.long)
        return x,y
