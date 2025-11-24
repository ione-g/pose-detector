# stgcn_mediapipe33.py
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from utils.dataset_windows import build_adj, POSE_EDGES

def get_adj():
    A = build_adj(J=33, edges=POSE_EDGES)        # [33,33]
    A = torch.from_numpy(A)                      # fixed graph
    return A

class STGCNBlock(nn.Module):
    def __init__(self, Cin, Cout, A):
        super().__init__()
        self.register_buffer("A", A)  # [J,J]
        self.theta = nn.Conv2d(Cin, Cout, kernel_size=(1,1))
        self.temporal = nn.Conv2d(Cout, Cout, kernel_size=(9,1), padding=(4,0))
        self.bn = nn.BatchNorm2d(Cout)

    def forward(self, x):  # x: [B,C,T,J]
        # Spatial graph conv: X * A over joints
        xs = torch.einsum("bctj,jk->bctk", x, self.A)     # [B,C,T,J]
        xs = self.theta(xs)                               # [B,Cout,T,J]
        xt = self.temporal(xs)                            # temporal conv
        return F.relu(self.bn(xt))

class STGCN33(nn.Module):
    def __init__(self, n_classes, Cin=3):
        super().__init__()
        A = get_adj().float()
        self.g1 = STGCNBlock(Cin, 64, A)
        self.g2 = STGCNBlock(64, 64, A)
        self.g3 = STGCNBlock(64, 96, A)
        self.g4 = STGCNBlock(96,128, A)
        self.head = nn.Sequential(
            nn.AdaptiveMaxPool2d((1,None)),   # pool over time -> [B,128,1,J]
            nn.Flatten(1),                    # -> [B,128*J]
            nn.Dropout(0.25),
            nn.Linear(128*33, n_classes)
        )

    def forward(self, x):  # [B,3,T,J]
        h = self.g1(x)
        h = self.g2(h)
        h = self.g3(h)
        h = self.g4(h)
        return self.head(h)
