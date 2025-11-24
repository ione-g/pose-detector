import argparse, os, json, time
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from utils.dataset_windows import PoseWindows
from .stgcn import STGCN33

def train_one_epoch(model, loader, opt, device):
    model.train()
    ce = nn.CrossEntropyLoss()
    total, correct, running = 0, 0, 0.0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = ce(logits, y)
        loss.backward()
        opt.step()
        running += loss.item() * y.size(0)
        total += y.size(0)
        correct += (logits.argmax(1)==y).sum().item()
    return correct/total, running/total

@torch.no_grad()
def eval_once(model, loader, device):
    model.eval()
    total, correct = 0, 0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        logits = model(x)
        total += y.size(0)
        correct += (logits.argmax(1)==y).sum().item()
    return correct/total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--val-csv", required=True)
    ap.add_argument("--classes", nargs="+", required=True,
                    help="List of class names, e.g. correct knees_in shallow")
    ap.add_argument("--win", type=int, default=96)
    ap.add_argument("--hop", type=int, default=48)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", type=str, default="checkpoints/stgcn33.pth")
    ap.add_argument("--augment", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = PoseWindows(args.train_csv, args.classes, win=args.win, hop=args.hop, augment=args.augment, center=True)
    val_ds   = PoseWindows(args.val_csv,   args.classes, win=args.win, hop=args.win, augment=False, center=True)
    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=max(64,args.bs), shuffle=False, num_workers=4, pin_memory=True)

    model = STGCN33(n_classes=len(args.classes)).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    best = 0.0
    for ep in range(1, args.epochs+1):
        tr_acc, tr_loss = train_one_epoch(model, train_dl, opt, device)
        va_acc = eval_once(model, val_dl, device)
        print(f"epoch {ep:02d} | train_acc {tr_acc:.3f} | train_loss {tr_loss:.4f} | val_acc {va_acc:.3f}")
        if va_acc > best:
            best = va_acc
            torch.save({"state_dict": model.state_dict(),
                        "classes": args.classes}, args.out)
            print(f"  ↳ saved best to {args.out} (val_acc={best:.3f})")

if __name__ == "__main__":
    main()
