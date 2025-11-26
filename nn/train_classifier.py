import argparse
import os
import sys
import time
import json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Local imports
try:
    from utils.dataset_windows import PoseWindows, csv_label_summary  # csv_label_summary might not exist; fallback below
except Exception:
    # fallback: import PoseWindows directly if path setup differs
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from utils.dataset_windows import PoseWindows

try:
    from nn.stgcn import STGCN33
except Exception:
    # Ensure package import path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from nn.stgcn import STGCN33

# Provide csv_label_summary if not available from utils
def csv_label_summary(csv_path):
    cnt = Counter()
    items = []
    import csv
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            p = row[0].strip()
            lab = row[1].strip() if len(row) > 1 else ""
            cnt[lab] += 1
            items.append((p, lab))
    return cnt, items

def train_one_epoch(model, loader, opt, device, criterion=None, grad_clip=2.0):
    model.train()
    total, correct, running_loss = 0, 0, 0.0
    for batch in loader:
        if len(batch) == 2:
            x, y = batch
        else:
            x, y, _ = batch
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = criterion(logits, y) if criterion is not None else nn.CrossEntropyLoss()(logits, y)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        opt.step()
        running_loss += float(loss.item()) * y.size(0)
        total += y.size(0)
        correct += int((logits.argmax(1) == y).sum().item())
    acc = correct / total if total else 0.0
    avg_loss = running_loss / total if total else 0.0
    return acc, avg_loss

@torch.no_grad()
def eval_once(model, loader, device, criterion=None):
    model.eval()
    total, correct, running_loss = 0, 0, 0.0
    ys, preds = [], []
    for batch in loader:
        if len(batch) == 2:
            x, y = batch
        else:
            x, y, _ = batch
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        if criterion is not None:
            loss = criterion(logits, y)
            running_loss += float(loss.item()) * y.size(0)
        total += y.size(0)
        pred = logits.argmax(1)
        correct += int((pred == y).sum().item())
        ys.extend(y.cpu().tolist())
        preds.extend(pred.cpu().tolist())
    acc = correct / total if total else 0.0
    avg_loss = running_loss / total if total else 0.0
    return acc, avg_loss, ys, preds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-csv', required=True)
    ap.add_argument('--val-csv', required=True)
    ap.add_argument('--classes', nargs='+', required=False, help='Optional list of classes in training order')
    ap.add_argument('--checkpoint-out', default='checkpoints/stgcn_best.pth')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--win', type=int, default=96)
    ap.add_argument('--hop', type=int, default=48)
    ap.add_argument('--augment', action='store_true')
    ap.add_argument('--center', action='store_true')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--num-workers', type=int, default=0)
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--grad-clip', type=float, default=2.0)
    args = ap.parse_args()

    device = torch.device(args.device)

    # Summarize CSVs and classes
    train_cnt, train_items = csv_label_summary(args.train_csv)
    val_cnt, val_items = csv_label_summary(args.val_csv)
    print(f"Train CSV: {args.train_csv} -> {len(train_items)} rows, {len(train_cnt)} unique labels")
    print("Train label counts:", dict(train_cnt))
    print(f"Val CSV: {args.val_csv} -> {len(val_items)} rows, {len(val_cnt)} unique labels")
    print("Val label counts:", dict(val_cnt))

    if args.classes:
        classes = args.classes
    else:
        classes = sorted(set(list(train_cnt.keys()) + list(val_cnt.keys())))
        print("No --classes provided. Using classes from CSV (sorted):", classes)
    # warn if CSV labels not in classes
    missing_train = set(train_cnt.keys()) - set(classes)
    missing_val = set(val_cnt.keys()) - set(classes)
    if missing_train: print("WARNING: train CSV contains labels not in provided classes:", missing_train)
    if missing_val: print("WARNING: val CSV contains labels not in provided classes:", missing_val)

    # intersection check
    train_paths = set(p for p, _ in train_items)
    val_paths = set(p for p, _ in val_items)
    overlap = train_paths & val_paths
    if overlap:
        print("WARNING: train/val overlap (first 10):", list(overlap)[:10])

    # Datasets
    print("Building datasets ...")
    train_ds = PoseWindows(args.train_csv, class_names=classes, win=args.win, hop=args.hop, augment=True, center=args.center)
    val_ds = PoseWindows(args.val_csv, class_names=classes, win=args.win, hop=args.hop, augment=False, center=args.center)
    print("Train windows:", len(train_ds))
    print("Val windows:", len(val_ds))

    # Balanced weights
    class_counts = {c: train_cnt.get(c, 0) for c in classes}
    weights = np.array([1.0 / max(1, class_counts[c]) for c in classes], dtype=np.float32)
    weights = torch.tensor(weights / weights.sum() * len(classes), dtype=torch.float32).to(device)
    print("Computed class weights:", weights.cpu().numpy())

    # Model, criterion, optimizer, scheduler
    model = STGCN33(n_classes=len(classes), Cin=3)
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=4)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=args.num_workers)

    # Training loop
    best_val = -1.0
    best_ckpt = args.checkpoint_out
    no_improve = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_acc, train_loss = train_one_epoch(model, train_loader, opt, device, criterion=criterion, grad_clip=args.grad_clip)
        val_acc, val_loss, y_true, y_pred = eval_once(model, val_loader, device, criterion=criterion)
        if val_acc > best_val:
            best_val = val_acc
            no_improve = 0
            os.makedirs(os.path.dirname(best_ckpt), exist_ok=True)
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'val_acc': val_acc, 'classes': classes}, best_ckpt)
            print(f"  ↳ saved best to {best_ckpt} (val_acc={val_acc:.3f})")
        else:
            no_improve += 1
        scheduler.step(val_acc)
        print(f"epoch {epoch:02d} | train_acc {train_acc:.3f} | train_loss {train_loss:.4f} | val_acc {val_acc:.3f} | val_loss {val_loss:.4f} | time {time.time()-t0:.1f}s")
        if no_improve >= args.patience:
            print("Early stopping due to no improvement.")
            break

    # Final info and optional metrics
    try:
        from sklearn.metrics import classification_report, confusion_matrix
        print("\nFinal validation report:")
        print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))
        print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    except Exception:
        print("sklearn not available, skipping detailed report.")

    print("Training finished. Best val_acc:", best_val)
    print("Saved checkpoint:", best_ckpt)


if __name__ == "__main__":
    main()