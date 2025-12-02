import argparse
import os
import sys
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from utils.dataset_windows import PoseWindows
from nn.stgcn import STGCN33


def csv_label_summary(csv_path: str):
    import csv

    counts = Counter()
    items = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            path = row[0].strip()
            label = row[1].strip() if len(row) > 1 else ""
            counts[label] += 1
            items.append((path, label))
    return counts, items


def train_one_epoch(model, loader, optimizer, device, criterion, grad_clip: float | None = 2.0):
    model.train()
    total = 0
    correct = 0
    running_loss = 0.0

    for batch in loader:
        if len(batch) == 2:
            x, y = batch
        else:
            x, y, _ = batch

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        batch_size = y.size(0)
        running_loss += float(loss.item()) * batch_size
        total += batch_size
        correct += int((logits.argmax(1) == y).sum().item())

    acc = correct / total if total else 0.0
    avg_loss = running_loss / total if total else 0.0
    return acc, avg_loss


@torch.no_grad()
def eval_once(model, loader, device, criterion):
    model.eval()
    total = 0
    correct = 0
    running_loss = 0.0
    y_true = []
    y_pred = []

    for batch in loader:
        if len(batch) == 2:
            x, y = batch
        else:
            x, y, _ = batch

        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        batch_size = y.size(0)
        running_loss += float(loss.item()) * batch_size
        total += batch_size

        preds = logits.argmax(1)
        correct += int((preds == y).sum().item())

        y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    acc = correct / total if total else 0.0
    avg_loss = running_loss / total if total else 0.0
    return acc, avg_loss, y_true, y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--classes", nargs="+", required=False, help="Optional list of classes in training order")
    parser.add_argument("--checkpoint-out", default="checkpoints/stgcn_best.pth")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--win", type=int, default=96)
    parser.add_argument("--hop", type=int, default=48)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--center", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    args = parser.parse_args()

    device = torch.device(args.device)

    train_counts, train_items = csv_label_summary(args.train_csv)
    val_counts, val_items = csv_label_summary(args.val_csv)

    print(f"Train CSV: {args.train_csv} -> {len(train_items)} rows, {len(train_counts)} unique labels")
    print("Train label counts:", dict(train_counts))
    print(f"Val CSV: {args.val_csv} -> {len(val_items)} rows, {len(val_counts)} unique labels")
    print("Val label counts:", dict(val_counts))

    if args.classes:
        classes = args.classes
    else:
        classes = sorted(set(train_counts.keys()) | set(val_counts.keys()))
        print("No --classes provided. Using classes from CSV (sorted):", classes)

    missing_train = set(train_counts.keys()) - set(classes)
    missing_val = set(val_counts.keys()) - set(classes)

    if missing_train:
        print("WARNING: train CSV contains labels not in provided classes:", missing_train)
    if missing_val:
        print("WARNING: val CSV contains labels not in provided classes:", missing_val)

    train_paths = {p for p, _ in train_items}
    val_paths = {p for p, _ in val_items}
    overlap = train_paths & val_paths
    if overlap:
        print("WARNING: train/val overlap (first 10):", list(overlap)[:10])

    print("Building datasets ...")
    train_ds = PoseWindows(
        args.train_csv,
        class_names=classes,
        win=args.win,
        hop=args.hop,
        augment=args.augment,
        center=args.center,
    )
    val_ds = PoseWindows(
        args.val_csv,
        class_names=classes,
        win=args.win,
        hop=args.hop,
        augment=False,
        center=args.center,
    )
    print("Train windows:", len(train_ds))
    print("Val windows:", len(val_ds))

    class_counts = {c: train_counts.get(c, 0) for c in classes}
    raw_weights = np.array([1.0 / max(1, class_counts[c]) for c in classes], dtype=np.float32)
    norm_weights = raw_weights / raw_weights.sum() * len(classes)
    weights = torch.tensor(norm_weights, dtype=torch.float32, device=device)
    print("Computed class weights:", weights.detach().cpu().numpy())

    model = STGCN33(n_classes=len(classes), Cin=3)
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    best_val = -1.0
    best_train = -1.0
    best_ckpt = args.checkpoint_out
    no_improve = 0
    y_true = []
    y_pred = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_acc, train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            criterion=criterion,
            grad_clip=args.grad_clip,
        )
        val_acc, val_loss, y_true, y_pred = eval_once(
            model,
            val_loader,
            device,
            criterion=criterion,
        )

        ckpt_dir = os.path.dirname(best_ckpt)
        if val_acc > best_val or (val_acc == best_val and train_acc > best_train):
            best_val = val_acc
            best_train = train_acc
            no_improve = 0
            if ckpt_dir:
                os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "classes": classes,
                },
                best_ckpt,
            )
            print(f"  ↳ saved best to {best_ckpt} (val_acc={val_acc:.3f})")
        else:
            no_improve += 1

        scheduler.step(val_acc)

        dt = time.time() - t0
        print(
            f"epoch {epoch:02d} | "
            f"train_acc {train_acc:.3f} | train_loss {train_loss:.4f} | "
            f"val_acc {val_acc:.3f} | val_loss {val_loss:.4f} | time {dt:.1f}s"
        )

        if no_improve >= args.patience:
            print("Early stopping due to no improvement.")
            break

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
