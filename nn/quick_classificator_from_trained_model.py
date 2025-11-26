import argparse
import sys
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# Try your project's dataset and model imports
try:
    from utils.dataset_windows import PoseWindows
except Exception:
    PoseWindows = None
try:
    from nn.stgcn import STGCN33
except Exception:
    # allow running if invoked directly from project root
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from nn.stgcn import STGCN33

# Fallback simple dataset (when utils.dataset_windows isn't available)
class SimplePoseDataset(Dataset):
    def __init__(self, csv_path: str, classes: list):
        self.rows = []
        self.classes = classes
        with open(csv_path, 'r') as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                # CSV format: path,label
                parts = [p.strip() for p in ln.split(',') if p.strip()]
                if len(parts) < 2:
                    continue
                npz_path, label = parts[0], parts[1]
                self.rows.append((npz_path, label))
        self.label2idx = {c: i for i, c in enumerate(classes)}

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        npz_path, label = self.rows[idx]
        npz = np.load(npz_path)
        xyz = npz["poses"][:, :, :3]  # [T, 33, 3]
        # convert to model shape: [3, T, 33] -> model expects [N, C, T, V]
        tensor = torch.tensor(xyz, dtype=torch.float32).permute(2, 0, 1)
        return tensor, torch.tensor(self.label2idx[label], dtype=torch.long)

def evaluate(model, loader, device):
    model.eval()
    ys, preds = [], []
    total, correct = 0, 0
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(1)
            ys.extend(y.cpu().tolist())
            preds.extend(pred.cpu().tolist())
            total += y.size(0)
            correct += (pred == y).sum().item()
    acc = correct / total if total else 0.0
    return acc, ys, preds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test-csv', required=True, help='CSV file with path,label per line')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--classes', nargs='+', required=True, help='List of classes in same order as training')
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--no-report', action='store_true', help='Do not print sklearn report')
    args = ap.parse_args()

    device = torch.device(args.device)

    # Dataset loader
    if PoseWindows is not None:
        dataset = PoseWindows(args.test_csv, classes=args.classes, mode='test')  # adjust if PoseWindows signature differs
    else:
        dataset = SimplePoseDataset(args.test_csv, args.classes)
    loader = DataLoader(dataset, batch_size=args.bs, shuffle=False, num_workers=2)

    # Model
    model = STGCN33(n_classes=len(args.classes))
    ckpt = torch.load(args.checkpoint, map_location=device)
    # support both state_dict and full dict
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        sd = ckpt['state_dict']
    elif isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        sd = ckpt['model_state_dict']
    else:
        sd = ckpt
    # rename keys if necessary
    try:
        model.load_state_dict(sd)
    except RuntimeError:
        # try keys with "module." prefix removal
        new_sd = { (k.replace('module.', '') if k.startswith('module.') else k): v for k,v in sd.items() }
        model.load_state_dict(new_sd)

    model.to(device)

    acc, ys, preds = evaluate(model, loader, device)
    print(f"Accuracy: {acc:.4f} ({sum([1 for a,b in zip(ys,preds) if a==b])}/{len(ys)})")

    if not args.no_report:
        try:
            from sklearn.metrics import classification_report, confusion_matrix
            print("\nClassification Report:\n", classification_report(ys, preds, target_names=args.classes, zero_division=0))
            print("Confusion Matrix:\n", confusion_matrix(ys, preds))
        except Exception:
            print("sklearn not available, skipping detailed report.")

if __name__ == "__main__":
    main()