import argparse
import sys
import os
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


try:
    from utils.dataset_windows import PoseWindows
except Exception:
    PoseWindows = None

try:
    from nn.stgcn import STGCN33
except Exception:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from nn.stgcn import STGCN33


class SimplePoseDataset(Dataset):
    def __init__(self, csv_path: str, classes: list[str]):
        self.rows = []
        self.classes = classes
        with open(csv_path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                parts = [p.strip() for p in ln.split(",")]
                if len(parts) < 2:
                    continue
                npz_path, label = parts[0], parts[1]
                self.rows.append((npz_path, label))
        self.label2idx = {c: i for i, c in enumerate(classes)}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        npz_path, label = self.rows[idx]
        npz = np.load(npz_path)
        xyz = npz["poses"][:, :, :3]
        tensor = torch.tensor(xyz, dtype=torch.float32).permute(2, 0, 1)
        return tensor, torch.tensor(self.label2idx[label], dtype=torch.long), npz_path


def evaluate(model, loader, device, dataset=None):
    model.eval()
    ys, preds, probs, paths = [], [], [], []
    global_index = 0

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 2:
                x, y = batch
                batch_paths = None
            else:
                x, y, batch_paths = batch

            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            p = torch.softmax(logits, dim=1)
            pred = logits.argmax(1)
            batch_size = y.size(0)

            ys.extend(y.cpu().tolist())
            preds.extend(pred.cpu().tolist())
            probs.extend(p.cpu().tolist())

            if batch_paths is None:
                if hasattr(dataset, "items"):
                    for i in range(batch_size):
                        paths.append(dataset.items[global_index + i][0])
                elif hasattr(dataset, "rows"):
                    for i in range(batch_size):
                        paths.append(dataset.rows[global_index + i][0])
                else:
                    for i in range(batch_size):
                        paths.append(str(global_index + i))
            else:
                paths.extend(batch_paths)

            global_index += batch_size

    return ys, preds, probs, paths


def load_classes_from_csv(path: str):
    cls = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rdr = csv.reader(f)
            first = next(rdr, None)
            if first is None:
                return []

            if len(first) == 1:
                cls.append(first[0].strip())
                for r in rdr:
                    if not r:
                        continue
                    cls.append(r[0].strip())
            else:
                candidates = [c.strip().lower() for c in first]
                if "class" in candidates:
                    idx = candidates.index("class")
                    for r in rdr:
                        if not r:
                            continue
                        cls.append(r[idx].strip())
                elif "label" in candidates:
                    idx = candidates.index("label")
                    for r in rdr:
                        if not r:
                            continue
                        cls.append(r[idx].strip())
                else:
                    all_rows = [first] + list(rdr)
                    for r in all_rows:
                        if not r or len(r) < 2:
                            continue
                        cls.append(r[1].strip())
    except Exception:
        return []

    seen = set()
    out = []
    for c in cls:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def remap_probabilities(probs, map_ckpt_to_args, num_classes_target):
    if map_ckpt_to_args is None:
        return probs
    remapped = []
    for pb in probs:
        pb = np.asarray(pb, dtype=np.float32)
        new_pb = np.zeros(num_classes_target, dtype=np.float32)
        for src_idx, target_idx in enumerate(map_ckpt_to_args):
            if src_idx < len(pb) and 0 <= target_idx < num_classes_target:
                new_pb[target_idx] = pb[src_idx]
        remapped.append(new_pb.tolist())
    return remapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-csv", required=True, help="CSV file with path,label per line")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--classes", nargs="+", required=False, help="List of classes in same order as training")
    ap.add_argument("--classes-csv", type=str, required=False, help="CSV with classes (one per line or header class/label)")
    ap.add_argument("--win", type=int, default=96)
    ap.add_argument("--hop", type=int, default=48)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--save-preds", type=str, default=None)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)

    classes = None
    if args.classes:
        classes = args.classes
    elif args.classes_csv:
        classes = load_classes_from_csv(args.classes_csv)
        if not classes:
            print("Warning: classes CSV provided but no classes loaded.")

    with open(args.test_csv, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    print("CSV lines (sample up to 10):", lines[:10])

    labels_in_csv = [ln.split(",")[-1].strip() for ln in lines]
    unique_labels = sorted(set(labels_in_csv))
    print("Unique CSV labels:", unique_labels)
    print("Provided classes (raw):", classes)

    if classes is None:
        classes = unique_labels
        print("No --classes provided; inferred classes from CSV:", classes)

    missing_labels = sorted(set(unique_labels) - set(classes))
    if missing_labels:
        print("Warning: labels in CSV not included in provided classes:", missing_labels)

    if PoseWindows is not None:
        dataset = PoseWindows(
            args.test_csv,
            class_names=classes,
            win=args.win,
            hop=args.hop,
            augment=False,
            center=True,
        )
    else:
        dataset = SimplePoseDataset(args.test_csv, classes)

    print("Dataset created:", type(dataset))
    if hasattr(dataset, "class_to_id"):
        print("class_to_id:", dataset.class_to_id)
    if hasattr(dataset, "items"):
        print("Num windows (items):", len(dataset.items))
        print("First items (sample):", dataset.items[:5])
    elif hasattr(dataset, "rows"):
        print("Num rows:", len(dataset.rows))
        print("First rows (sample):", dataset.rows[:5])
    print("Dataset length (windows):", len(dataset))

    missing_files = []
    for i in range(min(20, len(dataset))):
        if hasattr(dataset, "items"):
            p = dataset.items[i][0]
        elif hasattr(dataset, "rows"):
            p = dataset.rows[i][0]
        else:
            p = None
        if p and not os.path.exists(p):
            missing_files.append(p)

    if missing_files:
        print("Missing files (first 20 checked):", missing_files)

    if len(dataset) == 0:
        raise RuntimeError("Dataset length is 0, nothing to evaluate. Check CSV, classes and file paths.")

    loader = DataLoader(dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers)

    model = STGCN33(n_classes=len(classes))
    print("Model created:", model.__class__.__name__)

    last_linear = None
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            last_linear = (name, m)
    if last_linear is not None:
        print("Last linear:", last_linear[0], "out_features:", last_linear[1].out_features)
    else:
        print("Warning: couldn't find Linear layer in model to inspect final output shape.")

    ckpt = torch.load(args.checkpoint, map_location=device)
    print("Checkpoint loaded type:", type(ckpt))

    if isinstance(ckpt, dict):
        print("Checkpoint keys:", list(ckpt.keys()))

    ckpt_classes = None
    if isinstance(ckpt, dict) and "classes" in ckpt:
        ckpt_classes = ckpt["classes"]
        print("Saved classes in checkpoint:", ckpt_classes)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    else:
        sd = ckpt

    try:
        model.load_state_dict(sd)
        print("State dict loaded into model.")
    except RuntimeError:
        new_sd = {k.replace("module.", "") if k.startswith("module.") else k: v for k, v in sd.items()}
        model.load_state_dict(new_sd)
        print("State dict loaded after adjusting 'module.' prefix.")

    model.to(device)

    ys, preds, probs, paths = evaluate(model, loader, device, dataset=dataset)

    map_to_args = None
    if ckpt_classes and classes:
        if ckpt_classes != classes and set(ckpt_classes) == set(classes):
            map_to_args = [classes.index(c) for c in ckpt_classes]
            print("Built mapping from checkpoint class indices -> provided class indices.")
        elif ckpt_classes != classes:
            print("Warning: checkpoint classes and provided classes differ and are not the same set. Results may be incorrect.")

    mapped_probs = remap_probabilities(probs, map_to_args, len(classes))
    remapped_preds = [int(np.argmax(pb)) for pb in mapped_probs]

    total = len(ys)
    correct = sum(1 for a, b in zip(ys, remapped_preds) if a == b)
    acc = correct / total if total else 0.0
    print(f"Window-level Accuracy: {acc:.4f} ({correct}/{total})")

    if args.save_preds:
        with open(args.save_preds, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "true_id", "pred_id", "pred_prob"])
            for p, yt, pr, pb in zip(paths, ys, remapped_preds, mapped_probs):
                w.writerow([p, int(yt), int(pr), float(pb[int(pr)])])
        print("Saved per-window predictions to", args.save_preds)

    if args.aggregate:
        path_to_true = {}
        if hasattr(dataset, "items"):
            for path, s, e, lab in dataset.items:
                path_to_true[path] = lab
        elif hasattr(dataset, "rows"):
            for path, lab_name in dataset.rows:
                if lab_name in getattr(dataset, "label2idx", {}):
                    path_to_true[path] = dataset.label2idx[lab_name]
                else:
                    path_to_true[path] = None

        clip_votes = defaultdict(list)
        clip_probs_sum = defaultdict(lambda: np.zeros(len(classes), dtype=np.float32))

        for p, yt, pr, pb in zip(paths, ys, remapped_preds, mapped_probs):
            clip_votes[p].append(pr)
            clip_probs_sum[p] += np.array(pb, dtype=np.float32)

        clip_results = []
        correct_clips = 0
        total_clips = 0

        for p in clip_votes:
            votes = clip_votes[p]
            if not votes:
                continue
            avg_probs = clip_probs_sum[p] / max(1, len(votes))
            pred_id = int(np.argmax(avg_probs))
            true_id = path_to_true.get(p, None)
            if true_id is None:
                continue
            total_clips += 1
            if pred_id == true_id:
                correct_clips += 1
            clip_results.append((p, true_id, pred_id, avg_probs[pred_id]))

        clip_acc = (correct_clips / total_clips) if total_clips else 0.0
        print(f"Clip-level Accuracy (aggregate): {clip_acc:.4f} ({correct_clips}/{total_clips})")
        print("Sample clip predictions (first 10):")
        for p, t, pr, prob in clip_results[:10]:
            print(f" clip: {p} true: {classes[t]} pred: {classes[pr]} prob: {prob:.4f}")

    if not args.no_report:
        try:
            from sklearn.metrics import classification_report, confusion_matrix

            print(
                "\nClassification Report (window-level):\n",
                classification_report(ys, remapped_preds, target_names=classes, zero_division=0),
            )
            print("Confusion Matrix (window-level):\n", confusion_matrix(ys, remapped_preds))
        except Exception:
            print("sklearn not available, skipping detailed report.")


if __name__ == "__main__":
    main()
