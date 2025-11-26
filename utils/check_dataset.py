import argparse
import csv
import os
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

try:
    from utils.dataset_windows import load_npz_xyz, PoseWindows, make_windows
except Exception:
    # fallback: allow running when script is invoked as module
    from dataset_windows import load_npz_xyz, PoseWindows, make_windows

def csv_label_summary(csv_path):
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.reader(f):
            if not row or len(row) < 2:
                continue
            p = row[0].strip()
            lab = row[1].strip()
            rows.append((p, lab))
    cnt = Counter([lab for _, lab in rows])
    return rows, cnt

def check_npz(path, min_frames=1):
    try:
        d = np.load(path)
    except Exception as e:
        return False, f"np.load error: {e}"
    if 'poses' not in d:
        return False, "missing 'poses' key"
    poses = d['poses']
    if poses.ndim < 2:
        return False, f"bad poses shape {poses.shape}"
    if poses.shape[0] < min_frames:
        return False, f"too few frames {poses.shape[0]} < {min_frames}"
    return True, f"OK frames {poses.shape[0]}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='CSV path: path,label')
    ap.add_argument('--classes', nargs='+', help='Optional classes list passed into training')
    ap.add_argument('--win', type=int, default=96, help='Window size used for make_windows')
    ap.add_argument('--max-inspect', type=int, default=30, help='Max files to inspect')
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print("CSV not found:", csv_path)
        return 2

    rows, cnt = csv_label_summary(csv_path)
    print("CSV rows:", len(rows))
    print("Unique labels:", len(cnt), dict(cnt))

    if args.classes:
        missing_labels = sorted(set(cnt.keys()) - set(args.classes))
        if missing_labels:
            print("WARNING: CSV contains labels not in provided classes:", missing_labels)
        else:
            print("Classes from args cover CSV labels.")

    missing_files = []
    bad_npz = []
    paths_checked = 0
    for p, lab in rows:
        pth = Path(p)
        if not pth.exists():
            # try resolve relative to project root or csv location
            alt = (csv_path.parent / pth).resolve()
            if alt.exists():
                pth = alt
            else:
                missing_files.append(str(p))
                continue
        ok, msg = check_npz(str(pth), min_frames=max(1, args.win // 2))
        if not ok:
            bad_npz.append((str(pth), msg))
        paths_checked += 1
        if paths_checked >= args.max_inspect:
            break

    print("Checked files:", paths_checked)
    if missing_files:
        print("Missing files (first 20):", missing_files[:20])
    if bad_npz:
        print("Bad npz (first 20):", bad_npz[:20])

    # if PoseWindows available, summarise windows per clip
    try:
        ds = PoseWindows(str(csv_path), class_names=args.classes or sorted(cnt.keys()), win=args.win, hop=args.win//2, augment=False, center=True)
        print("PoseWindows created. Total windows:", len(ds))
        # windows per clip
        per_clip = defaultdict(int)
        for p,s,e,y in ds.items:
            per_clip[p] += 1
        counts = sorted(per_clip.values())
        if counts:
            print("Windows per clip: min,median,max ->", counts[0], counts[len(counts)//2], counts[-1])
            sample = list(per_clip.items())[:10]
            print("Sample windows per clip:", sample)
        else:
            print("No windows created.")
        # show which CSV rows were not represented in ds.items
        csv_paths = [p for p, _ in rows]
        ds_paths = set([p for p,_,_,_ in ds.items])
        missing_from_ds = [p for p in csv_paths if p not in ds_paths]
        if missing_from_ds:
            print("CSV paths not present in PoseWindows items (first 30):", missing_from_ds[:30])
            print("Count not included:", len(missing_from_ds))
        else:
            print("All CSV paths are included in PoseWindows items.")
        # windows per class id and name
        class_counts = defaultdict(int)
        for _, _, _, cid in ds.items:
            class_counts[cid] += 1
        if class_counts:
            print("Windows per class id (id:name:count):")
            for cid, cntv in sorted(class_counts.items()):
                name = (args.classes or sorted(set([lab for _,lab in rows])))[cid]
                print(" ", cid, name, cntv)
    except Exception as e:
        print("PoseWindows creation failed:", e)

    print("Done.")

if __name__ == "__main__":
    main()