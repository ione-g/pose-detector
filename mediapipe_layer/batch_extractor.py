# batch_pose_extract.py
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import your extractor module (must be in PYTHONPATH or same folder)
import extractor as mpe

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}

def is_video_file(p: Path) -> bool:
    return p.suffix.lower() in VIDEO_EXTS

def process_one(video_path: Path, out_dir: Path, params: dict) -> tuple[Path, bool, str]:
    """Run extractor for a single video. Returns (video_path, ok, msg)."""
    try:
        out_prefix = out_dir / video_path.stem
        out_prefix.parent.mkdir(parents=True, exist_ok=True)

        mpe.extract(
            video_path=str(video_path),
            out_prefix=str(out_prefix),
            use_world=params["use_world"],
            model_complexity=params["model_complexity"],
            min_det=params["min_det"],
            min_track=params["min_track"],
            smooth_ema_alpha=params["smooth_ema"],
            center=not params["no_center"],
            scale=not params["no_scale"],
            keep_missing_as_nan=params["save_missing_as_nan"],
            canon_orient=params["canon_orient"],
            canon_smooth=params["canon_smooth"],
            write_meta=False,  # <-- force NO META
        )
        return video_path, True, "ok"
    except Exception as e:
        return video_path, False, str(e)

def main():
    ap = argparse.ArgumentParser("Batch pose extraction over a directory (no meta).")
    ap.add_argument("--in-dir", required=True, type=str, help="Input directory with videos")
    ap.add_argument("--out-dir", required=True, type=str, help="Output directory for <stem>.npz")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing .npz")
    ap.add_argument("--workers", type=int, default=1, help="Parallel workers (threads)")
    # Mirror useful extractor params
    ap.add_argument("--use-world", action="store_true")
    ap.add_argument("--model-complexity", type=int, default=2, choices=[0, 1, 2])
    ap.add_argument("--min-det", type=float, default=0.6)
    ap.add_argument("--min-track", type=float, default=0.6)
    ap.add_argument("--smooth-ema", type=float, default=0.0)
    ap.add_argument("--no-center", action="store_true")
    ap.add_argument("--no-scale", action="store_true")
    ap.add_argument("--save-missing-as-nan", action="store_true")
    ap.add_argument("--canon-orient", action="store_true")
    ap.add_argument("--canon-smooth", type=float, default=0.2)
    args = ap.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Gather videos
    if args.recursive:
        videos = [p for p in in_dir.rglob("*") if p.is_file() and is_video_file(p)]
    else:
        videos = [p for p in in_dir.iterdir() if p.is_file() and is_video_file(p)]

    videos.sort()
    if not videos:
        print("[INFO] No videos found.")
        return

    # Filter by overwrite flag
    if not args.overwrite:
        keep = []
        for v in videos:
            out_npz = out_dir / f"{v.stem}.npz"
            if out_npz.exists():
                print(f"[SKIP] {v.name} (exists). Use --overwrite to redo.")
            else:
                keep.append(v)
        videos = keep
        if not videos:
            print("[INFO] Nothing to do.")
            return

    params = dict(
        use_world=args.use_world,
        model_complexity=args.model_complexity,
        min_det=args.min_det,
        min_track=args.min_track,
        smooth_ema=args.smooth_ema,
        no_center=args.no_center,
        no_scale=args.no_scale,
        save_missing_as_nan=args.save_missing_as_nan,
        canon_orient=args.canon_orient,
        canon_smooth=args.canon_smooth,
    )

    print(f"[RUN] {len(videos)} videos, workers={args.workers}")
    if args.workers <= 1:
        # sequential
        for v in videos:
            vp, ok, msg = process_one(v, out_dir, params)
            print(("[OK] " if ok else "[ERR] ") + vp.name, ("" if ok else f" — {msg}"))
    else:
        # threaded
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_one, v, out_dir, params): v for v in videos}
            for fut in as_completed(futs):
                v = futs[fut]
                ok = False
                msg = ""
                try:
                    _, ok, msg = fut.result()
                except Exception as e:
                    msg = str(e)
                print(("[OK] " if ok else "[ERR] ") + v.name, ("" if ok else f" — {msg}"))

if __name__ == "__main__":
    main()
