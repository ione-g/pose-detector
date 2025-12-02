"""
mp_pose_extractor.py
--------------------
Extract 33-joint pose sequences from a single-view video using MediaPipe.
Outputs:
  - <out_prefix>.npz : poses[T,33,4], vis_mask[T,33] (float32)
  - <out_prefix>.json: metadata

Args:
  --video PATH              Input video file
  --out-prefix PATH         Output prefix (dir/prefix)
  --use-world               Use world landmarks (meter-scale) instead of image-normalized coords
  --model-complexity {0,1,2}  MediaPipe model size (2=most accurate)
  --min-det 0.6             Min detection confidence
  --min-track 0.6           Min tracking confidence
  --smooth-ema 0.0          EMA smoothing factor (0 disables)
  --no-center               Disable root-centering (pelvis)
  --no-scale                Disable scale normalization
  --save-missing-as-nan     Keep undetected frames as NaN rows (default). If false, repeats last.
"""

import json
import math
import argparse
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp


# --------- Utility: simple EMA smoother (per-joint, per-dim) ----------
def ema_smooth(arr: np.ndarray, alpha: float) -> np.ndarray:
    """
    Exponential moving average over time dimension.
    arr: [T, J, C]
    alpha in (0,1]; higher = stronger smoothing.
    """
    if not (0.0 < alpha <= 1.0):
        return arr
    out = arr.copy()
    T = out.shape[0]
    for t in range(1, T):
        # Only smooth finite entries; leave NaNs as-is
        mask = np.isfinite(out[t])
        out[t][mask] = alpha * out[t][mask] + (1.0 - alpha) * out[t - 1][mask]
    return out


# --------- Normalization: root-center + scale by torso/height proxy ----------
def normalize_sequence(
    poses: np.ndarray,
    center: bool = True,
    scale: bool = True,
    use_world: bool = False,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    poses: [T,33,4] (x,y,z,vis), may contain NaNs for missing frames.
    center: subtract pelvis (avg of L/R hips) from xyz.
    scale: divide by a robust body scale (shoulder-hip diagonal length).
    use_world: if True, we assume meter-scale; normalization still helps for model invariance.

    Returns modified poses (copy).
    """
    out = poses.copy()
    T, J, C = out.shape
    assert J == 33 and C >= 3

    # MediaPipe indices
    L_HIP, R_HIP = 23, 24
    L_SHOULDER, R_SHOULDER = 11, 12

    # Build a per-frame scale from hips/shoulders (robust to partial occlusion)
    def frame_scale(frame_xyz: np.ndarray) -> float:
        # Use diagonal between mid-shoulders and mid-hips as scale proxy
        ls = frame_xyz[L_SHOULDER]
        rs = frame_xyz[R_SHOULDER]
        lh = frame_xyz[L_HIP]
        rh = frame_xyz[R_HIP]
        if np.any(~np.isfinite([ls, rs, lh, rh])):
            return np.nan
        mid_sh = (ls + rs) / 2.0
        mid_hp = (lh + rh) / 2.0
        return float(np.linalg.norm(mid_sh - mid_hp) + eps)

    xyz = out[..., :3]  # [T,33,3]
    vis = out[..., 3]   # [T,33]

    for t in range(T):
        # Skip if entire frame is NaN
        if not np.isfinite(xyz[t]).any():
            continue

        # Centering by pelvis
        if center:
            hips = []
            if np.isfinite(xyz[t, L_HIP]).all():
                hips.append(xyz[t, L_HIP])
            if np.isfinite(xyz[t, R_HIP]).all():
                hips.append(xyz[t, R_HIP])
            if hips:
                pelvis = np.mean(np.stack(hips, axis=0), axis=0)  # [3]
                # subtract only from finite joints
                for j in range(xyz.shape[1]):
                    if np.isfinite(xyz[t, j]).all():
                        xyz[t, j] -= pelvis

        # Scaling
        if scale:
            s = frame_scale(xyz[t])
            if np.isfinite(s) and s > eps:
                xyz[t] /= s

    out[..., :3] = xyz
    # For visibility: keep as provided (0..1). Optionally clamp.
    out[..., 3] = np.clip(vis, 0.0, 1.0)
    return out


# --------- Filling missing frames (optional) ----------
def fill_missing_frames(seq: np.ndarray) -> np.ndarray:
    """
    Replace all-NaN frames by the last valid frame (forward fill).
    seq: [T,33,4]
    """
    out = seq.copy()
    last = None
    for t in range(out.shape[0]):
        if np.isfinite(out[t]).any():
            last = out[t].copy()
        else:
            if last is not None:
                out[t] = last
    return out

def _safe_norm(v, eps=1e-8):
    n = np.linalg.norm(v)
    return v / (n + eps), n

def _compute_body_axes(frame_xyz: np.ndarray) -> tuple | None:
    """
    Compute per-frame body axes from world coords.
    Returns (origin, x_axis, y_axis, z_axis) or None if not enough joints.
    origin = pelvis (mid-hips). Axes are unit and orthonormal, right-handed.
    """
    L_HIP, R_HIP = 23, 24
    L_SH,  R_SH  = 11, 12

    if not (np.isfinite(frame_xyz[L_HIP]).all() and
            np.isfinite(frame_xyz[R_HIP]).all() and
            np.isfinite(frame_xyz[L_SH]).all() and
            np.isfinite(frame_xyz[R_SH]).all()):
        return None

    lhip, rhip = frame_xyz[L_HIP], frame_xyz[R_HIP]
    lsh,  rsh  = frame_xyz[L_SH],  frame_xyz[R_SH]

    origin = 0.5 * (lhip + rhip)              # pelvis center
    up_raw = 0.5 * (lsh + rsh) - origin       # hips->shoulders

    # left->right using both shoulders and hips (more stable)
    lr1 = rsh - lsh
    lr2 = rhip - lhip
    lr_raw = 0.5 * (lr1 + lr2)

    # Normalize & build right-handed frame
    x_axis, nx = _safe_norm(lr_raw)           # left->right
    y_axis, ny = _safe_norm(up_raw)           # up
    # Make forward z = up × x (right-handed)
    z_axis = np.cross(y_axis, x_axis)
    z_axis, nz = _safe_norm(z_axis)
    # Re-orthonormalize x to ensure perfect ortho
    x_axis = np.cross(z_axis, y_axis)
    x_axis, _ = _safe_norm(x_axis)

    # Sanity: if any axis collapsed, bail
    if min(nx, ny, nz) < 1e-6 or not (np.isfinite(x_axis).all() and np.isfinite(y_axis).all() and np.isfinite(z_axis).all()):
        return None
    return origin, x_axis, y_axis, z_axis

def _ema_vec(prev: np.ndarray | None, cur: np.ndarray, alpha: float) -> np.ndarray:
    if prev is None or alpha <= 0.0:
        return cur
    return alpha * cur + (1.0 - alpha) * prev

def canonicalize_world_sequence(
    poses: np.ndarray,        # [T,33,4] (x,y,z,vis)
    vis_mask: np.ndarray,     # [T,33]
    ema_alpha: float = 0.2
) -> tuple[np.ndarray, list]:
    """
    Rotate world coords per frame to a canonical body frame.
    Returns (poses_rotated, per_frame_info), where per_frame_info[t] contains
    {'R': 3x3 rotation, 'origin': 3, 'ok': bool}.
    """
    out = poses.copy()
    xyz = out[..., :3]       # [T,33,3]
    T = xyz.shape[0]

    prev_axes = None  # (x_axis, y_axis, z_axis)
    last_valid = None # (origin, R)

    info = []
    for t in range(T):
        frame = xyz[t]
        axes = _compute_body_axes(frame)
        ok = axes is not None
        if ok:
            origin, x_axis, y_axis, z_axis = axes

            # Optional EMA smoothing of axes
            if ema_alpha > 0.0:
                if prev_axes is not None:
                    x_axis = _ema_vec(prev_axes[0], x_axis, ema_alpha)
                    y_axis = _ema_vec(prev_axes[1], y_axis, ema_alpha)
                    z_axis = _ema_vec(prev_axes[2], z_axis, ema_alpha)
                    # Re-orthonormalize after EMA
                    z_axis, _ = _safe_norm(np.cross(y_axis, x_axis))
                    x_axis, _ = _safe_norm(np.cross(z_axis, y_axis))
                    y_axis, _ = _safe_norm(y_axis)
                prev_axes = (x_axis, y_axis, z_axis)

            # Rotation that maps body axes -> canonical axes:
            # If body basis B has columns [x y z], then R = B^T maps p to canonical.
            B = np.stack([x_axis, y_axis, z_axis], axis=1)  # 3x3
            R = B.T
            # Apply: p' = R (p - origin)
            for j in range(33):
                if np.isfinite(frame[j]).all():
                    xyz[t, j] = (R @ (frame[j] - origin))
                else:
                    # keep NaNs
                    pass
            last_valid = (origin, R)
            info.append({"R": R, "origin": origin, "ok": True})
        else:
            # If we have a last valid transform, apply it to keep continuity
            if last_valid is not None:
                origin, R = last_valid
                for j in range(33):
                    if np.isfinite(frame[j]).all():
                        xyz[t, j] = (R @ (frame[j] - origin))
                info.append({"R": R, "origin": origin, "ok": False})
            else:
                info.append({"R": None, "origin": None, "ok": False})

    out[..., :3] = xyz
    return out, info

# --------- Main extractor ----------
def extract(
    video_path: str,
    out_prefix: str,
    use_world: bool = False,
    model_complexity: int = 2,
    min_det: float = 0.6,
    min_track: float = 0.6,
    smooth_ema_alpha: float = 0.0,
    center: bool = True,
    scale: bool = True,
    keep_missing_as_nan: bool = True,
    canon_orient: bool = False, 
    canon_smooth: float = 0.2,
    write_meta: bool = True,
):
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=min_det,
        min_tracking_confidence=min_track,
    )

    seq = []
    vis_mask = []
    timestamps = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if use_world and res.pose_world_landmarks:
            lm_list = res.pose_world_landmarks.landmark
        elif (not use_world) and res.pose_landmarks:
            lm_list = res.pose_landmarks.landmark
        else:
            lm_list = None

        if lm_list is not None:
            pts = np.zeros((33, 4), dtype=np.float32)
            mask = np.zeros((33,), dtype=np.float32)
            for i, lm in enumerate(lm_list):
                pts[i, 0] = lm.x
                pts[i, 1] = lm.y
                pts[i, 2] = lm.z
                # Some MediaPipe builds use 'visibility' for both world & image; if absent, default 1.0
                vis = getattr(lm, "visibility", 1.0)
                pts[i, 3] = float(vis)
                mask[i] = 1.0
            seq.append(pts)
            vis_mask.append(mask)
        else:
            if keep_missing_as_nan:
                seq.append(np.full((33, 4), np.nan, dtype=np.float32))
                vis_mask.append(np.zeros((33,), dtype=np.float32))
            else:
                # Placeholder (will be forward-filled later)
                seq.append(np.full((33, 4), np.nan, dtype=np.float32))
                vis_mask.append(np.zeros((33,), dtype=np.float32))

        timestamps.append(frame_idx / fps)
        frame_idx += 1

    cap.release()
    pose.close()

    poses = np.asarray(seq, dtype=np.float32)      # [T,33,4]
    mask = np.asarray(vis_mask, dtype=np.float32)  # [T,33]
    T = poses.shape[0]

    # Optionally forward-fill missing frames (before smoothing/normalization)
    if not keep_missing_as_nan:
        poses = fill_missing_frames(poses)

    # Smoothing (on xyz only; leave vis as-is)
    if smooth_ema_alpha > 0.0:
        xyz = poses[..., :3]
        xyz = ema_smooth(xyz, alpha=smooth_ema_alpha)
        poses[..., :3] = xyz
    canon_info = None
    if use_world and canon_orient:
        poses, canon_info = canonicalize_world_sequence(
            poses=poses,
            vis_mask=mask,
            ema_alpha=float(canon_smooth)
        )
    # Normalize
    poses = normalize_sequence(
        poses,
        center=center,
        scale=scale,
        use_world=use_world,
    )

    # Save
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        f"{out_prefix}.npz",
        poses=poses,
        vis_mask=mask,
    )
    meta = {
        "video_path": str(video_path),
        "fps": float(fps),
        "width": int(w),
        "height": int(h),
        "num_frames": int(T),
        "uses_world_landmarks": bool(use_world),
        "coord_format": "world(x,y,z in meters) + visibility"
        if use_world
        else "image-normalized(x,y in [0..1], z relative) + visibility",
        "landmarks": "MediaPipe Pose 33",
        "preprocessing": {
            "centered_by": "pelvis(avg L/R hip)" if center else None,
            "scaled_by": "torso diag (mid-shoulders to mid-hips)" if scale else None,
            "ema_alpha": float(smooth_ema_alpha),
            "keep_missing_as_nan": bool(keep_missing_as_nan),
        },
        "timestamps_sec": timestamps,
        "mediapipe": {
            "model_complexity": int(model_complexity),
            "min_detection_confidence": float(min_det),
            "min_tracking_confidence": float(min_track),
        },
         "canonical_orientation": {
            "enabled": bool(use_world and center and scale and canon_orient),
            "ema_alpha": float(canon_smooth) if (use_world and canon_orient) else 0.0,
            "axes": "x=left->right, y=hips->shoulders (up), z=up×x (forward, right-handed)"
        },
    }
    if write_meta:
        Path(f"{out_prefix}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),encoding='utf-8')
        print(f"[OK] Saved {out_prefix}.npz and {out_prefix}.json")

    print(f"[OK] Saved {out_prefix}.npz")


def main():
    p = argparse.ArgumentParser()
    # --- existing extractor flags ---
    p.add_argument("--video", required=True, type=str)
    p.add_argument("--out-prefix", type=str)
    p.add_argument("--use-world", action="store_true")
    p.add_argument("--model-complexity", type=int, default=2, choices=[0, 1, 2])
    p.add_argument("--min-det", type=float, default=0.6)
    p.add_argument("--min-track", type=float, default=0.6)
    p.add_argument("--smooth-ema", type=float, default=0.0)
    p.add_argument("--no-center", action="store_true")
    p.add_argument("--no-scale", action="store_true")
    p.add_argument("--save-missing-as-nan", action="store_true")
    p.add_argument("--no-meta", action="store_true", help="Do not write the .json metadata file.")

    # --- rendering flags ---
    p.add_argument("--render", action="store_true",
                   help="Render overlay. If poses/meta are provided and not world, use them; else run MP live.")
    p.add_argument("--poses", type=str, help="Path to saved .npz (from extractor)")
    p.add_argument("--meta", type=str, help="Path to saved .json (from extractor)")
    p.add_argument("--render-out", type=str, default=None, help="Output mp4 path")
    p.add_argument("--show", action="store_true", help="Show preview window")
    p.add_argument("--viz-vis-thresh", type=float, default=0.5)
    p.add_argument("--viz-skip", type=int, default=1)
    
    p.add_argument("--canon-orient", action="store_true",
                   help="Canonicalize orientation for WORLD coords (rotate to body-fixed axes).")
    p.add_argument("--canon-smooth", type=float, default=0.2,
                   help="EMA smoothing for body axes in [0,1]. 0 disables.")

    p.add_argument("--debug-axes", action="store_true",
                   help="Draw body axes (X=green, Y=blue, Z=red) at pelvis (2D approximation).")
    p.add_argument("--axes-scale", type=float, default=80.0,
                   help="Length of debug axis arrows in pixels.")
    p.add_argument("--axes-alpha", type=float, default=0.8,
                   help="Opacity (0..1) for debug axes overlay.")

    args = p.parse_args()

    if args.render:
        # Try using saved poses first (only if provided and not world)
        if args.poses and args.meta:
            try:
                render_overlay_from_saved(
                    video_path=args.video,
                    poses_path=args.poses,
                    meta_path=args.meta,
                    out_path=args.render_out,
                    show=args.show,
                    vis_thresh=args.viz_vis_thresh,
                    viz_skip=args.viz_skip,
                    debug_axes=args.debug_axes,
                    axes_scale=args.axes_scale,
                    axes_alpha=args.axes_alpha,
                )
                return
            except Exception as e:
                print("[viewer] Saved-poses overlay not possible:", e)
                print("[viewer] Falling back to MediaPipe live rendering...")

        # Fallback path
        render_overlay_from_mediapipe(
            video_path=args.video,
            out_path=args.render_out,
            show=args.show,
            model_complexity=max(1, args.model_complexity),
            min_det=args.min_det,
            min_track=args.min_track,
            vis_thresh=args.viz_vis_thresh,
            viz_skip=args.viz_skip,
            debug_axes=args.debug_axes,
            axes_scale=args.axes_scale,
            axes_alpha=args.axes_alpha,
        )
        return

    # Otherwise: run extractor (requires --out-prefix)
    if not args.out_prefix:
        raise SystemExit("When not using --render, you must provide --out-prefix to save extracted poses.")
    extract(
       video_path=args.video,
        out_prefix=args.out_prefix,
        use_world=args.use_world,
        model_complexity=args.model_complexity,
        min_det=args.min_det,
        min_track=args.min_track,
        smooth_ema_alpha=args.smooth_ema,
        center=not args.no_center,
        scale=not args.no_scale,
        keep_missing_as_nan=args.save_missing_as_nan,
        canon_orient=args.canon_orient,
        canon_smooth=args.canon_smooth,
        write_meta=not args.no_meta,
    )




# Minimal skeleton for MediaPipe's 33 joints (indices per MP Pose)
POSE_EDGES = [
    # torso
    (11, 12), (23, 24), (11, 23), (12, 24),
    # left arm
    (11, 13), (13, 15),
    # right arm
    (12, 14), (14, 16),
    # left leg
    (23, 25), (25, 27), (27, 29), (29, 31),
    # right leg
    (24, 26), (26, 28), (28, 30), (30, 32),
    # head/face (sparse)
    (0, 2), (2, 3), (3, 7),   # nose -> left eye -> left eye outer -> left ear
    (0, 5), (5, 6), (6, 8),   # nose -> right eye -> right eye outer -> right ear
]

def draw_skeleton(
    frame_bgr: np.ndarray,
    kpts_xyv: np.ndarray,       # [33, 3] with (x_px, y_px, visibility) or NaNs
    vis_thresh: float = 0.5,
):
    
    """Draw circles and lines for the 33 joints on a BGR frame."""
    H, W = frame_bgr.shape[:2]
    # joints
    for j in range(33):
        x, y, v = kpts_xyv[j]
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        if np.isfinite(v) and v < vis_thresh:
            continue
        cv2.circle(frame_bgr, (int(round(x)), int(round(y))), 3, (0, 255, 0), -1)

    # edges
    for a, b in POSE_EDGES:
        xa, ya, va = kpts_xyv[a]
        xb, yb, vb = kpts_xyv[b]
        if not (np.isfinite(xa) and np.isfinite(ya) and np.isfinite(xb) and np.isfinite(yb)):
            continue
        if (np.isfinite(va) and va < vis_thresh) or (np.isfinite(vb) and vb < vis_thresh):
            continue
        cv2.line(frame_bgr, (int(round(xa)), int(round(ya))), (int(round(xb)), int(round(yb))), (0, 200, 255), 2)

def render_overlay_from_saved(
    video_path: str,
    poses_path: str,
    meta_path: str,
    out_path: str | None = None,
    show: bool = False,
    vis_thresh: float = 0.5,
    viz_skip: int = 1,
    debug_axes: bool = False,
    axes_scale: float = 80.0,
    axes_alpha: float = 0.8,
):
    ...
    DEBUG_AXES_ENABLED = debug_axes
    axes_scale = float(axes_scale)
    axes_alpha = float(axes_alpha)
    """Overlay using saved normalized image coordinates from the .npz file."""
    # Load data
    data = np.load(poses_path)
    poses = data["poses"]  # [T,33,4]
    meta = json.loads(Path(meta_path).read_text())
    uses_world = meta.get("uses_world_landmarks", False)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # If saved are world coords, we can't overlay; caller should fallback
    if uses_world:
        raise ValueError("Saved poses are world-coordinates; cannot overlay. Use render_overlay_from_mediapipe().")

    # Video writer
    writer = None
    if out_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps / viz_skip, (W, H))

    t = 0
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % viz_skip == 0:
            if t < poses.shape[0]:
                # Convert normalized (x,y) -> pixels; keep z as-is; use visibility as 3rd
                kpts = poses[t, :, :4]  # [33,4]
                xyv = np.full((33, 3), np.nan, dtype=np.float32)
                for j in range(33):
                    x, y, z, v = kpts[j]
                    if np.isfinite(x) and np.isfinite(y):
                        xyv[j, 0] = x * W
                        xyv[j, 1] = y * H
                        xyv[j, 2] = v
                draw_skeleton(frame, xyv, vis_thresh=vis_thresh)
                if vis_thresh < 0:  # (no-op trick to keep patching simple)
                    pass
                if DEBUG_AXES_ENABLED:  # <- we’ll pass this via function args
                    draw_body_axes_overlay(frame,
                                           xyv,
                                           scale_px=axes_scale,
                                           alpha=axes_alpha)

            if show:
                cv2.imshow("Pose Overlay", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                    break

            if writer is not None:
                writer.write(frame)

            t += 1

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    print("[OK] Rendered overlay from saved poses.",
          f"Saved to {out_path}" if out_path else "")

def render_overlay_from_mediapipe(
    video_path: str,
    out_path: str | None = None,
    show: bool = False,
    model_complexity: int = 1,
    min_det: float = 0.6,
    min_track: float = 0.6,
    vis_thresh: float = 0.5,
    viz_skip: int = 1,
    debug_axes: bool = False,
    axes_scale: float = 80.0,
    axes_alpha: float = 0.8,
):
    DEBUG_AXES_ENABLED = debug_axes
    axes_scale = float(axes_scale)
    axes_alpha = float(axes_alpha)

    """Overlay by running MediaPipe live on the video (works for any input, including world-saved)."""
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=min_det,
        min_tracking_confidence=min_track,
    )

    writer = None
    if out_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps / viz_skip, (W, H))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if frame_idx % viz_skip != 0:
            frame_idx += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        xyv = np.full((33, 3), np.nan, dtype=np.float32)
        if res.pose_landmarks:
            for j, lm in enumerate(res.pose_landmarks.landmark):
                xyv[j, 0] = lm.x * W
                xyv[j, 1] = lm.y * H
                xyv[j, 2] = getattr(lm, "visibility", 1.0)

        draw_skeleton(frame, xyv, vis_thresh=vis_thresh)
        if DEBUG_AXES_ENABLED:
            draw_body_axes_overlay(frame,
                                   xyv,
                                   scale_px=axes_scale,
                                   alpha=axes_alpha)
        if show:
            cv2.imshow("Pose Overlay", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        if writer is not None:
            writer.write(frame)

        frame_idx += 1

    cap.release()
    pose.close()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    print("[OK] Rendered overlay via MediaPipe.",
          f"Saved to {out_path}" if out_path else "")

def _pelvis_xy_from_xyv(xyv: np.ndarray) -> np.ndarray | None:
    """Return pelvis pixel coords from 2D landmarks (mid-hips)."""
    L_HIP, R_HIP = 23, 24
    if not (np.isfinite(xyv[L_HIP, :2]).all() and np.isfinite(xyv[R_HIP, :2]).all()):
        return None
    return 0.5 * (xyv[L_HIP, :2] + xyv[R_HIP, :2])

def _axes_2d_from_xyv(xyv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Build 2D axis directions from image landmarks:
      X ~ left->right (avg of shoulders & hips),
      Y ~ hips->shoulders,
      Z ~ pseudo-forward = rotate Y towards screen by 90° (perp to Y in image plane),
    All unit-length in 2D.
    """
    L_HIP, R_HIP = 23, 24
    L_SH,  R_SH  = 11, 12
    need = [L_HIP, R_HIP, L_SH, R_SH]
    if any(not np.isfinite(xyv[i, :2]).all() for i in need):
        return None

    lhip, rhip = xyv[L_HIP, :2], xyv[R_HIP, :2]
    lsh,  rsh  = xyv[L_SH,  :2], xyv[R_SH,  :2]

    # left->right using both shoulders & hips (more stable)
    lr = 0.5 * ((rsh - lsh) + (rhip - lhip))
    y  = 0.5 * ((lsh + rsh) - (lhip + rhip))  # hips->shoulders

    def _safe_unit(v):
        n = np.linalg.norm(v)
        if n < 1e-6: return None
        return v / n

    x = _safe_unit(lr)
    y = _safe_unit(y)
    if x is None or y is None:
        return None

    # Pseudo forward: +Z is a 2D perpendicular to Y (rotate Y clockwise)
    z = np.array([ y[1], -y[0] ], dtype=np.float32)
    z = _safe_unit(z)
    if z is None:
        return None
    return x, y, z

def draw_body_axes_overlay(
    frame_bgr: np.ndarray,
    xyv: np.ndarray,            # [33,3] (x_px, y_px, vis)
    scale_px: float = 80.0,
    alpha: float = 0.8,
):
    """
    Draw semi-transparent axes triad at pelvis:
      X=green, Y=blue, Z=red. Uses 2D approximations.
    """
    origin = _pelvis_xy_from_xyv(xyv)
    axes = _axes_2d_from_xyv(xyv)
    if origin is None or axes is None:
        return
    x_dir, y_dir, z_dir = axes
    o = origin.astype(np.int32)

    # Prepare overlay
    overlay = frame_bgr.copy()
    def _arrow(dst, vec, color):
        p2 = (o + (vec * scale_px)).astype(np.int32)
        cv2.arrowedLine(overlay, tuple(o), tuple(p2), color, 2, tipLength=0.25)

    _arrow(overlay, x_dir, (0,255,0))   # X
    _arrow(overlay, y_dir, (255,0,0))   # Y (blue in BGR would be (255,0,0), we’ll keep note)
    _arrow(overlay, z_dir, (0,0,255))   # Z (red in BGR)

    # Alpha blend
    cv2.addWeighted(overlay, alpha, frame_bgr, 1.0 - alpha, 0, frame_bgr)

    # Legend
    cv2.putText(frame_bgr, "X", tuple((o + x_dir* (scale_px+12)).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
    cv2.putText(frame_bgr, "Y", tuple((o + y_dir* (scale_px+12)).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
    cv2.putText(frame_bgr, "Z", tuple((o + z_dir* (scale_px+12)).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)


if __name__ == "__main__":
    main()

