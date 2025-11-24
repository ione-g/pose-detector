import os, csv, math, random, argparse
from pathlib import Path
import numpy as np

# MediaPipe indices (subset we’ll animate)
NOSE=0
L_EYE=2; R_EYE=5
L_EAR=7; R_EAR=8
L_SH=11; R_SH=12
L_EL=13; R_EL=14
L_WR=15; R_WR=16
L_HIP=23; R_HIP=24
L_KNEE=25; R_KNEE=26
L_ANK=27; R_ANK=28
L_HEEL=29; R_HEEL=30
L_TOE=31; R_TOE=32

J = 33

def base_tpose():
    """Return a neutral standing pose (33x3) in 'world-like' meters."""
    P = np.zeros((J,3), dtype=np.float32)
    # Pelvis at origin, up = +Y, left->right = +X, forward = +Z
    pelvis = np.array([0.0, 1.0, 0.0], np.float32)  # ~1m high
    hip_w = 0.25; sh_w = 0.30; arm = 0.55; leg = 0.9

    # Hips & spine
    P[L_HIP] = pelvis + np.array([-hip_w, 0.0, 0.0])
    P[R_HIP] = pelvis + np.array([ hip_w, 0.0, 0.0])
    spine_up = np.array([0.0, 0.35, 0.0])
    LSM = 0.0

    # Shoulders
    shoulder_center = pelvis + spine_up + np.array([0,0.25,0])
    P[L_SH] = shoulder_center + np.array([-sh_w, 0.0, 0.0])
    P[R_SH] = shoulder_center + np.array([ sh_w, 0.0, 0.0])

    # Knees & ankles (straight)
    P[L_KNEE] = P[L_HIP] + np.array([0.0, -leg*0.5, 0.05])
    P[R_KNEE] = P[R_HIP] + np.array([0.0, -leg*0.5, 0.05])
    P[L_ANK]  = P[L_HIP] + np.array([0.0, -leg, 0.10])
    P[R_ANK]  = P[R_HIP] + np.array([0.0, -leg, 0.10])
    P[L_HEEL] = P[L_ANK]  + np.array([0.0,  0.0, -0.05])
    P[R_HEEL] = P[R_ANK]  + np.array([0.0,  0.0, -0.05])
    P[L_TOE]  = P[L_ANK]  + np.array([0.0,  0.0,  0.15])
    P[R_TOE]  = P[R_ANK]  + np.array([0.0,  0.0,  0.15])

    # Arms (down)
    P[L_EL] = P[L_SH] + np.array([-0.05, -arm*0.5, 0.05])
    P[R_EL] = P[R_SH] + np.array([ 0.05, -arm*0.5, 0.05])
    P[L_WR] = P[L_SH] + np.array([-0.05, -arm, 0.05])
    P[R_WR] = P[R_SH] + np.array([ 0.05, -arm, 0.05])

    # Head (rough)
    P[NOSE] = shoulder_center + np.array([0, 0.25, 0.05])
    P[L_EYE] = NOSE; P[R_EYE] = NOSE
    P[L_EAR] = P[L_SH] + np.array([0.0, 0.20, 0.0])
    P[R_EAR] = P[R_SH] + np.array([0.0, 0.20, 0.0])

    # Fill any remaining (copy nearest meaningful joint)
    for i in range(J):
        if not np.isfinite(P[i]).all():
            P[i] = shoulder_center
    return P

def squat_motion(T, reps=2, amp=0.35, phase=0.0):
    """Return per-frame scalar 'depth' in [0..1] and velocity; sinusoidal reps."""
    t = np.linspace(0, 2*np.pi*reps, T) + phase
    depth = 0.5*(1 - np.cos(t))  # 0 at top, 1 at bottom
    vel = np.gradient(depth)
    return depth.astype(np.float32), vel.astype(np.float32)

def apply_squat(P0, depth, cls, noise=0.005):
    """
    Generate frames from base pose P0 (33x3), given depth[t] in [0..1].
    cls in {"correct","knees_in","shallow","forward_lean"}.
    """
    T = depth.shape[0]
    seq = np.repeat(P0[None,...], T, axis=0)  # [T,33,3]
    # Parameters
    max_down = 0.40  # meters pelvis descend at depth=1
    knee_flex = 0.45
    torso_lean = 0.35 if cls=="forward_lean" else 0.15
    knee_valgus = 0.10 if cls=="knees_in" else 0.02
    depth_scale = 0.50 if cls=="shallow" else 1.0

    for t in range(T):
        d = depth[t] * depth_scale
        # Pelvis down
        dpelv = np.array([0.0, -max_down*d, 0.0], np.float32)
        # Update hips & shoulders (translate with pelvis)
        for j in (L_HIP,R_HIP,L_SH,R_SH,NOSE,L_EAR,R_EAR,L_EYE,R_EYE):
            seq[t,j] = P0[j] + dpelv

        # Knees: move forward (Z+) and down (Y-)
        for hip,knee,ank in ((L_HIP,L_KNEE,L_ANK),(R_HIP,R_KNEE,R_ANK)):
            knee_dir = np.array([0.0, -knee_flex*d, 0.12*d], np.float32)
            seq[t,knee] = P0[knee] + dpelv + knee_dir
            # Ankles mostly anchored, slight forward drift
            seq[t,ank]  = P0[ank]  + np.array([0, 0, 0.03*d], np.float32)

        # Feet follow ankles
        for heel,toe,ank in ((L_HEEL,L_TOE,L_ANK),(R_HEEL,R_TOE,R_ANK)):
            base_heel = np.array([0,0,-0.05], np.float32)
            base_toe  = np.array([0,0, 0.15], np.float32)
            seq[t,heel] = seq[t,ank] + base_heel
            seq[t,toe]  = seq[t,ank] + base_toe

        # Torso lean: rotate shoulders forward around hips in X-axis plane
        hip_center = 0.5*(seq[t,L_HIP] + seq[t,R_HIP])
        sh_center  = 0.5*(seq[t,L_SH]  + seq[t,R_SH])
        v = sh_center - hip_center
        v = v + np.array([0, 0,  torso_lean*d], np.float32)  # push forward with depth
        seq[t,L_SH] = hip_center + (v + np.array([-0.30,0,0],np.float32))
        seq[t,R_SH] = hip_center + (v + np.array([ 0.30,0,0],np.float32))
        # Head follows shoulders
        seq[t,NOSE] = 0.5*(seq[t,L_SH] + seq[t,R_SH]) + np.array([0,0.20,0.05],np.float32)
        seq[t,L_EAR] = seq[t,L_SH] + np.array([0,0.20,0],np.float32)
        seq[t,R_EAR] = seq[t,R_SH] + np.array([0,0.20,0],np.float32)

        # Valgus: pull knees inward on X
        if cls == "knees_in":
            seq[t,L_KNEE,0] += -knee_valgus * d
            seq[t,R_KNEE,0] +=  knee_valgus * d

        # Small noise
        seq[t] += np.random.normal(0, noise, seq[t].shape).astype(np.float32)

    return seq

def to_npz(path, xyz):
    """Save as .npz with 'poses' [T,33,4] and 'vis_mask' [T,33]."""
    T = xyz.shape[0]
    poses = np.zeros((T,J,4), np.float32)
    poses[...,:3] = xyz
    poses[..., 3] = 1.0
    vis_mask = np.ones((T,J), np.float32)
    np.savez_compressed(path, poses=poses, vis_mask=vis_mask)

def gen_split(out_dir, n_per_class=20, T=120, train_ratio=0.8, seed=42):
    random.seed(seed); np.random.seed(seed)
    classes = ["correct","knees_in","shallow","forward_lean"]
    out_dir = Path(out_dir)
    (out_dir/"clips").mkdir(parents=True, exist_ok=True)

    rows_train, rows_val = [], []
    for cls in classes:
        for i in range(n_per_class):
            P0 = base_tpose()
            reps = random.choice([2,3])
            depth, vel = squat_motion(T, reps=reps, phase=random.uniform(0,2*math.pi))
            xyz = apply_squat(P0, depth, cls, noise=0.004)
            # random global yaw (rotate around Y) to simulate multi-view, then canonicalization in extractor should fix later
            yaw = random.uniform(-math.pi/4, math.pi/4)
            R = np.array([[ math.cos(yaw), 0, math.sin(yaw)],
                          [ 0,            1, 0           ],
                          [-math.sin(yaw),0, math.cos(yaw)]], np.float32)
            xyz = (R @ xyz.reshape(-1,3).T).T.reshape(T,J,3)

            clip_name = f"{cls}_{i:03d}.npz"
            to_npz(str(out_dir/"clips"/clip_name), xyz)
            row = [str(out_dir/"clips"/clip_name), cls]
            if random.random() < train_ratio:
                rows_train.append(row)
            else:
                rows_val.append(row)

    # write CSVs
    with open(out_dir/"train.csv","w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerows(rows_train)
    with open(out_dir/"val.csv","w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerows(rows_val)
    print(f"[OK] Synth data in {out_dir}")
    print(f"  train: {len(rows_train)}  val: {len(rows_val)}  classes={classes}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="synthetic_mp33")
    ap.add_argument("--clips-per-class", type=int, default=20)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--train-ratio", type=float, default=0.8)
    args = ap.parse_args()
    gen_split(args.out, n_per_class=args.clips_per_class, T=args.frames, train_ratio=args.train_ratio)
