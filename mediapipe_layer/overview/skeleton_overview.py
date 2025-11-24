"""
3D Pose Viewer
----------------
Visualize canonicalized MediaPipe-33 world poses (.npz) in 3D.

Usage:
  python skeleton_overview.py --poses out/clip.npz --meta out/clip.json
"""

import argparse, json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# same 33-edge skeleton
POSE_EDGES = [
    (11,12),(23,24),(11,23),(12,24),
    (11,13),(13,15),(12,14),(14,16),
    (23,25),(25,27),(27,29),(29,31),
    (24,26),(26,28),(28,30),(30,32),
    (0,2),(2,3),(3,7),(0,5),(5,6),(6,8)
]

def draw_axes(ax, origin, R, scale=0.3):
    """Draw 3D triad (X=green, Y=blue, Z=red) at origin."""
    colors = [(0,1,0),(0,0,1),(1,0,0)]
    for i,c in enumerate(colors):
        v = R[:,i] * scale
        ax.quiver(origin[0], origin[1], origin[2],
                  v[0], v[1], v[2],
                  color=c, linewidth=2)

def plot_frame(ax, xyz):
    for a,b in POSE_EDGES:
        if np.all(np.isfinite(xyz[[a,b],:])):
            ax.plot(*xyz[[a,b],:].T, color="gray", linewidth=1)
    ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], s=15, c="k")

def view3d(poses_path, meta_path, frame_step=5):
    data = np.load(poses_path)
    meta = json.loads(open(meta_path).read())
    poses = data["poses"]       # [T,33,4]
    xyz = poses[::frame_step, :, :3]
    print(f"Loaded {poses.shape[0]} frames -> showing every {frame_step}th.")

    fig = plt.figure(figsize=(7,7))
    ax: Axes3D = fig.add_subplot(111, projection="3d")

    # choose a consistent scale
    lim = np.nanmax(np.abs(xyz))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel("X (left→right)")
    ax.set_ylabel("Y (up)")
    ax.set_zlabel("Z (forward)")

    for t in range(xyz.shape[0]):
        ax.cla()
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.view_init(elev=0, azim=0,roll=90)
        plot_frame(ax, xyz[t])
        # draw pelvis axes if available
        L_HIP,R_HIP,L_SH,R_SH=23,24,11,12
        if all(np.isfinite(xyz[t,[L_HIP,R_HIP,L_SH,R_SH],0])):
            pelvis=(xyz[t,L_HIP]+xyz[t,R_HIP])/2
            lr=0.5*((xyz[t,R_SH]-xyz[t,L_SH])+(xyz[t,R_HIP]-xyz[t,L_HIP]))
            up=0.5*((xyz[t,L_SH]+xyz[t,R_SH])-(xyz[t,L_HIP]+xyz[t,R_HIP]))
            lr/=np.linalg.norm(lr)+1e-6
            up/=np.linalg.norm(up)+1e-6
            fwd=np.cross(up,lr); fwd/=np.linalg.norm(fwd)+1e-6
            R=np.stack([lr,up,fwd],1)
            draw_axes(ax,pelvis,R)
        plt.pause(0.5)
    plt.show()

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--poses",required=True)
    ap.add_argument("--meta",required=True)
    ap.add_argument("--step",type=int,default=5)
    args=ap.parse_args()
    view3d(args.poses,args.meta,args.step)
