import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

ref = np.load('processed_data/ref_landmarks.npy')
test = np.load('processed_data/all_landmarks.npy')

# Flatten each frame to 1D (33*3 for x,y,z)
ref_flat = [frame[:, :3].flatten() for frame in ref]
test_flat = [frame[:, :3].flatten() for frame in test]

distance, path = fastdtw(ref_flat, test_flat, dist=euclidean)
print(f"DTW quality score (lower is better): {distance:.4f}")