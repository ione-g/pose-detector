# nn/quick_classificator_from_trained_model.py

import torch
import numpy as np
import torch.nn.functional as F
from nn.stgcn import STGCN33  # Імпортуємо вашу модель
import sys
from pathlib import Path

# === Налаштування ===
# NPZ_PATH = "out/push_ups/clips/push_up_sideview_clip009.npz"  # 🔁 ← Вкажіть шлях до .npz
NPZ_PATH = "out/squats/clips/squat_clip003.npz"  # 🔁 ← Вкажіть шлях до .npz

CHECKPOINT_PATH = "checkpoints/stgcn33_correctonly_2diff_exercises.pth"
NUM_CLASSES = 2  # Залежно від кількості класів при тренуванні

# === Завантаження даних ===
def load_npz_pose(npz_path):
    npz = np.load(npz_path)
    xyz = npz["poses"][:, :, :3]  # беремо тільки x, y, z
    tensor = torch.tensor(xyz, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # [1, 3, T, 33]
    return tensor

# === Ініціалізація та завантаження моделі ===
model = STGCN33(n_classes=NUM_CLASSES)
checkpoint = torch.load(CHECKPOINT_PATH)

if "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()

# === Інференс ===
x = load_npz_pose(NPZ_PATH)
with torch.no_grad():
    logits = model(x)
    probs = F.softmax(logits, dim=1)
    pred = probs.argmax(dim=1).item()

print("Файл:", NPZ_PATH)
print("Передбачений клас:", pred)
print("Ймовірності:", probs.numpy())
