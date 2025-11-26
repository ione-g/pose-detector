# Pose Detector / Activity Classifier

Коротко
-------
Цей репозиторій містить інструменти для:
- екстракції поз (MediaPipe) з відео;
- збереження й візуалізації збережених поз (2D/3D);
- простого UI для вибору фрагменту відео (slicer);
- порівняння/оцінки якості виконання вправ;
- навчання та швидкої перевірки класифікатора на базі ST-GCN.

Структура проекту (коротко)
---------------------------
- `data_sets/` — сирі відео (по папках, класи як підпапки).
- `out/` — результати (npz, json, кліпи та csv).
- `utils/`
  - `pose_extractor.py` — скрипт/функції для екстракції поз з відео.
  - `mp4slicer.py` — простий PyQt5 GUI (timeline + 3 маркери).
  - `pose_comparator.py` — DTW / відстані між послідовностями поз.
  - інші корисні утиліти.
- `mediapipe_layer/` — основний екстрактор, batch утиліти та візуалізація.
- `nn/` — тренування та quick-eval класифікатора (STGCN).
- `checkpoints/` — готові вайти моделі.

Залежності
----------
Системні:
- Homebrew (macOS) або відповідні пакунки для Linux.
- ffmpeg
- yt-dlp (для завантаження з YouTube)

macOS (Homebrew) приклад:
```bash
brew install ffmpeg yt-dlp
```

Python (venv)
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install opencv-python mediapipe numpy torch torchvision scikit-learn fastdtw moviepy pyqt5 matplotlib
```

(для Apple Silicon або спеціального CUDA/ROCm використайте інструкції для `torch` від PyTorch)

Рекомендований Requirements (приклад)
```bash
pip install -r requirements.txt
# або поелементно:
pip install opencv-python mediapipe numpy torch torchvision scikit-learn fastdtw moviepy pyqt5 matplotlib
```

Екстракція поз (з відео)
-----------------------
Запуск екстрактора (один файл):
```bash
python mediapipe_layer/extractor.py --video path/to/video.mp4 --out-prefix out/clip
```

Параметри:
- `--use-world` — використовувати WORLD (3D) координати, якщо потрібно.
- `--model-complexity` — 0/1/2.
- `--min-det`, `--min-track` — поріг детекції/трекінгу.
- `--no-meta` — не записувати JSON метадані.

Batch-генерація (через `batch_extractor.py`) — дивіться help:
```bash
python mediapipe_layer/batch_extractor.py --indir data_sets --out out --pattern '*.mp4'
```

Формат, який зберігається
- `.npz`:
  - `poses` — масив розміру (T, 33, [x,y,z,vis]) — або (T, 33, 3) (без visibility).
- `.json` — метадані (fps, width, height, canonicalization flags тощо).

Візуалізація (overlay / 3D)
---------------------------
Оверлей на відео (якщо `--render`):
```bash
python mediapipe_layer/extractor.py --render --video path/to/video.mp4 --poses out/clip.npz --meta out/clip.json --show
```

3D огляд (кості):
```bash
python mediapipe_layer/overview/skeleton_overview.py --poses out/clip.npz --meta out/clip.json --step 1
```

Slicing / GUI timeline
----------------------
Запуск PyQt5 GUI:
```bash
python utils/mp4slicer.py
```

Функції GUI:
- Відкрити відео;
- перший маркер — `current` (ручне управління положенням відтворення);
- другий маркер — `start` (початок фрагмента);
- третій маркер — `end` (кінець фрагмента);
- replay segment — відтворює лише виділену область (істотно для швидкого перегляду);
- slice & save — зберегти сегмент у файл (`ffmpeg -ss ... -to ...` з мілісекундами).

Зауваження:
- Точність — мілісекунди; GUI синхронізунується з маркерами.

Порівняння / Оцінка якості
-------------------------
- Використовуйте `pose_comparator.py` або `fastdtw` для вирівнювання послідовностей (DTW) за часом.
- Для швидкої перевірки схожості:
  - нормальна відстань (x,y,z) між відповідними суглобами;
  - усереднення по часах та суглобам — дає загальний score (нижче — краще).
- Краща практика: вирівняти послідовності DTW → пер-джойн компресія → вирахувати per-frame score → усереднити.

Нейронна мережа / Тренування
---------------------------
Сценарії:
- Підготовка вікон (windows) з `utils/dataset_windows.py`.
- Навчання ST-GCN (в каталозі `nn/`).

Приклад запуску тренування (якщо скрипт `train_classifier.py` завершений):
```bash
python -m nn.train_classifier \
  --train-csv out/synth/train.csv \
  --val-csv out/synth/val.csv \
  --classes correct knees_in shallow forward_lean \
  --epochs 10 --bs 32 --augment
```

Оцінка вже натренованої моделі на тестовому наборі:
```bash
python -m nn.quick_classificator_from_trained_model \
  --test-csv out/test.csv \
  --checkpoint checkpoints/stgcn33_7classes.pth \
  --classes correct knees_in shallow forward_lean ...
```

Формат CSV:
```
/path/to/out/clip.npz,label
```

Налаштування навчальної моделі
- Модель STGCN33 знаходиться у `nn/stgcn.py`.
- Контроль параметрів моделі та навчання — через аргументи у `train_classifier.py`.

Запуск і відладка (поширені помилки)
-----------------------------------
- ModuleNotFoundError: No module named 'utils'
  - Запускайте скрипти з кореневої директорії проекту (`cd <project_root>`).
  - Або додайте `export PYTHONPATH=$(pwd)` перед запуском.

- ModuleNotFoundError: No module named '_tkinter' / `python -m tkinter` помилка
  - Інсталяція tcl-tk через Homebrew та перезбірка/перевстановлення Python:
    ```bash
    brew install tcl-tk
    brew reinstall python@3.11
    ```
  - Пересоздайте venv після встановлення.

- HTTP 403 у `yt-dlp`:
  - Оновіть `yt-dlp` або спробуйте `--user-agent` опцію.
  - Деякі відео мають обмеження (приватні, регіональні, age-restricted).

- ffmpeg command not found:
  - Встановіть `ffmpeg` через Homebrew: `brew install ffmpeg`.

- Tkinter у venv:
  - Tkinter поставляється з самою збіркою Python; переконайтесь, що Python для venv має підтримку tcl-tk.

Корисні підказки
----------------
- Щоб уникнути повторних завантажень відео через `yt-dlp`, використовуйте кешування — зберігайте оригінальні mp4 і обрізайте від них (`ffmpeg -ss ... -to ... -c copy`).
- Для вирівнювання різних швидкостей виконання вправ використовуйте DTW (fastdtw).
- Зберігайте `.npz` у форматі `(T, 33, 4)` (x,y,z,visibility) або `(T,33,3)` якщо visibility непотрібний.
- Використовуйте `--step 1` при візуалізації невеликих відео, щоб не 'пропускати' кадри.



