import sys
import os
import shutil
import math
import subprocess
from typing import Optional

from PyQt5.QtCore import Qt, QUrl, QTimer, QRectF, QPointF, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
    QHBoxLayout
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget

# ---------- Optional backends ----------
try:
    import cv2
    _CV2_AVAILABLE = True
except Exception:
    _CV2_AVAILABLE = False

try:
    # MoviePy is optional; will be used only if ffmpeg CLI is missing.
    import moviepy.editor as mpy
    _MOVIEPY_AVAILABLE = True
except Exception:
    _MOVIEPY_AVAILABLE = False


# =========================================================
#  ThreeMarkerSlider (triangles, collision-safe hit testing)
# =========================================================
class ThreeMarkerSlider(QWidget):
    """
    Custom horizontal slider with 3 draggable triangle markers:
      - current : playback position (blue)
      - start   : segment start (green)
      - end     : segment end (red)

    If markers overlap (same ms or very close), they are visually separated
    with small horizontal nudges and hit-testing uses the separated positions.
    Units are milliseconds (ms).
    """
    currentChanged = pyqtSignal(int)  # ms
    startChanged = pyqtSignal(int)    # ms
    endChanged = pyqtSignal(int)      # ms

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(42)
        self._duration = 0
        self._current = 0
        self._start = 0
        self._end = 0

        # dragging
        self._dragging = None  # "current" | "start" | "end" | None
        self._left_pad = 12
        self._right_pad = 12
        self._track_height = 6
        self._handle_r = 9      # triangle radius
        self._grab_tol = 18     # px
        self._min_sep = 18      # minimal separation between display positions

        # display (collision-resolved) positions for hit-testing
        self._disp_x = {"current": None, "start": None, "end": None}

        # colors
        self._pen = QPen(Qt.black, 1)
        self._colors = {
            "current": QColor(60, 120, 255),  # blue
            "start":   QColor(60, 180, 75),   # green
            "end":     QColor(230, 70, 70),   # red
        }

    # -------- Public API --------
    def setDuration(self, ms: int):
        self._duration = max(0, int(ms))
        if self._end == 0:
            self._end = self._duration
        self._clamp_all()
        self.update()

    def setCurrent(self, ms: int, emit=True):
        self._current = self._clamp(ms)
        self.update()
        if emit:
            self.currentChanged.emit(self._current)

    def setStart(self, ms: int, emit=True):
        self._start = self._clamp(ms)
        if self._start > self._end:
            self._start = self._end
        self.update()
        if emit:
            self.startChanged.emit(self._start)

    def setEnd(self, ms: int, emit=True):
        self._end = self._clamp(ms)
        if self._end < self._start:
            self._end = self._start
        self.update()
        if emit:
            self.endChanged.emit(self._end)

    def duration(self): return self._duration
    def current(self):  return self._current
    def start(self):    return self._start
    def end(self):      return self._end

    # -------- Internals --------
    def _clamp(self, ms: int) -> int:
        if self._duration <= 0:
            return 0
        return max(0, min(int(ms), self._duration))

    def _clamp_all(self):
        self._start = self._clamp(self._start)
        self._end = self._clamp(self._end)
        if self._end < self._start:
            self._end = self._start
        self._current = self._clamp(self._current)

    def _track_rect(self) -> QRectF:
        h = self.height()
        y = (h - self._track_height) / 2
        return QRectF(self._left_pad, y, self.width() - self._left_pad - self._right_pad, self._track_height)

    def _ms_to_x(self, ms: int) -> float:
        rect = self._track_rect()
        if self._duration <= 0:
            return rect.left()
        return rect.left() + (rect.width() * (ms / self._duration))

    def _x_to_ms(self, x: float) -> int:
        rect = self._track_rect()
        x = max(rect.left(), min(x, rect.right()))
        if rect.width() <= 0 or self._duration <= 0:
            return 0
        frac = (x - rect.left()) / rect.width()
        return int(round(frac * self._duration))

    def _resolve_collisions(self, raw_x):
        """
        raw_x: dict{handle: x} for {'start','current','end'}
        Returns display_x dict with minimal separation enforced.
        """
        rect = self._track_rect()
        items = [
            ["start",   raw_x["start"]],
            ["current", raw_x["current"]],
            ["end",     raw_x["end"]],
        ]
        items.sort(key=lambda t: t[1])

        # left-to-right pass: push right when too close
        for i in range(1, len(items)):
            left = items[i-1][1]
            cur = items[i][1]
            if cur - left < self._min_sep:
                items[i][1] = left + self._min_sep

        # right-to-left pass: pull left when too close
        for i in reversed(range(len(items)-1)):
            right = items[i+1][1]
            cur = items[i][1]
            if right - cur < self._min_sep:
                items[i][1] = right - self._min_sep

        # clamp to track bounds by shifting the cluster if needed
        left_bound = rect.left()
        right_bound = rect.right()
        min_x = min(v for _, v in items)
        max_x = max(v for _, v in items)
        shift = 0.0
        if min_x < left_bound:
            shift = left_bound - min_x
        elif max_x > right_bound:
            shift = right_bound - max_x
        if shift != 0.0:
            for it in items:
                it[1] += shift

        disp = {name: x for name, x in items}
        return disp

    def _triangle_points(self, x: float, up: bool = True):
        """Equilateral-ish triangle centered at track center."""
        r = self._handle_r
        cy = self._track_rect().center().y()
        if up:
            return [QPointF(x, cy - r), QPointF(x - r, cy + r), QPointF(x + r, cy + r)]
        else:
            return [QPointF(x - r, cy - r), QPointF(x + r, cy - r), QPointF(x, cy + r)]

    def _nearest_handle(self, x: float) -> str:
        # use display (collision-resolved) positions
        dists = {k: abs(self._disp_x[k] - x) for k in self._disp_x if self._disp_x[k] is not None}
        if not dists:
            return "current"
        handle, dist = min(dists.items(), key=lambda kv: kv[1])
        return handle if dist <= self._grab_tol else "current"

    # -------- Painting --------
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        # Track
        track = self._track_rect()
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(Qt.gray))
        p.drawRoundedRect(track, 3, 3)

        # Raw positions
        raw_x = {
            "start":   self._ms_to_x(self._start),
            "current": self._ms_to_x(self._current),
            "end":     self._ms_to_x(self._end),
        }
        # Resolve collisions for display & hit-testing
        self._disp_x = self._resolve_collisions(raw_x)

        # Selected segment fill (use raw positions for accuracy along the track)
        seg_left = min(raw_x["start"], raw_x["end"])
        seg_right = max(raw_x["start"], raw_x["end"])
        sel = QRectF(seg_left, track.top(), max(2.0, seg_right - seg_left), track.height())
        p.setBrush(QBrush(Qt.darkGray))
        p.drawRoundedRect(sel, 3, 3)

        # Draw triangles (all up-pointing)
        p.setPen(self._pen)
        for name in ("start", "current", "end"):
            x = self._disp_x[name]
            p.setBrush(QBrush(self._colors[name]))
            pts = self._triangle_points(x, up=True)
            p.drawPolygon(*pts)

    # -------- Interaction --------
    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton:
            return
        handle = self._nearest_handle(e.x())
        self._dragging = handle

        ms = self._x_to_ms(e.x())
        if handle == "current":
            self.setCurrent(ms)
        elif handle == "start":
            self.setStart(min(ms, self._end))
        elif handle == "end":
            self.setEnd(max(ms, self._start))
        self.update()

    def mouseMoveEvent(self, e):
        if not (e.buttons() & Qt.LeftButton) or self._dragging is None:
            return
        ms = self._x_to_ms(e.x())
        if self._dragging == "current":
            self.setCurrent(ms)
        elif self._dragging == "start":
            self.setStart(min(ms, self._end))
        elif self._dragging == "end":
            self.setEnd(max(ms, self._start))
        self.update()

    def mouseReleaseEvent(self, _):
        self._dragging = None


# ===========================================
#              VideoSlicer Window
# ===========================================
class VideoSlicer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Slicer (3 markers, frame-accurate scrub)")

        # --- Core widgets
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.videoWidget = QVideoWidget()

        # Seek throttling for QMediaPlayer
        self._seek_pending_ms: Optional[int] = None
        self._seek_timer = QTimer(self)
        self._seek_timer.setInterval(16)  # ~60fps throttle
        self._seek_timer.setSingleShot(True)
        self._seek_timer.timeout.connect(self._flush_seek)

        # Controls
        self.openBtn = QPushButton("Open Video")
        self.sliceBtn = QPushButton("Slice and Save")
        self.replayBtn = QPushButton("Replay Segment")
        self.playPauseBtn = QPushButton("Play/Pause")
        self.markInBtn = QPushButton("Set Start = Current")
        self.markOutBtn = QPushButton("Set End = Current")

        # Labels
        self.curLabel = QLabel("Current: 0 ms")
        self.startLabel = QLabel("Start: 0 ms")
        self.endLabel = QLabel("End: 0 ms")

        # 3-marker timeline
        self.timeline = ThreeMarkerSlider()
        self.timeline.setEnabled(False)

        # Layout
        root = QVBoxLayout(self)
        root.addWidget(self.videoWidget)
        root.addWidget(self.timeline)

        info = QHBoxLayout()
        info.addWidget(self.curLabel)
        info.addStretch(1)
        info.addWidget(self.startLabel)
        info.addWidget(self.endLabel)
        root.addLayout(info)

        row = QHBoxLayout()
        row.addWidget(self.openBtn)
        row.addWidget(self.playPauseBtn)
        row.addWidget(self.replayBtn)
        row.addWidget(self.markInBtn)
        row.addWidget(self.markOutBtn)
        row.addStretch(1)
        row.addWidget(self.sliceBtn)
        root.addLayout(row)

        # Wire up media
        self.mediaPlayer.setVideoOutput(self.videoWidget)
        self.mediaPlayer.durationChanged.connect(self._on_duration)
        self.mediaPlayer.positionChanged.connect(self._on_position)
        self.mediaPlayer.setNotifyInterval(16)

        # Wire up buttons
        self.openBtn.clicked.connect(self.open_file)
        self.sliceBtn.clicked.connect(self.slice_video)
        self.replayBtn.clicked.connect(self.replay_segment)
        self.playPauseBtn.clicked.connect(self.toggle_play_pause)
        self.markInBtn.clicked.connect(self.set_start_from_current)
        self.markOutBtn.clicked.connect(self.set_end_from_current)

        # Wire up timeline signals
        self.timeline.currentChanged.connect(self._on_user_scrub_current)
        self.timeline.startChanged.connect(self._on_user_set_start)
        self.timeline.endChanged.connect(self._on_user_set_end)

        # Stop at end marker while replaying
        self.playback_timer = QTimer(self)
        self.playback_timer.setInterval(40)
        self.playback_timer.timeout.connect(self._check_segment_end)

        self._updating_from_player = False
        self.video_path = None

        # ---- Frame-accurate preview overlay (OpenCV)
        self.preview = QLabel(self.videoWidget)
        self.preview.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.preview.setStyleSheet("background: transparent;")
        self.preview.hide()

        self._cap = None
        self._fps = 0.0
        self._total_frames = 0

    # ----- UI actions -----
    def open_file(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "", "Video Files (*.mp4 *.mov *.m4v *.avi *.mkv)"
        )
        if not file:
            return
        self.video_path = file
        self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(file)))
        self.mediaPlayer.play()  # start to populate duration/position
        self.timeline.setEnabled(True)

        # Try to open with OpenCV for frame-accurate preview
        if _CV2_AVAILABLE:
            try:
                if self._cap is not None:
                    self._cap.release()
                self._cap = cv2.VideoCapture(self.video_path)
                if self._cap.isOpened():
                    self._fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
                    self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                else:
                    self._cap = None
                    self._fps = 0.0
                    self._total_frames = 0
            except Exception:
                self._cap = None
                self._fps = 0.0
                self._total_frames = 0

    def set_start_from_current(self):
        ms = self.timeline.current()
        self.timeline.setStart(ms)
        self.startLabel.setText(f"Start: {ms} ms")

    def set_end_from_current(self):
        ms = self.timeline.current()
        self.timeline.setEnd(ms)
        self.endLabel.setText(f"End: {ms} ms")

    def toggle_play_pause(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.preview.hide()
            self.mediaPlayer.play()

    # ----- Media callbacks -----
    def _on_duration(self, duration_ms: int):
        self.timeline.setDuration(duration_ms)
        if self.timeline.end() == 0:
            self.timeline.setEnd(duration_ms)
        self.endLabel.setText(f"End: {self.timeline.end()} ms")

    def _on_position(self, pos_ms: int):
        # Update slider without feedback loop
        self._updating_from_player = True
        self.timeline.setCurrent(pos_ms, emit=False)
        self._updating_from_player = False
        self.curLabel.setText(f"Current: {pos_ms} ms")

        # Hide preview while actually playing
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.preview.hide()

        # Stop at end marker during replay
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState and pos_ms >= self.timeline.end():
            self.mediaPlayer.pause()
            self.playback_timer.stop()

    # ----- Timeline callbacks (user scrubbing) -----
    def _on_user_scrub_current(self, ms: int):
        # throttle + pause seek for the platform backend
        self._queue_seek(ms)
        self.curLabel.setText(f"Current: {ms} ms")
        # exact frame preview
        self._show_exact_preview(ms)

    def _on_user_set_start(self, ms: int):
        self.startLabel.setText(f"Start: {ms} ms")

    def _on_user_set_end(self, ms: int):
        self.endLabel.setText(f"End: {ms} ms")

    # ----- Segment replay -----
    def replay_segment(self):
        if not self.video_path:
            return
        self.preview.hide()
        self.mediaPlayer.setPosition(self.timeline.start())
        self.mediaPlayer.play()
        self.playback_timer.start()

    def _check_segment_end(self):
        if self.mediaPlayer.position() >= self.timeline.end():
            self.mediaPlayer.pause()
            self.playback_timer.stop()

    # ----- Slice with best-available backend -----
    def slice_video(self):
        if not self.video_path:
            return
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Sliced Video", "", "Video Files (*.mp4)")
        if not save_path:
            return

        start_sec = self.timeline.start() / 1000.0
        end_sec = self.timeline.end() / 1000.0
        if end_sec <= start_sec:
            self._show_status("End must be greater than Start.")
            return

        # Prefer ffmpeg "copy" for speed + audio
        if self._has_ffmpeg():
            cmd = [
                "ffmpeg", "-y",
                "-i", self.video_path,
                "-ss", f"{start_sec:.3f}",
                "-to", f"{end_sec:.3f}",
                "-c", "copy",
                save_path
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True)
                if proc.returncode != 0:
                    self._show_status("ffmpeg error; trying MoviePy/OpenCV...\n" + proc.stderr.decode(errors="ignore"))
                    self._slice_with_python(save_path, start_sec, end_sec)
                else:
                    self._show_status("Done!")
            except Exception as e:
                self._show_status(f"ffmpeg exception; trying MoviePy/OpenCV... {e}")
                self._slice_with_python(save_path, start_sec, end_sec)
        else:
            # No ffmpeg CLI — try MoviePy, else OpenCV fallback
            self._slice_with_python(save_path, start_sec, end_sec)

    def _slice_with_python(self, save_path: str, start_sec: float, end_sec: float):
        # MoviePy first (keeps audio), else OpenCV (video only)
        if _MOVIEPY_AVAILABLE:
            try:
                clip = mpy.VideoFileClip(self.video_path).subclip(start_sec, end_sec)
                # write_videofile will re-encode; default uses installed ffmpeg binary
                clip.write_videofile(save_path, codec="libx264", audio_codec="aac", temp_audiofile="temp-audio.m4a", remove_temp=True)
                clip.close()
                self._show_status("Done! (MoviePy)")
                return
            except Exception as e:
                self._show_status(f"MoviePy failed; trying OpenCV (video only). {e}")

        if _CV2_AVAILABLE:
            try:
                self._slice_with_opencv(save_path, start_sec, end_sec)
                self._show_status("Done (OpenCV video only; no audio).")
            except Exception as e:
                self._show_status(f"OpenCV slicing failed: {e}")
        else:
            self._show_status("No available backend to slice (need ffmpeg or moviepy or opencv).")

    def _slice_with_opencv(self, save_path: str, start_sec: float, end_sec: float):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError("OpenCV failed to open video.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

        start_idx = int(round(start_sec * fps))
        end_idx = int(round(end_sec * fps))
        start_idx = max(0, min(start_idx, max(0, total - 1)))
        end_idx = max(0, min(end_idx, max(0, total - 1)))
        if end_idx <= start_idx:
            end_idx = min(start_idx + 1, total - 1)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        idx = start_idx
        while idx <= end_idx:
            ok, frame = cap.read()
            if not ok:
                break
            out.write(frame)
            idx += 1

        out.release()
        cap.release()

    # ----- Frame-accurate preview (OpenCV) -----
    def _show_exact_preview(self, ms: int):
        if not (_CV2_AVAILABLE and self._cap is not None and self._fps > 0):
            self.preview.hide()
            return

        frame_idx = int(round((ms / 1000.0) * self._fps))
        if self._total_frames > 0:
            frame_idx = max(0, min(frame_idx, self._total_frames - 1))

        # Random access
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self.preview.hide()
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        qimg = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)

        pix = QPixmap.fromImage(qimg).scaled(
            self.videoWidget.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.preview.setPixmap(pix)
        self.preview.resize(pix.size())
        self.preview.move(
            (self.videoWidget.width() - pix.width()) // 2,
            (self.videoWidget.height() - pix.height()) // 2
        )
        self.preview.show()

    # Keep preview centered on resize
    def resizeEvent(self, e):
        super().resizeEvent(e)
        if not self.preview.isHidden() and self.preview.pixmap():
            pix = self.preview.pixmap()
            if pix:
                self.preview.move(
                    (self.videoWidget.width() - pix.width()) // 2,
                    (self.videoWidget.height() - pix.height()) // 2
                )

    # ----- Seek throttling (QMediaPlayer) -----
    def _queue_seek(self, ms: int):
        if self.mediaPlayer.state() != QMediaPlayer.PausedState:
            self.mediaPlayer.pause()
        self._seek_pending_ms = ms
        if self._seek_timer.isActive():
            self._seek_timer.stop()
        self._seek_timer.start()

    def _flush_seek(self):
        if self._seek_pending_ms is None:
            return
        ms = self._seek_pending_ms
        self._seek_pending_ms = None
        # nudge helps some backends settle on exact frame
        self.mediaPlayer.setPosition(max(0, ms - 1))
        self.mediaPlayer.setPosition(ms)

    # ----- Utilities -----
    def _has_ffmpeg(self) -> bool:
        # Detect whether ffmpeg executable is available on PATH
        return shutil.which("ffmpeg") is not None

    def _show_status(self, text: str):
        self.setWindowTitle(f"Video Slicer — {text}")
        toast = QLabel(text, self)
        toast.setStyleSheet("""
            QLabel {
                background-color: rgba(0, 0, 0, 180);
                color: white;
                padding: 10px 18px;
                border-radius: 8px;
                font-size: 14px;
            }
        """)

        toast.setWindowFlags(
            Qt.Tool |
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.X11BypassWindowManagerHint
        )
        toast.setAttribute(Qt.WA_TranslucentBackground)
        toast.setAlignment(Qt.AlignCenter)
        parent_rect = self.geometry()
        toast.adjustSize()
        toast.move(
            parent_rect.x() + (parent_rect.width() - toast.width()) // 2,
            parent_rect.y() + (parent_rect.height() - toast.height()) // 2,
        )
        toast.show()
        QTimer.singleShot(4000, toast.close)

# ===========================================
#                     main
# ===========================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = VideoSlicer()
    win.resize(960, 640)
    win.show()
    sys.exit(app.exec_())
