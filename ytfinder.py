import os
import re
import threading
import subprocess
import shutil
import tkinter as tk
from tkinter import messagebox, filedialog
from typing import Optional

# ---------- Utilities ----------

TIME_RE = re.compile(r"^(?:(\d{1,2}):)?([0-5]?\d):([0-5]?\d)$")  # H:MM:SS or MM:SS

def ensure_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(
            f"Required tool '{name}' not found in PATH.\n"
            f"Please install it and restart the app."
        )

def parse_time_to_seconds(ts: str) -> int:
    """
    Accepts H:MM:SS or MM:SS. Returns seconds as int. Raises ValueError on invalid input.
    """
    m = TIME_RE.match(ts.strip())
    if not m:
        raise ValueError(f"Invalid time '{ts}'. Use H:MM:SS or MM:SS, e.g. 0:15 or 1:02:03")
    h = int(m.group(1)) if m.group(1) else 0
    mm = int(m.group(2))
    ss = int(m.group(3))
    return h * 3600 + mm * 60 + ss

def safe_output_path(path: str) -> str:
    """
    If path exists, auto-increment: "name (1).ext", "name (2).ext", ...
    """
    base, ext = os.path.splitext(path)
    if not ext:
        ext = ".mp4"
    out = base + ext
    i = 1
    while os.path.exists(out):
        out = f"{base} ({i}){ext}"
        i += 1
    return out

def run_cmd(cmd: list[str]) -> None:
    """
    Run a subprocess, raising RuntimeError with helpful stderr on failure.
    """
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        stderr = (result.stderr or b"").decode(errors="replace").strip()
        stdout = (result.stdout or b"").decode(errors="replace").strip()
        msg = stderr or stdout or "Unknown error"
        raise RuntimeError(msg)

def get_video_id(url: str) -> str:
    """
    Use yt-dlp to resolve the canonical video ID (stable cache key).
    """
    cmd = ["yt-dlp", "--get-id", url]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        err = (result.stderr or b"").decode(errors="replace")
        raise RuntimeError(f"yt-dlp failed to get video ID:\n{err}")
    vid = (result.stdout or b"").decode().strip()
    if not vid:
        raise RuntimeError("Could not determine video ID.")
    return vid

def cache_path_for(url: str) -> str:
    vid = get_video_id(url)
    return f"cache_{vid}.mp4"

def download_full_video_if_needed(url: str, cache_path: str) -> None:
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        return
    # Best available; remux to mp4 when possible to keep trimming simple
    cmd = [
        "yt-dlp",
        "-f", "bestvideo*+bestaudio/best",
        "--merge-output-format", "mp4",
        "-o", cache_path,
        url
    ]
    run_cmd(cmd)

def trim_segment_fast(cache_path: str, start: int, end: int, output_path: str) -> None:
    """
    Fast, stream-copy trim. Places -ss before -i. Accuracy is GOP-limited but very fast.
    """
    dur = max(0, end - start)
    if dur <= 0:
        raise ValueError("End time must be greater than start time.")

    # -nostdin prevents ffmpeg from waiting for user input on errors
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-ss", str(start),
        "-i", cache_path,
        "-t", str(dur),
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        output_path
    ]
    run_cmd(cmd)

# ---------- GUI App ----------

class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("YouTube Segment Downloader")

        # Grid config
        root.columnconfigure(1, weight=1)

        # Inputs
        tk.Label(root, text="YouTube URL:").grid(row=0, column=0, sticky="e", padx=6, pady=4)
        self.url_entry = tk.Entry(root, width=60)
        self.url_entry.grid(row=0, column=1, sticky="we", padx=6, pady=4)

        tk.Label(root, text="Start (H:MM:SS or MM:SS):").grid(row=1, column=0, sticky="e", padx=6, pady=4)
        self.start_entry = tk.Entry(root)
        self.start_entry.grid(row=1, column=1, sticky="we", padx=6, pady=4)

        tk.Label(root, text="End (H:MM:SS or MM:SS):").grid(row=2, column=0, sticky="e", padx=6, pady=4)
        self.end_entry = tk.Entry(root)
        self.end_entry.grid(row=2, column=1, sticky="we", padx=6, pady=4)

        tk.Label(root, text="Output file (.mp4):").grid(row=3, column=0, sticky="e", padx=6, pady=4)
        self.output_entry = tk.Entry(root)
        self.output_entry.grid(row=3, column=1, sticky="we", padx=6, pady=4)

        browse_btn = tk.Button(root, text="Browse…", command=self.on_browse)
        browse_btn.grid(row=3, column=2, padx=4)

        self.download_btn = tk.Button(root, text="Download Segment", command=self.on_download)
        self.download_btn.grid(row=4, column=0, columnspan=3, pady=10)

        self.clear_cache_btn = tk.Button(root, text="Clear Cache for URL", command=self.on_clear_cache)
        self.clear_cache_btn.grid(row=5, column=0, columnspan=3, pady=4)

        self.status_var = tk.StringVar(value="Ready.")
        self.status_lbl = tk.Label(root, textvariable=self.status_var, anchor="w")
        self.status_lbl.grid(row=6, column=0, columnspan=3, sticky="we", padx=6, pady=6)

    def set_status(self, text: str):
        self.status_var.set(text)
        self.root.update_idletasks()

    def on_browse(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
        )
        if path:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, path)

    def on_clear_cache(self):
        url = self.url_entry.get().strip()
        if not url:
            messagebox.showerror("Error", "Please enter a URL to clear its cache.")
            return
        try:
            cache_path = cache_path_for(url)
            if os.path.exists(cache_path):
                os.remove(cache_path)
                messagebox.showinfo("Cache", f"Removed cache: {cache_path}")
            else:
                messagebox.showinfo("Cache", "No cache file found for this URL.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def on_download(self):
        # Kick heavy work to a background thread
        t = threading.Thread(target=self._do_download, daemon=True)
        t.start()

    # ...existing code in App class...

    def _do_download(self):
        try:
            self.download_btn.config(state="disabled")
            self.clear_cache_btn.config(state="disabled")
            self.set_status("Validating…")

            ensure_tool("yt-dlp")
            ensure_tool("ffmpeg")

            url = self.url_entry.get().strip()
            start_s = parse_time_to_seconds(self.start_entry.get())
            end_s = parse_time_to_seconds(self.end_entry.get())
            if end_s <= start_s:
                raise ValueError("End time must be greater than start time.")

            requested_output = self.output_entry.get().strip()
            if not requested_output:
                raise ValueError("Please provide an output file name (e.g., clip.mp4).")

            out_path = safe_output_path(requested_output)

            self.set_status("Resolving video ID and cache path…")
            cache_path = cache_path_for(url)

            self.set_status("Downloading original (if not cached)…")
            download_full_video_if_needed(url, cache_path)

            self.set_status("Trimming segment…")
            trim_segment_fast(cache_path, start_s, end_s, out_path)

            self.set_status("Done.")
            # Schedule messagebox on main thread
            self.root.after(0, lambda: messagebox.showinfo("Success", f"Saved to:\n{out_path}"))
        except Exception as e:
            self.set_status("Error.")
            # Schedule messagebox on main thread
            self.root.after(0, lambda err=e: messagebox.showerror("Error", str(err)))
        finally:
            self.download_btn.config(state="normal")
            self.clear_cache_btn.config(state="normal")

# ---------- Main ----------

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
