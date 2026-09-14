import os
import Quartz
import cv2
import numpy as np
import threading
import time

class VisualIngestion:
    """Captures periodic downscaled frames of the active Zoom meeting window to
    `frames_dir`, named by elapsed milliseconds. The vision-LLM name reader
    consumes these in post-processing; no OCR happens here."""

    def __init__(self, polling_interval=2.0, min_window_size=300, frames_dir=None,
                 frame_interval=4.0, max_frame_width=1280):
        self.polling_interval = polling_interval
        self.min_window_size = min_window_size
        self.frames_dir = frames_dir
        self.frame_interval = frame_interval          # seconds between saved frames
        self.max_frame_width = max_frame_width
        self._running = False
        self._thread = None
        self.frames = []                              # [(elapsed_seconds, path)]
        self.frame_count = 0
        self._t0 = None
        self._last_save = None
        self._cached_window_id = None
        self._green_confirmed = False

    def _list_zoom_windows(self):
        window_list = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID
        )
        candidates = []
        for w in window_list:
            if w.get('kCGWindowOwnerName') != 'zoom.us':
                continue
            if w.get('kCGWindowLayer') != 0:
                continue
            bounds = w.get('kCGWindowBounds')
            if not bounds:
                continue
            ww = bounds.get('Width', 0)
            wh = bounds.get('Height', 0)
            if ww >= self.min_window_size and wh >= self.min_window_size:
                candidates.append(w)
        return candidates

    def _measure_green_border(self, img_bgra):
        img_bgr = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0
        return max(cv2.contourArea(c) for c in contours)

    def _pick_meeting_window(self, candidates):
        if len(candidates) == 1:
            # Single window: assume it's the meeting, skip green check
            return candidates[0], True

        best_win = None
        best_green = 0
        for win in candidates:
            img = self._capture_window(win)
            if img is None:
                continue
            green = self._measure_green_border(img)
            wid = win['kCGWindowNumber']
            print(f"[VisualIngestion] Window ID {wid}: green area = {green:.0f} px²")
            if green > best_green:
                best_green = green
                best_win = win

        if best_win and best_green > 1000:
            return best_win, True

        fallback = max(candidates, key=lambda w: (
            w['kCGWindowBounds']['Width'] * w['kCGWindowBounds']['Height']
        ))
        return fallback, False

    def _find_zoom_window(self):
        if self._cached_window_id is not None:
            window_list = Quartz.CGWindowListCopyWindowInfo(
                Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID
            )
            for w in window_list:
                if w.get('kCGWindowNumber') == self._cached_window_id:
                    # Window still exists
                    if self._green_confirmed:
                        return w  # Green-verified: fast path
                    # Assumed (single-window): check if count changed
                    if len(self._list_zoom_windows()) == 1:
                        return w  # Still single window, keep using it
                    # Multiple windows now — fall through to re-scan with green
                    break

            print(f"[VisualIngestion] Window switched: cached window ID {self._cached_window_id} "
                  f"gone — re-scanning")
            self._cached_window_id = None
            self._green_confirmed = False

        candidates = self._list_zoom_windows()
        if not candidates:
            return None

        chosen, has_green = self._pick_meeting_window(candidates)
        if has_green:
            self._cached_window_id = chosen['kCGWindowNumber']
            self._green_confirmed = len(candidates) > 1
            bounds = chosen['kCGWindowBounds']
            if len(candidates) > 1:
                print(f"[VisualIngestion] Green border confirmed — meeting window ID "
                      f"{self._cached_window_id} ({bounds['Width']}x{bounds['Height']})")
            else:
                print(f"[VisualIngestion] Single Zoom window — using ID "
                      f"{self._cached_window_id} ({bounds['Width']}x{bounds['Height']})")
        return chosen

    def _capture_window(self, window_info):
        window_id = window_info['kCGWindowNumber']
        bounds = window_info['kCGWindowBounds']
        cg_rect = Quartz.CGRectMake(
            bounds['X'], bounds['Y'], bounds['Width'], bounds['Height']
        )
        cg_image = Quartz.CGWindowListCreateImage(
            cg_rect,
            Quartz.kCGWindowListOptionIncludingWindow,
            window_id,
            Quartz.kCGWindowImageBoundsIgnoreFraming | Quartz.kCGWindowImageNominalResolution
        )
        if not cg_image:
            return None
        width = Quartz.CGImageGetWidth(cg_image)
        height = Quartz.CGImageGetHeight(cg_image)
        provider = Quartz.CGImageGetDataProvider(cg_image)
        data = Quartz.CGDataProviderCopyData(provider)
        if not data:
            return None
        bytes_per_row = Quartz.CGImageGetBytesPerRow(cg_image)
        buffer = np.frombuffer(data, dtype=np.uint8)
        try:
            img = buffer.reshape((height, bytes_per_row // 4, 4))
            return img[:, :width, :]
        except ValueError:
            return None

    def _save_frame(self, img_bgra, elapsed):
        img_bgr = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)
        h, w = img_bgr.shape[:2]
        if w > self.max_frame_width:
            scale = self.max_frame_width / w
            img_bgr = cv2.resize(img_bgr, (self.max_frame_width, int(h * scale)))
        os.makedirs(self.frames_dir, exist_ok=True)
        path = os.path.join(self.frames_dir, f"{int(elapsed * 1000):09d}.jpg")
        cv2.imwrite(path, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        self.frames.append((elapsed, path))
        self.frame_count += 1

    def _ingestion_loop(self):
        while self._running:
            if self.frames_dir:
                now = time.monotonic()
                if self._last_save is None or (now - self._last_save) >= self.frame_interval:
                    zoom_win = self._find_zoom_window()
                    if zoom_win is not None:
                        img = self._capture_window(zoom_win)
                        if img is not None:
                            self._save_frame(img, now - self._t0)
                            self._last_save = now
            time.sleep(self.polling_interval)

    def start(self):
        if not self._running:
            self._running = True
            self._t0 = time.monotonic()
            self._thread = threading.Thread(target=self._ingestion_loop, daemon=True)
            self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()
