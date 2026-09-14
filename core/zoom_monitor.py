import threading
import time
import subprocess

class ZoomMonitor:
    def __init__(self, polling_interval=1.0):
        self.polling_interval = polling_interval
        # Default to UNMUTED so that if the AppleScript fails (e.g. no permissions),
        # we still record audio instead of silently recording nothing.
        self.is_muted = False
        self._running = False
        self._thread = None
        self._permission_warning_shown = False

    def _check_mute_status_applescript(self):
        """
        Runs AppleScript to check if 'Mute audio' is present in the Zoom meeting menu.
        Returns True if muted, False if unmuted, or None on error/not found.
        """
        script = """
        tell application "System Events"
            if exists application process "zoom.us" then
                tell application process "zoom.us"
                    if exists (menu bar 1) then
                        if exists (menu bar item "Meeting" of menu bar 1) then
                            if exists (menu item "Mute audio" of menu 1 of menu bar item "Meeting" of menu bar 1) then
                                return "Unmuted"
                            else if exists (menu item "Unmute audio" of menu 1 of menu bar item "Meeting" of menu bar 1) then
                                return "Muted"
                            end if
                        end if
                    end if
                end tell
            end if
        end tell
        return "Unknown"
        """
        try:
            result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
            output = result.stdout.strip()
            if output == "Unmuted":
                return False
            elif output == "Muted":
                return True
            else:
                return self.is_muted # Keep previous state if unknown (e.g., no meeting active)
        except Exception as e:
            if not self._permission_warning_shown:
                print(f"\n[Warning] Could not check Zoom mute status: {e}")
                print("[Warning] This is usually because Terminal lacks Accessibility permissions, or Zoom isn't running.")
                print("[Warning] Defaulting to UNMUTED for recording.")
                self._permission_warning_shown = True
            return self.is_muted # Keep previous state on error

    def _monitor_loop(self):
        while self._running:
            self.is_muted = self._check_mute_status_applescript()
            time.sleep(self.polling_interval)

    def start(self):
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()

if __name__ == "__main__":
    monitor = ZoomMonitor(polling_interval=1.0)
    monitor.start()
    print("Starting Zoom Monitor... Press Ctrl+C to stop.")
    try:
        while True:
            print(f"Zoom Muted: {monitor.is_muted}")
            time.sleep(1)
    except KeyboardInterrupt:
        monitor.stop()
        print("\nZoom Monitor stopped.")
