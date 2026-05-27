import os
import threading
import time


class _BgReviewSpinner:
    """Spinner that writes to /dev/tty above the prompt_toolkit input area.

    Uses ANSI cursor-save (\033[s) / cursor-restore (\033[u) sequences so
    the animation appears on the line just above the input box without
    overwriting the user's typed text or displacing the input cursor.
    Falls back gracefully (no-op writes) when /dev/tty is unavailable.
    """
    FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']

    def __init__(self, message: str = "background review"):
        self.message = message
        self.running = False
        self.thread: threading.Thread | None = None
        self._idx = 0
        self._start: float = 0.0

    # -- low-level write ------------------------------------------------

    @staticmethod
    def _tty_write(text: str) -> None:
        try:
            fd = os.open("/dev/tty", os.O_WRONLY)
            try:
                os.write(fd, text.encode("utf-8", errors="replace"))
            finally:
                os.close(fd)
        except (OSError, IOError):
            pass

    # -- animation loop ------------------------------------------------

    def _animate(self) -> None:
        while self.running:
            frame = self.FRAMES[self._idx % len(self.FRAMES)]
            elapsed = time.time() - self._start
            line = f"  {frame} {self.message} ({elapsed:.1f}s)"
            # \033[s = save cursor (at input box)
            # \033[A = cursor up one line
            # \r    = carriage return to start of line
            # \033[u = restore cursor (back to input box)
            self._tty_write(f"\033[s\033[A\r{line}\033[u")
            self._idx += 1
            time.sleep(0.12)
        # Clear the status line
        self._tty_write("\033[s\033[A\r\033[2K\033[u")

    # -- public API ----------------------------------------------------

    def start(self) -> None:
        self.running = True
        self._start = time.time()
        self.thread = threading.Thread(target=self._animate, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.5)
        # Clear the spinner line (final message is printed via
        # agent._safe_print() so it survives prompt_toolkit redraws).
