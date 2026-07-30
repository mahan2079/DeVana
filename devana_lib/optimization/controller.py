import threading
from typing import Optional

class ExecutionController:
    """
    Manages the lifecycle of a headless AlgorithmEngine.
    Thread-safe implementation allowing UI/REST to pause/resume/abort.
    """
    def __init__(self) -> None:
        self._abort: bool = False
        self._abort_lock: threading.Lock = threading.Lock()
        self._pause_event: threading.Event = threading.Event()
        self._pause_event.set() # Set means "not paused" (allowed to proceed)

    def should_abort(self) -> bool:
        with self._abort_lock:
            return self._abort

    def abort(self) -> None:
        with self._abort_lock:
            self._abort = True
        # Ensure we unblock if aborted while paused
        self._pause_event.set()

    def pause(self) -> None:
        self._pause_event.clear()

    def resume(self) -> None:
        self._pause_event.set()

    def wait_if_paused(self, timeout: Optional[float] = None) -> bool:
        """
        Blocks the calling thread (the engine thread) if paused.
        Returns immediately if not paused.
        Raises TimeoutError if the timeout expires while waiting.
        """
        success: bool = self._pause_event.wait(timeout)
        if not success:
            raise TimeoutError("Timed out waiting for engine to resume.")
        return success
