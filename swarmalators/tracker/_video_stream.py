from threading import Thread
import time
import cv2
import subprocess, os


class VideoStream:
    """A CV2 VideoStream wrapper for threading.

    Attributes:
        device: The device index of the camera to use.
    """

    def __init__(self, device: int, settings: str):
        """
        Initialize the video stream and the camera settings.

        Args:
            device (int): The device index of the camera to use.
            settings (str): The path to the camera settings file.
        """

        is_windows = os.name == "nt"

        if is_windows:
            self._cam = cv2.VideoCapture(device, cv2.CAP_DSHOW)
        else:
            self._cam = cv2.VideoCapture(device)

        self._apply_uvc_settings(settings)

        self._stopped = False
        self._frame = None

    def start(self):
        """Start the thread to read frames from the video stream."""
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        """Keep looping indefinitely until the thread is stopped."""
        while True:
            if self._stopped:
                return

            ret, frame = self._cam.read()
            if ret:
                self._frame = frame

    def read(self):
        """Return the current frame."""
        return self._frame

    def stop(self):
        """Indicate that the thread should be stopped."""
        self._stopped = True
        # Wait a moment to avoid segfaults
        time.sleep(0.5)
        # Release the stream
        self._cam.release()
        # Wait a moment to avoid segfaults
        time.sleep(1)

    """
    Private methods
    """

    def _apply_uvc_settings(self, settings: str):
        """
        Uses the uvcc tool to apply camera settings.

        Args:
            settings (str): The path to the camera settings file.
        """

        # Apply camera settings
        input_config = open(settings, "r")

        is_windows = os.name == "nt"

        # Needs to be run twice to apply settings sometimes
        res = subprocess.run(
            ["uvcc", "--product", "2115", "import"],
            shell=is_windows,
            stdin=input_config,
            capture_output=True,
            text=True,
        )

        input_config.seek(0)

        res = subprocess.run(
            ["uvcc", "--product", "2115", "import"],
            shell=is_windows,
            stdin=input_config,
            capture_output=True,
            text=True,
        )
        input_config.close()

        print(res.stdout)
        print(res.stderr)

        # Allow time for the camera to apply settings
        time.sleep(1)
