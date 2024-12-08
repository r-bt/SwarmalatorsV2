from threading import Thread
import time
import uvc, logging, json
from typing import NamedTuple, Optional
import cv2


class CameraSpec(NamedTuple):
    idProduct: int
    idVendor: int
    width: int
    height: int
    fps: int
    bandwidth_factor: float = 2.0
    controls: list = []


UVCC_TO_PYUVC_MAPPING = {
    "absolute_exposure_time": "Absolute Exposure Time",
    "absolute_focus": "Absolute Focus",
    "absolute_zoom": "Zoom absolute control",
    "auto_exposure_mode": "Auto Exposure Mode",
    "auto_exposure_priority": "Auto Exposure Priority",
    "auto_focus": "Auto Focus",
    "auto_white_balance_temperature": "White Balance temperature,Auto",
    "backlight_compensation": "Backlight Compensation",
    "brightness": "Brightness",
    "contrast": "Contrast",
    "gain": "Gain",
    "power_line_frequency": "Power Line frequency",
    "saturation": "Saturation",
    "sharpness": "Sharpness",
    "white_balance_temperature": "White Balance temperature",
}


class VideoStream:
    """A CV2 VideoStream wrapper for threading.

    Attributes:
        device: The device index of the camera to use.
    """

    def __init__(self, settings: str):
        """
        Initialize the video stream and the camera settings.

        Args:
            device (int): The device index of the camera to use.
            settings (str): The path to the camera settings file.
        """

        self._controls = self._load_camera_controls(settings)

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
        camera = CameraSpec(2115, 1133, 1920, 1080, 30, 2.0, self._controls)

        cap = self._init_camera(camera)

        if cap is None:
            return

        while True:
            if self._stopped:
                break

            frame = cap.get_frame_robust()

            self._frame = frame

        cap.close()

    def read(self):
        """Return the current frame."""
        if self._frame is None:
            return None

        return cv2.rotate(self._frame.bgr, cv2.ROTATE_180)

    def stop(self):
        """Indicate that the thread should be stopped."""
        self._stopped = True
        # Wait a moment to avoid segfaults
        time.sleep(1.5)

    """
    Private methods
    """

    def _init_camera(self, camera: CameraSpec) -> Optional[uvc.Capture]:
        """
        Initialize a camera using the UVC library.

        Args:
            camera (CameraSpec): The camera specifications.
        """
        for device in uvc.device_list():
            if (
                device["idProduct"] == camera.idProduct
                and device["idVendor"] == camera.idVendor
            ):
                capture = uvc.Capture(device["uid"])

                # Set the bandwidth factor
                capture.bandwidth_factor = camera.bandwidth_factor

                # Select the correct mode
                for mode in capture.available_modes:
                    if mode[:3] == camera[2:5]:  # compare width, height, fps
                        capture.frame_mode = mode
                        break
                else:
                    logging.warning(
                        f"None of the available modes matched: {capture.available_modes}"
                    )
                    capture.close()
                    return

                time.sleep(1)

                # Set the camera controls
                controls_dict = dict([(c.display_name, c) for c in capture.controls])

                for control, value in camera.controls:
                    if control in controls_dict:
                        controls_dict[control].value = value
                    else:
                        logging.warning(f"Control not found: {control}")

                return capture
        else:
            logging.warning(f"Camera not found: {camera}")

    def _load_camera_controls(self, settings_path: str):
        """
        Loads the camera controls from a JSON file.
        """
        with open(settings_path, "r") as f:
            data = json.load(f)

            controls = []

            for key, value in data.items():
                if key in UVCC_TO_PYUVC_MAPPING:
                    controls.append((UVCC_TO_PYUVC_MAPPING[key], value))
                elif key == "absolute_pan_tilt":
                    controls.append(("Pan control", value[0]))
                    controls.append(("Tilt control", value[1]))

            # Remove absolute_exposure_time if auto exposure mode is enabled
            controls.sort(key=lambda x: x[0] != "Auto Exposure Mode")
            if data["auto_exposure_mode"] == 8:
                controls = [c for c in controls if c[0] != "Absolute Exposure Time"]

            # Remove absolute_focus if auto focus is enabled
            controls.sort(key=lambda x: x[0] != "Auto Focus")
            if data["auto_focus"] == 1:
                controls = [c for c in controls if c[0] != "Absolute Focus"]

            # Remove white_balance_temperature if auto white balance temperature is enabled
            controls.sort(key=lambda x: x[0] != "White Balance temperature,Auto")
            if data["auto_white_balance_temperature"] == 1:
                controls = [c for c in controls if c[0] != "White Balance temperature"]

            return controls
