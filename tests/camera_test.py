import uvc, cv2, time
from typing import NamedTuple, Optional
import logging
import json, pdb, threading


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


def load_camera_controls(settings_path: str):
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


class CameraSpec(NamedTuple):
    idProduct: int
    idVendor: int
    width: int
    height: int
    fps: int
    bandwidth_factor: float = 2.0
    controls: list = []


def init_camera(camera: CameraSpec) -> Optional[uvc.Capture]:
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

            pdb.set_trace()

            return capture
    else:
        logging.warning(f"Camera not found: {camera}")


# Thread to adjust camera controls dynamically
def adjust_camera_controls(cap, lock):
    while True:
        try:
            # Replace with your own input mechanism or trigger condition
            new_settings = load_camera_controls("tests/direction_camera.json")

            # Lock the capture to update controls safely
            with lock:
                controls_dict = dict([(c.display_name, c) for c in cap.controls])
                for control, value in new_settings:
                    if control in controls_dict:
                        controls_dict[control].value = value
                    else:
                        logging.warning(f"Control not found: {control}")

            time.sleep(1)  # Adjust frequency as needed
        except Exception as e:
            logging.error(f"Error adjusting controls: {e}")


controls = load_camera_controls("tests/direction_camera.json")
camera = CameraSpec(2115, 1133, 1920, 1080, 30, 2.0, controls)
cap = init_camera(camera)

if cap is None:
    logging.error("Camera not found")
    exit(1)

controls_dict = dict([(c.display_name, c) for c in cap.controls])
lock = threading.Lock()

# Start the camera control thread
control_thread = threading.Thread(target=adjust_camera_controls, args=(cap, lock))
control_thread.daemon = True
control_thread.start()

prev_time = 0

# Start the camera streaming loop
while True:
    frame = cap.get_frame_robust()
    data = frame.bgr if hasattr(frame, "bgr") else frame.gray

    # Calculate the FPS
    now = time.time()
    dt = now - prev_time
    prev_time = now
    fps = 1 / dt

    cv2.putText(
        data, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )
    cv2.imshow("Camera Feed", data)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
