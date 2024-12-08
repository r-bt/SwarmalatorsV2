import uvc, cv2, time
from typing import NamedTuple, Optional
import logging
import json, pdb, threading
from tracker import VideoStream
import numpy as np

stream = VideoStream("swarmalators/tracker/tracker_camera.json").start()

while True:
    frame = stream.read()

    while frame is None:
        frame = stream.read()

    display_frame = frame.copy()

    cv2.imshow("Frame", result)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
