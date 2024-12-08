import uvc, cv2, time
from typing import NamedTuple, Optional
import logging
import json, pdb, threading
from swarmalators.tracker import VideoStream

stream = VideoStream("tests/tracker_camera.json").start()

while True:
    frame = stream.read()
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
