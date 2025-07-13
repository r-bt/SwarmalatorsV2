import uvc, cv2, time
from typing import NamedTuple, Optional
import logging
import json, pdb, threading
from tracker import VideoStream
import numpy as np

stream = VideoStream("swarmalators/tracker/tracker_camera.json").start()


# def filter_bricks(frame):
#     """
#     Given a frame with bricks, filter out the bricks
#     """
#     # Convert to HSV
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     # Define the range for red color (adjust as needed)
#     lower_red1 = np.array([0, 40, 50])
#     upper_red1 = np.array([10, 255, 255])
#     lower_red2 = np.array([170, 40, 50])
#     upper_red2 = np.array([180, 255, 255])

#     # Create masks for red
#     mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
#     mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
#     red_mask = mask1 | mask2

#     # Define kernel sizes
#     kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

#     for k in [kernel1, kernel2, kernel3]:
#         cleaned_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, k, iterations=2)

#     for k in [kernel2]:
#         cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, k, iterations=2)

#     # Remove the red color from the frame
#     result = frame.copy()

#     result[cleaned_mask == 255] = [0, 0, 0]

#     return result


while True:
    frame = stream.read()

    while frame is None:
        frame = stream.read()

    display_frame = frame.copy()

    # Filter out the bricks
    # filtered_frame = filter_bricks(frame)

    cv2.imshow("Frame", display_frame)
    # cv2.imshow("Filtered Frame", filtered_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
