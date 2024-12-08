"""
Finds a blue laser pointer from a video stream and tracks it

Assumes in a dark room

Author: Richard Beattie
"""

import cv2
from tracker import VideoStream
import os
import numpy as np
from scipy.signal import savgol_filter
import time
import atexit


class LaserTracker:
    """
    Tracks a blue laser pointer to create a path
    """

    def __init__(self):
        """
        Initializes the laser tracker

        Parameters
        ----------
        camera_id : int
            The camera id to use
        """
        self._video_stream = VideoStream(self._get_settings_path()).start()
        self._path = []

        self._set_scale_factor()

        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self._add_point)

        self._prev_point = (0, 0)

        self._init_recording()

    def close(self):
        self._video_stream.stop()
        self.out.release()

    def track(self):
        """
        Tracks the laser pointer
        """
        # Start the video stream

        while True:
            frame = self._video_stream.read()

            if frame is None:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            display_frame = frame.copy()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

            thresh = cv2.erode(thresh, None, iterations=2)

            thresh = cv2.dilate(thresh, None, iterations=1)

            # Find the largest contour

            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if len(contours) == 0:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            largest_contour = max(contours, key=cv2.contourArea)

            # Get the centroid

            M = cv2.moments(largest_contour)

            if M["m00"] == 0:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            cv2.circle(display_frame, (cx, cy), 5, (0, 0, 255), -1)

            # Add to the path

            self._path.append((cx, cy))

            # Smooth the path
            smoothed_path = self._smooth_path(self._path)

            # Draw the smoothed path
            if len(smoothed_path) > 1:
                for i in range(1, len(smoothed_path)):
                    cv2.line(
                        display_frame,
                        (int(smoothed_path[i - 1][0]), int(smoothed_path[i - 1][1])),
                        (int(smoothed_path[i][0]), int(smoothed_path[i][1])),
                        (0, 0, 255),
                        2,
                    )

            cv2.imshow("Frame", display_frame)

            self.out.write(display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    def get_path(self):
        """
        Gets the path of the laser pointer
        """
        normalized_path = [self._normalize_coordinates(x, y) for x, y in self._path]

        return [self._smooth_path(normalized_path), normalized_path]

    """
    PRIVATE
    """

    def _get_settings_path(self):
        """
        Gets the path to tracker_camera.json

        Returns:
            The path to direction_camera.json
        """

        current_file_path = os.path.abspath(__file__)
        current_directory = os.path.dirname(current_file_path)

        return os.path.join(current_directory, "tracker", "tracker_camera.json")

    def _smooth_path(self, path, window_length=5, polyorder=2):
        """
        Given a path, smooth it using a Savitzky-Golay filter
        """
        if len(path) < window_length:  # Ensure enough points to apply the filter
            return path
        x_coords, y_coords = zip(*path)
        smoothed_x = savgol_filter(x_coords, window_length, polyorder)
        smoothed_y = savgol_filter(y_coords, window_length, polyorder)
        return list(zip(smoothed_x, smoothed_y))

    def _set_scale_factor(self):
        """
        Sets the scale factor so we can normalize coordinates
        """
        frame = self._video_stream.read()
        while frame is None:
            frame = self._video_stream.read()

        self._height, self._width = frame.shape[:2]

        self._scale_factor = 8.0 / max(self._width, self._height)

    def _normalize_coordinates(self, x, y):
        normalized_x = self._scale_factor * (x - self._width / 2)
        normalized_y = self._scale_factor * (y - self._height / 2)
        return normalized_x, -normalized_y

    def _add_point(self, event, x, y, flags, param):
        # if event == cv2.EVENT_LBUTTONDBLCLK:
        #     print("Mouse clicked!")
        #     self._path.append((x, y))
        if self._prev_point[0] != x or self._prev_point[1] != y:
            self._prev_point = (x, y)
            print(self._normalize_coordinates(x, y))

    def _init_recording(self):
        """
        Initalize recording
        """
        # Set up video writer using FFmpeg
        current_time = time.strftime("%Y%m%d%H%M%S")
        output_file = f"outputs/laser_recording_{current_time}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Change the codec as needed
        out = cv2.VideoWriter(
            output_file, fourcc, 20.0, (self._width, self._height)
        )  # Adjust parameters as needed
        self.out = out

        # Setup cleanup function
        def cleanup():
            out.release()

        atexit.register(cleanup)


if __name__ == "__main__":
    laser_tracker = LaserTracker()
    laser_tracker.track()
    print(laser_tracker.get_path())
    laser_tracker.close()
