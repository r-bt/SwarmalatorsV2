import cv2
from ._video_stream import VideoStream
import numpy as np
import os
from typing import Callable
import time

CONTOUR_DISTANCE_THRESHOLD = 350


class DirectionFinder:
    """
    DirectionFinder. Identify the direction of a single sphero
    """

    def __init__(self, device: int):
        # Get path to the settings file

        settings_path = self._get_settings_path()

        # Get camera on OpenCV
        self.stream = VideoStream(device, settings_path).start()

    def __del__(self):
        self.stream.stop()

    def stop(self):
        self.stream.stop()

    def find_sphero_direction(self, next_led: Callable):
        """
        Find the direction of a single sphero

        We find the direction from the x-axis counter clockwise

        Returns:
            Angle: The angle of the direction in degrees
            Center: The center of the sphero
        """
        frame = self.stream.read()

        if frame is None:
            print("Error reading from camera")
            return None

        # First find the black mat

        canvas_approx = self._find_black_mat(frame)

        # Find the blue light emitted from Sphero's front LED

        lower_blue = np.array([100, 150, 100])
        upper_blue = np.array([140, 255, 255])

        front_led_contour = self._find_colored_led(
            frame, canvas_approx, lower_blue, upper_blue
        )

        if front_led_contour is None:
            print("Can't find the front LED")

            # Show the frame to the user
            while True:
                cv2.imshow("Frame", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            return None

        # Switch to the back blue LED

        next_led()

        time.sleep(0.5)

        # Find the blue light emitted from the Sphero's back LED

        frame = self.stream.read()

        if frame is None:
            print("Error reading from camera")
            return None

        back_led_contour = self._find_colored_led(
            frame, canvas_approx, lower_blue, upper_blue
        )

        if back_led_contour is None:
            print("Failed to find the back LED")

            while True:
                cv2.imshow("Frame", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            return None

        # Get center of the blue and green LED contours

        M_front = cv2.moments(front_led_contour)
        cX_front = M_front["m10"] / M_front["m00"]
        cY_front = M_front["m01"] / M_front["m00"]

        M_back = cv2.moments(back_led_contour)
        cX_back = M_back["m10"] / M_back["m00"]
        cY_back = M_back["m01"] / M_back["m00"]

        # Calculate the direction vector from the green to the blue LED

        direction_vector = np.array([cX_front - cX_back, cY_back - cY_front])

        angle_radians = np.arctan2(direction_vector[1], direction_vector[0])

        angle_degrees = np.degrees(angle_radians)

        if angle_degrees < 0:
            angle_degrees += 360

        # Combine the blue and green LED contours to get the bounding box

        x_front, y_front, w_front, h_front = cv2.boundingRect(front_led_contour)
        x_back, y_back, w_back, h_back = cv2.boundingRect(back_led_contour)

        # Combine the bounding boxes

        x = min(x_front, x_back)
        y = min(y_front, y_back)

        x_2 = max(x_front + w_front, x_back + w_back)
        y_2 = max(y_front + h_front, y_back + h_back)

        # Get the center of the bounding box
        center = (x + x_2) // 2, (y + y_2) // 2

        return int(angle_degrees), center

    """
    Private methods
    """

    def _find_black_mat(self, frame):
        """
        Assume experiments take place on a black mat

        Args:
            frame: The frame to find the black mat in

        Returns:
            The approximated contour of the black mat
        """

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        canvas_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(canvas_contour, True)
        canvas_approx = cv2.approxPolyDP(canvas_contour, epsilon, True)

        return canvas_approx

    def _find_colored_led(self, frame, canvas_approx, lower, upper):
        """
        Sphero lights front or back LED a specific color

        Args:
            frame: The frame to find the LED in
            canvas_approx: The approximated contour of the black mat
            lower: The lower bound of the color
            upper: The upper bound of the color
        """

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, lower, upper)

        # Erode and dilate
        mask = cv2.dilate(mask, None, iterations=4)
        mask = cv2.erode(mask, None, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Sort the contours by area

        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Find the largest contour within the black mat

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if (
                cv2.pointPolygonTest(canvas_approx, (x + w // 2, y + h // 2), False)
                == 1
            ):
                return contour

        return None

    def _get_settings_path(self):
        """
        Gets the path to direction_camera.json

        Returns:
            The path to direction_camera.json
        """

        current_file_path = os.path.abspath(__file__)
        current_directory = os.path.dirname(current_file_path)

        return os.path.join(current_directory, "direction_camera.json")
