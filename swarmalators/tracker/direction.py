import cv2
from ._video_stream import VideoStream
import numpy as np
import os

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

    def find_sphero_direction(self):
        """
        Find the direction of a single sphero

        We find the direction from the x-axis counter clockwise

        Returns:
            Angle: The angle of the direction in degrees
            Center: The center of the sphero
        """
        frame = self.stream.read()

        if frame is None:
            print("Failed to read frame")
            return None

        # First find the black mat

        canvas_approx = self._find_black_mat(frame)

        # Find the blue light emitted from Sphero's front LED

        lower_blue = np.array([100, 150, 100])
        upper_blue = np.array([140, 255, 255])

        blue_led_contour = self._find_colored_led(
            frame, canvas_approx, lower_blue, upper_blue
        )

        if blue_led_contour is None:
            print("Failed to find blue LED")
            return None

        # Find the green light emitted from the Sphero's back LED

        lower_green = np.array([0, 100, 50])
        upper_green = np.array([100, 255, 255])

        green_led_contour = self._find_colored_led(
            frame, canvas_approx, lower_green, upper_green
        )

        if green_led_contour is None:
            print("Failed to find green LED")

            while True:
                cv2.imshow("Frame", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            return None

        # Get center of the blue and green LED contours

        M_blue = cv2.moments(blue_led_contour)
        cX_blue = M_blue["m10"] / M_blue["m00"]
        cY_blue = M_blue["m01"] / M_blue["m00"]

        M_green = cv2.moments(green_led_contour)
        cX_green = M_green["m10"] / M_green["m00"]
        cY_green = M_green["m01"] / M_green["m00"]

        # Calculate the direction vector from the green to the blue LED

        direction_vector = np.array([cX_blue - cX_green, cY_green - cY_blue])

        angle_radians = np.arctan2(direction_vector[1], direction_vector[0])

        angle_degrees = np.degrees(angle_radians)

        if angle_degrees < 0:
            angle_degrees += 360

        # Combine the blue and green LED contours to get the bounding box

        x_blue, y_blue, w_blue, h_blue = cv2.boundingRect(blue_led_contour)
        x_green, y_green, w_green, h_green = cv2.boundingRect(green_led_contour)

        # Combine the bounding boxes

        x = min(x_blue, x_green)
        y = min(y_blue, y_green)

        x_2 = max(x_blue + w_blue, x_green + w_green)
        y_2 = max(y_blue + h_blue, y_green + h_green)

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
