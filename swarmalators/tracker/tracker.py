import cv2
from ._video_stream import VideoStream
import multiprocessing as mp

# from ._sort import Sort
from .euclid_tracker import EuclideanDistTracker
import numpy as np
import time
import atexit
import os

MAX_LEN = 1
MAX_CONTOUR_AREA = 700
MIN_CONTOUR_AREA = 5


class SpheroTracker:
    """
    Tracker Class. Handles actual tracking of Spheros

    Attributes:
        init_positions: The initial positions of the Spheros
    """

    def __init__(
        self,
        device: int,
        spheros: int,
        tracking: mp.Event,
        positions,
        lock,
        velocities,
        init_positions: list = [],
        marks: list = [],
    ):
        self._spheros = spheros
        self._tracking = tracking
        self._positions = positions
        self._lock = lock
        self._velocities = velocities

        settings_path = self._get_settings_path()

        self._stream = VideoStream(settings_path).start()

        # Get scale factors
        self._set_scale_factor()

        # It takes some time for the camera to focus, etc
        print("Waiting for camera to calibrate")
        self._calibrate_camera()
        print("Calibrated camera")

        # Setup and initalize tracker
        self._euclid_tracker = EuclideanDistTracker()
        self._euclid_tracker.init(init_positions)

        self._init_recording()

        self._marks = marks

    """
    Private methods
    """

    def _init_recording(self):
        """
        Initalize recording
        """
        # Set up video writer using FFmpeg
        current_time = time.strftime("%Y%m%d%H%M%S")
        output_file = f"outputs/output_{current_time}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Change the codec as needed
        out = cv2.VideoWriter(
            output_file, fourcc, 20.0, (self._width, self._height)
        )  # Adjust parameters as needed
        self.out = out

        # Setup cleanup function
        def cleanup():
            out.release()

        atexit.register(cleanup)

    def _set_scale_factor(self):
        """
        Sets the scale factor so we can normalize coordinates
        """
        frame = self._stream.read()
        while frame is None:
            frame = self._stream.read()

        self._height, self._width = frame.shape[:2]

        self._scale_factor = 8.0 / max(self._width, self._height)

    def _calibrate_camera(self):
        count = 0

        while count < 100:
            frame = self._stream.read()

            if frame is None:
                continue

            count += 1

        count = 0

        while count < 100:
            frame = self._stream.read()

            if frame is None:
                continue

            dets, _ = self._detect_objects(frame)

            if len(dets) == self._spheros:
                count += 1
            else:
                print("Only found {} spheros".format(len(dets)))

    def _track_objects(self):
        """
        Track objects
        """

        frame_time = 0
        prev_frame_time = 0

        while self._tracking.is_set():
            frame = self._stream.read()
            self.out.write(frame)

            dets, (thresh, display_frame) = self._detect_objects(frame)

            active_tracks = []
            try:
                active_tracks = self._euclid_tracker.update(dets)
            except RuntimeError as e:
                print("Error updating tracker")
                self._stream.stop()  # Stop the video stream
                self.out.release()
                print(e)

                new_display_frame = frame.copy()

                for point in self._euclid_tracker.center_points.values():
                    cv2.circle(
                        new_display_frame,
                        (int(point[0]), int(point[1])),
                        5,
                        (0, 0, 255),
                        -1,
                    )

                while True:
                    cv2.imshow("New positions frame", display_frame)
                    cv2.imshow("Old positions frame", new_display_frame)
                    cv2.imshow("Thresh", thresh)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        raise RuntimeError("User quit")

            pos = self._update_positions(active_tracks)

            with self._lock:
                if len(self._positions) >= MAX_LEN:
                    self._positions.pop(0)
                self._positions.append(pos)

            for index, (sphero_id, track) in enumerate(active_tracks):
                if index < len(self._velocities):
                    velocity = self._velocities[index]

                    arrow_length = 40  # Adjust the arrow length as needed

                    heading_radians = np.radians(velocity[1])

                    # Calculate the endpoint of the arrow based on velocity and arrow_length
                    arrow_end = (
                        int(track[0] + arrow_length * np.cos(heading_radians)),
                        int(track[1] + arrow_length * np.sin(-heading_radians)),
                    )

                    # Draw the arrow on the frame
                    cv2.arrowedLine(
                        display_frame,
                        (int(track[0]), int(track[1])),
                        arrow_end,
                        (36, 255, 12),
                        5,
                    )

                    # Put the id of the Sphero on the frame
                    cv2.putText(
                        display_frame,
                        str(sphero_id),
                        (int(track[0]), int(track[1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )

            # We can't handle losing Spheros yet (fix in the future)

            if len(active_tracks) < self._spheros:
                print("Not enough spheros detected")
                print([len(active_tracks), len(dets)])
                while True:
                    cv2.imshow("Frame", display_frame)
                    cv2.imshow("Thresh", thresh)

                    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit the loop
                        break

            # Calculate the fps
            frame_time = time.time()

            fps = 1 / (frame_time - prev_frame_time)
            prev_frame_time = frame_time

            fps = str(int(fps))

            cv2.putText(
                display_frame,
                fps,
                (7, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (100, 255, 0),
                3,
                cv2.LINE_AA,
            )

            # Draw a line between mark history
            if len(self._marks) > 1:
                for i in range(1, len(self._marks)):
                    cv2.line(
                        display_frame,
                        (int(self._marks[i - 1][0]), int(self._marks[i - 1][1])),
                        (int(self._marks[i][0]), int(self._marks[i][1])),
                        (0, 0, 255),
                        2,
                    )

            # Display the image

            cv2.imshow("Frame", display_frame)
            cv2.imshow("Thresh", thresh)

            if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit the loop
                break

        print("Saving video...")
        cv2.destroyAllWindows()  # Close all OpenCV windows
        self._stream.stop()  # Stop the video stream
        self.out.release()

    def _detect_objects(self, frame):
        """
        Detect Spheros in a frame

        Args:
            frame: The frame to detect Spheros in

        Returns:
            A list of detections in form [x, y]
        """

        display_frame = frame.copy()

        # Find the black canvas mat
        canvas_approx = self._find_canvas(frame)
        cv2.polylines(display_frame, [canvas_approx], True, (0, 255, 0), 2)

        # Crop to just the canvas
        canvas_rect = cv2.boundingRect(canvas_approx)
        x, y, w, h = canvas_rect
        warped_canvas = frame[y : y + h, x : x + w]  # Crop the canvas area

        # Filter out the bricks
        # warped_canvas = self._filter_bricks(warped_canvas)

        # Apply Otsu's binarization
        gray = cv2.cvtColor(warped_canvas, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

        # Scale canvas approx into the cropped frame
        canvas_approx_scaled = canvas_approx - np.array([x, y])

        # Apply a mask to the thresholded image
        mask = np.zeros_like(thresh)
        cv2.fillPoly(mask, [canvas_approx_scaled], (255, 255, 255))
        thresh = cv2.bitwise_and(thresh, mask)

        # Noise removal
        thresh = cv2.erode(thresh, None, iterations=1)

        # # Dilate to increase the size of the bricks
        # thresh = cv2.dilate(thresh, None, iterations=2)

        # Downsample the image first
        thresh = cv2.resize(
            thresh, (0, 0), fx=0.25, fy=0.25
        )  # Downsample the image by 4x

        # Cluster white regions into 15 clusters
        try:
            centers, _ = self._cluster_spheros(
                thresh,
                num_clusters=self._spheros,
            )
        except Exception as e:
            print("Error when clustering")
            while True:
                cv2.imshow("Thresh", thresh)
                cv2.imshow("Display frame", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            raise e

        # Scale the centers back up
        dets = [(int(center[1] * 4) + x, int(center[0] * 4) + y) for center in centers]

        # Draw the centers
        for det in dets:
            cv2.circle(
                display_frame,
                det,
                5,
                (0, 255, 0),
                -1,
            )

        return dets, (thresh, display_frame)

    def _update_positions(self, tracks):
        """
        Update the positions of the spheros

        Args:
            dets: The detections to update the positions with in form [id, (x, y)]
        """

        pos = np.empty((len(tracks), 3))

        for i, track in enumerate(tracks):
            center_x, center_y = self._normalize_coordinates(track[1][0], track[1][1])

            pos[i] = np.array([center_x, center_y, track[0]])

        return pos

    def _normalize_coordinates(self, x, y):
        normalized_x = self._scale_factor * (x - self._width / 2)
        normalized_y = self._scale_factor * (y - self._height / 2)
        return normalized_x, -normalized_y

    def _unnormalize_coordinates(self, x, y):
        unnormalized_x = (x / self._scale_factor) + self._width / 2
        unnormalized_y = (-y / self._scale_factor) + self._height / 2
        return unnormalized_x, unnormalized_y

    def _get_settings_path(self):
        """
        Gets the path to tracker_camera.json

        Returns:
            The path to direction_camera.json
        """

        current_file_path = os.path.abspath(__file__)
        current_directory = os.path.dirname(current_file_path)

        return os.path.join(current_directory, "tracker_camera.json")

    def _find_canvas(self, frame):
        """
        Given a frame, find the canvas

        Args:
            frame: The frame to find the canvas in

        Returns:
            The canvas
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        _, thresh_2 = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

        cv2.imshow("Canvas thresh", thresh_2)

        contours, _ = cv2.findContours(
            thresh_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        canvas_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.005 * cv2.arcLength(canvas_contour, True)
        canvas_approx = cv2.approxPolyDP(canvas_contour, epsilon, True)
        return canvas_approx

    def _cluster_spheros(self, thresh, num_clusters=15):
        """
        Uses k-means clustering to cluster the Spheros

        Args:
            thresh: The thresholded image
            num_clusters: The number of clusters to use

        Returns:
            Centers: The centers of the clusters
            Labels: The labels of the clusters
        """

        # Find contours of white regions
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Create a mask to filter out large contours (obstacles)
        mask = np.zeros_like(thresh)
        for c in contours:
            _, _, w, h = cv2.boundingRect(c)
            bounding_area = w * h  # Calculate the area of the bounding box
            if bounding_area > MIN_CONTOUR_AREA:
                cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)

        cv2.imshow("Mask", mask)

        # Get white pixel coordinates from the filtered mask
        white_pixels = np.column_stack(np.where(mask > 0))

        # Perform k-means clustering

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 0.05)

        _, labels, centers = cv2.kmeans(
            white_pixels.astype(np.float32),
            num_clusters,
            None,
            criteria,
            10,
            cv2.KMEANS_PP_CENTERS,
        )

        return centers, labels

    def _filter_bricks(self, frame):
        """
        Given a frame with bricks, filter out the bricks
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the range for red color (adjust as needed)
        lower_red1 = np.array([0, 40, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 40, 50])
        upper_red2 = np.array([180, 255, 255])

        # Create masks for red
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 | mask2

        # Define kernel sizes
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        for k in [kernel1, kernel2, kernel3]:
            cleaned_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, k, iterations=2)

        for k in [kernel2]:
            cleaned_mask = cv2.morphologyEx(
                cleaned_mask, cv2.MORPH_OPEN, k, iterations=2
            )

        # Remove the red color from the frame
        result = frame.copy()

        result[cleaned_mask == 255] = [0, 0, 0]

        return result

    """
    Static methods
    """

    @staticmethod
    def start_tracking(
        device: int,
        spheros: int,
        tracking: mp.Event,
        positions,
        lock,
        velocities,
        init_positions: list = [],
        marks: list = [],
    ):
        """
        Start tracking spheros
        """
        print("Starting to track!")
        sphero_tracker = SpheroTracker(
            device,
            spheros,
            tracking,
            positions,
            lock,
            velocities,
            init_positions,
            marks,
        )

        sphero_tracker._track_objects()


class Tracker:
    """
    Manager Class. Handles creating tracker process and relying positions back
    """

    def __init__(self) -> None:
        self._tracking = mp.Event()
        self._tracking_process = None

        # Share positions between processes
        self._manager = mp.Manager()
        self._positions = self._manager.list()
        self._pos_lock = self._manager.Lock()
        self._velocities = self._manager.list()
        self._marks = self._manager.list()

    def start_tracking_objects(
        self, device: int, spheros: int, init_positions: list = []
    ):
        """
        Start tracking process
        """
        self._tracking.set()

        self._tracking_process = mp.Process(
            target=SpheroTracker.start_tracking,
            args=(
                device,
                spheros,
                self._tracking,
                self._positions,
                self._pos_lock,
                self._velocities,
                init_positions,
                self._marks,
            ),
        )
        self._tracking_process.daemon = True
        self._tracking_process.start()

    def mark(self):
        """
        Marks current position
        """

        center = np.mean(self._positions[-1][:, :2], axis=0)

        self._marks.append(center)

    def get_positions(self):
        with self._pos_lock:
            try:
                pos = self._positions.pop()
                return pos
            except:
                return None

    def set_velocities(self, velocities):
        with self._pos_lock:
            for i, velocity in enumerate(velocities):
                if i >= len(self._velocities):
                    self._velocities.append(velocity)
                else:
                    self._velocities[i] = velocity

    def cleanup(self):
        self._tracking.clear()
        if self._tracking_process:
            self._tracking_process.join()
        print("Finished tracking process")
