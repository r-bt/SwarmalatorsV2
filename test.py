import cv2
import numpy as np
from swarmalators.tracker.tracker import SpheroTracker


def process_single_image(image_path, spheros, init_positions=[]):
    """
    Process a single image using SpheroTracker.

    Args:
        image_path (str): Path to the image file.
        spheros (int): Number of spheros to detect.
        init_positions (list): Initial positions of the spheros (optional).
    """
    # Load the image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Create a dummy event and other required parameters
    tracking_event = None  # Not needed for single image processing
    positions = []
    lock = None  # Not needed for single image processing
    velocities = []

    # Initialize the tracker
    tracker = SpheroTracker(
        device=0,  # Device is irrelevant for single image processing
        spheros=spheros,
        tracking=tracking_event,
        positions=positions,
        lock=lock,
        velocities=velocities,
        init_positions=init_positions,
    )

    # Process the image
    dets, (thresh, display_frame) = tracker._detect_objects(frame)

    # Display the results
    print(f"Detected spheros: {dets}")
    cv2.imshow("Thresholded Image", thresh)
    cv2.imshow("Detected Spheros", display_frame)

    # Wait for user to close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage
if __name__ == "__main__":
    image_path = "output.png"  # Replace with the path to your image
    spheros = 14  # Replace with the number of spheros you expect
    process_single_image(image_path, spheros)
