import time
from nRFSwarmalator import nRFSwarmalator
from tracker import DirectionFinder, Tracker
import numpy as np
import colorsys
from simple_pid import PID
import csv

MAX_SPHEROS_PER_SWARMALATOR = 15


def init_spheros(spheros: int, nrf_swarmalators: list[nRFSwarmalator]) -> list:
    """
    Initialize the spheros by finding the direction of the spheros and then setting that to 0

    Parameters
    ----------
    spheros : int
        The number of spheros
    swarmalators : list[nRFSwarmalator]
        The list of swarmalators

    Returns
    -------
    list
        The centers of the spheros
    """
    # Turn on the direction finder
    finder = DirectionFinder(1)

    # Wait for the camera to be ready
    time.sleep(1)

    centers = []

    for nrf_swarmalator in nrf_swarmalators:
        nrf_swarmalator.set_mode(1)

    remaining_spheros = spheros
    while remaining_spheros > 0:
        nrf_swarmalator_index = (
            spheros - remaining_spheros
        ) // MAX_SPHEROS_PER_SWARMALATOR
        nrf_swarmalator = nrf_swarmalators[nrf_swarmalator_index]

        nrf_swarmalator.matching_orientation()

        time.sleep(1.25)  # Setting up the arrow animation takes time

        """
        Correct the orientation

        Notes:
        1. Sphero Logo (not text) is the front
        2. 90 degrees is the right side
        3. 180 degrees is the back
        4. 270 degrees is the left side
        """

        while True:
            res = finder.find_sphero_direction()

            if res is None:
                continue

            direction, center = res

            heading = direction

            if heading < 0:
                heading += 360

            nrf_swarmalator.matching_correct_heading(heading)

            break

        """
        Get the center to initalize the tracker (this is how we match Sphero IDs to tracker IDs)
        """

        centers.append(center)

        """
        Increment sphero count
        """

        remaining_spheros -= 1

        if remaining_spheros > 0:
            nrf_swarmalator.matching_next_sphero()

        print("Sphero calibrated! Remaining spheros: {}".format(remaining_spheros))

    for nrf_swarmalator in nrf_swarmalators:
        nrf_swarmalator.set_mode(0)

    finder.stop()

    # wait for the camera to be ready
    time.sleep(1)

    return centers


def angles_to_rgb(angles_rad):
    """
    Convert angles (in radians) to RGB colors
    """
    # Convert the angles to hue values (ranges from 0.0 to 1.0 in the HSV color space)
    hues = angles_rad / (2 * np.pi)

    # Set fixed values for saturation and value (you can adjust these as desired)
    saturation = np.ones_like(hues)
    value = np.ones_like(hues)

    hsv_colors = np.stack((hues, saturation, value), axis=-1)
    rgb_colors = np.apply_along_axis(lambda x: colorsys.hsv_to_rgb(*x), -1, hsv_colors)

    # Scale RGB values to 0-255 range
    rgb_colors *= 255
    rgb_colors = rgb_colors.astype(np.uint8)

    return rgb_colors


def connect_to_nordic(
    sphero_collection: list[list[str]], ports: list[str]
) -> list[nRFSwarmalator]:
    """
    Connects to the nRF5340 boards
    """
    nrf_swarmalators = []

    if len(sphero_collection) != len(ports):
        raise RuntimeError("Number of sphero collections must match number of ports")

    for collection, port in zip(sphero_collection, ports):
        nrf_swarmalator = nRFSwarmalator(collection, port)
        nrf_swarmalators.append(nrf_swarmalator)

    for nrf_swarmalator in nrf_swarmalators:
        nrf_swarmalator.wait_for_spheros()

    return nrf_swarmalators


def run_experiments(
    sphero_collection: list[list[str]],
    ports: list[str],
    swarmalator,
    Kp=80,
    Ki=1.1,
    Kd=0,
):
    """
    Runs the swarmalator experiments

    Parameters:
    -----------
    sphero_collection: list[list[str]]
        The collection of Sphero IDs
    ports: list[str]
        The list of ports to connect to
    swarmalator: Swarmalator
        The swarmalator model
    Kp: int
        The proportional gain for the PID controller
    Ki: int
        The integral gain for the PID controller
    Kd: int
        The derivative gain for the PID controller
    """

    # Connect to the nRF5340 boards
    nrf_swarmalators = connect_to_nordic(sphero_collection, ports)

    # Get the number of spheros
    spheros = sum([len(collection) for collection in sphero_collection])

    # Initalize all the Spheros
    print("Initializing Spheros")

    centers = init_spheros(spheros, nrf_swarmalators)

    # Switch Spheros to COLORS mode
    for nrf_swarmalator in nrf_swarmalators:
        nrf_swarmalator.set_mode(2)

    # Start tracking
    tracker = Tracker()

    tracker.start_tracking_objects(1, len(centers), centers)

    # Get the inital positions
    got = False
    while not got:
        try:
            positions = tracker.get_positions()
            if positions is not None:
                got = True
        except:
            continue

    # Provide it to the swarmalator model
    swarmalator.update(positions[:, :2])

    # Setup PID controllers to control Sphero velocities

    pid_controllers = [PID(Kp, Ki, Kd, setpoint=0) for _ in range(spheros)]
    for pid in pid_controllers:
        pid.output_limits = (0, 100)
        pid.sample_time = 0.1

    # Create output files
    current_time = time.strftime("%Y%m%d%H%M%S")
    output_file = f"outputs/state_{current_time}.csv"

    # Run the experiment

    now = time.monotonic()
    prev_positions = None

    with open(output_file, "w") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(
            [
                "Time",
                *["Phase {}".format(i) for i in range(spheros)],
                *["Position {}".format(i) for i in range(spheros)],
            ]
        )

        while True:
            # try:
            # Update and get values from swarmalator model
            positions = tracker.get_positions()

            if positions is None:
                continue

            swarmalator.update(positions[:, :2])
            swarmalator.update_phase(time.monotonic() - now)

            phase_state = swarmalator.get_phase_state()
            velocities = swarmalator.get_velocity()

            # Calculate the current velocity for all Spheros
            real_velocities = np.zeros(spheros)
            if prev_positions is not None:
                traveled = np.linalg.norm(
                    positions[:, :2] - prev_positions[:, :2], axis=1
                )
                real_velocities = traveled / (time.monotonic() - now)

            prev_positions = positions
            now = time.monotonic()

            # Update the PID controllers to get new velocities

            to_send_velocities = []
            for i, (controller, velocity) in enumerate(
                zip(pid_controllers, velocities)
            ):
                # Update the set point to the new desired velocity
                controller.setpoint = np.linalg.norm(velocity)

                # Update the PID controller
                command = controller(real_velocities[i])

                # Get the heading
                heading = int(np.degrees(np.arctan2(velocity[1], velocity[0])))
                # Sphero uses a different heading system (0 is the front, 90 is the right side, 180 is the back, 270 is the left side)
                # Effect is that left and right are switched
                heading = -heading

                if heading < 0:
                    heading += 360

                # Store the speed and heading
                to_send_velocities.append((int(command), heading))

            colors = angles_to_rgb(phase_state)

            for i, nrf_swarmalator in enumerate(nrf_swarmalators):
                color_selection = colors[
                    i
                    * MAX_SPHEROS_PER_SWARMALATOR : (i + 1)
                    * MAX_SPHEROS_PER_SWARMALATOR
                ]

                velocities_selection = to_send_velocities[
                    i
                    * MAX_SPHEROS_PER_SWARMALATOR : (i + 1)
                    * MAX_SPHEROS_PER_SWARMALATOR
                ]

                nrf_swarmalator.colors_set_colors(color_selection, velocities_selection)

            tracker.set_velocities([(v[0], -v[1]) for v in to_send_velocities])

            print("Took: ", time.monotonic() - now)

            writer.writerow([time.monotonic(), *phase_state, *positions[:, :2]])
        # except Exception as e:
        #     print(e)
        # continue
