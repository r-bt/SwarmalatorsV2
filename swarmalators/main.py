import numpy as np
import os
from swarmalator import Swarmalator
from experiments import run_experiments

## NOTE: MODIFY TO THE PORTs ON YOUR COMPUTER FOR THE NRF5340
is_windows = os.name == "nt"
PORT1 = "/dev/tty.usbmodem0010500530493" if not is_windows else "COM7"
PORT2 = "/dev/tty.usbmodem0010500746993" if not is_windows else "COM9"

# SPHEROS_OLD = [
#     "SB-1B35",
#     "SB-F860",
#     "SB-2175",
#     "SB-369C",
#     "SB-618E",
#     "SB-6B58",
#     "SB-9938",
#     "SB-BFD4",
#     "SB-C1D2",
#     "SB-CEFA",
#     "SB-DF1D",
#     "SB-F465",
#     "SB-F479",
#     "SB-F885",
#     "SB-FCB2",
# ]

SPHEROS_NEW = [
    "SB-31B8",
    "SB-9CA8",
    "SB-80C4",
    "SB-F509",
    "SB-5883",
    "SB-8893",
    "SB-7D7C",
    "SB-D64E",
    "SB-7D72",
    "SB-4483",
    "SB-378F",
    "SB-2C58",
    "SB-D9E2",
    "SB-2E4B",
    "SB-6320",
]

CAMERA_ID = 1


def main():
    """
    Performs the swarmalator model on the Spheros
    """

    spheros = len(SPHEROS_NEW)

    # Setup the swarmalator model
    natural_frequencies = np.zeros(spheros)

    phase = np.linspace(0, 2 * np.pi, spheros, endpoint=False)

    swarmalator = Swarmalator(spheros, 0, 0, phase, natural_frequencies, target=None)

    # Run the experiments
    run_experiments([SPHEROS_NEW], [PORT1], swarmalator, CAMERA_ID, targets=None)


if __name__ == "__main__":
    main()
