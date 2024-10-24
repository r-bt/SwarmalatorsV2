import numpy as np
import os
from swarmalator import Swarmalator
from experiments import run_experiments
import random

## NOTE: MODIFY TO THE PORTs ON YOUR COMPUTER FOR THE NRF5340
is_windows = os.name == "nt"
PORT1 = "/dev/tty.usbmodem0010500530493" if not is_windows else "COM7"
PORT2 = "/dev/tty.usbmodem0010500746993" if not is_windows else "COM9"

SPHEROS_OLD = [
    "SB-1B35",
    "SB-F860",
    "SB-2175",
    "SB-369C",
    "SB-618E",
    "SB-6B58",
    "SB-9938",
    "SB-BFD4",
    "SB-C1D2",
    "SB-CEFA",
    "SB-DF1D",
    "SB-F465",
    "SB-F479",
    "SB-F885",
    "SB-FCB2",
]

SPHEROS_NEW = [
    "SB-31B8",
    "SB-9CA8",
    "SB-80C4",
    "SB-F509",
    "SB-5883",
    "SB-8893",
    "SB-D64E",
    "SB-7D72",
    "SB-7D7C",
    "SB-4483",
    "SB-378F",
    "SB-2C58",
    "SB-D9E2",
    "SB-2E4B",
    "SB-6320",
]


def main():
    """
    Performs the swarmalator model on the Spheros
    """

    spheros = 15

    # Setup the swarmalator model
    natural_frequencies = np.ones(spheros)
    half = spheros // 2
    natural_frequencies[:half] = 1
    natural_frequencies[half:] = -1
    phase = np.linspace(0, 2 * np.pi, spheros, endpoint=False)

    random.shuffle(phase)

    swarmalator = Swarmalator(spheros, 1, 1, phase, natural_frequencies)

    # Run the experiments
    run_experiments([SPHEROS_NEW], [PORT1], swarmalator)


if __name__ == "__main__":
    main()
