import numpy as np
import time
import math

START_TIME = time.time()


class Swarmalator:

    def __init__(
        self,
        agents: int,
        K: int,
        J: int,
        phase: np.ndarray,
        natural_frequencies: np.ndarray,
        chiral: bool = False,
        target: np.ndarray = None,
    ):
        """
        Intialize swarmalator model

        Parameters
        ----------
        agents : int
                Number of agents
        K : int
            Phase coupling coefficient
        J: int
            Spatial-phase interaction coeffi- cient
        positions: list
            Initial positions of all the agents
        """
        np.random.seed(0)  # Debug have the same random numbers

        self._agents = agents
        self._K = K
        self._J = J
        self._A = 1
        self._B = 1

        # Init positon state
        self.inherent_velocity = np.random.rand(self._agents, 2)
        self.inherent_velocity[:, 0] = 0  # Inherent velocity in x-dir
        self.inherent_velocity[:, 1] = 0  # Inherent velocity in y-dir

        self.velocity = np.zeros((self._agents, 2))

        half_len = self._agents // 2

        self.c = np.random.rand(self._agents, 1)

        self._chiral = False  # Whether to do chiral behaviours

        self.c[:half_len, 0] = 0
        self.c[half_len:, 0] = 0

        # Init phase state (0 is natural freuqnecy, 1 is phase)
        self.phase_state = np.random.rand(self._agents, 2)

        # half_len = len(self.phase_state) // 2
        self.phase_state[:half_len, 0] = 0
        self.phase_state[half_len:, 0] = 0

        # self.phase_state[0:4, 0] = 0
        # self.phase_state[5:9,0] = 0
        # self.phase_state[10:14, 0] = 0
        # self.phase_state[:, 0] = 1/
        # self.phase_state[:, 1] *= 2*np.pi
        # self.phase_state[:, 1] = np.linspace(0, 2*np.pi, self._agents, endpoint=False)

        # Generate evenly spaced values
        evenly_spaced_values = np.linspace(0, 2 * np.pi, self._agents, endpoint=False)

        # Shuffle the values
        # np.random.shuffle(evenly_spaced_values)

        # Assign the shuffled values to self.phase_state[:, 1]
        self.phase_state[:, 1] = evenly_spaced_values
        # self.phase_state[:, 1] = np.ones(self._agents)

        # Keep track of time between updates
        self._updated = time.time()

        self._target = np.array([target]) if target is not None else None

        self._delta_phase = np.zeros(self._agents)

    def update(self, positions):
        """
        Perform one tick update of swarmalator model

        Note: We perform using matrix multiplication since numpy supports vectorization and is faster than for loops
        """
        """
        if time.time()-START_TIME < 80:
            #print("elapsed:", time.time()-START_TIME)
            self._K = -1
            self._J = 1
        elif time.time()-START_TIME >= 80 and time.time()-START_TIME < 100:
            #print("elapsed:",time.time()-START_TIME)
            self._K = 1
            self._J = 1
        elif time.time()-START_TIME >= 100:
            #print("elapsed:",time.time()-START_TIME)
            self.phase_state[:, 1] = np.linspace(0, 3*np.pi/2, self._agents, endpoint=False)
            self._K = 0
            self._J = 1
        """
        # self._K = 1
        # self._J = 1

        # target = np.array([[3,0]])

        # M
        # if time.time()-START_TIME < 50:
        #   target = np.array([[0,-0.5]])
        # elif time.time()-START_TIME >= 50 and time.time()-START_TIME < 100:
        #   target = np.array([[0,1.25]])
        # elif time.time()-START_TIME >= 100 and time.time()-START_TIME < 110:
        #   target = np.array([[1,0.5]])
        # elif time.time()-START_TIME >= 110 and time.time()-START_TIME < 120:
        #    target = np.array([[2,1.25]])
        # elif time.time()-START_TIME >= 120:
        #    target = np.array([[2,-0.5]])

        # T
        # if time.time() - START_TIME < 50:
        #     target = np.array([[0, -0.5]])
        # elif time.time() - START_TIME >= 50 and time.time() - START_TIME < 100:
        #     target = np.array([[0, 1.25]])
        # elif time.time() - START_TIME >= 100 and time.time() - START_TIME < 110:
        #     target = np.array([[-0.5, 1.25]])
        # elif time.time() - START_TIME >= 110:
        #     target = np.array([[0.5, 1.25]])

        if self._target is not None:
            distToTargetVector = self._target[0, :2] - positions[:, :2][:, np.newaxis]
            distToTarget = np.linalg.norm(distToTargetVector, axis=2)
            minDistToTarget = np.min(distToTarget)
            maxDistToTarget = np.max(distToTarget)
            J1 = np.zeros((self._agents, 1))
            J2 = np.zeros((self._agents, 1))
            J1 = (
                1
                * (np.absolute(distToTarget - minDistToTarget))
                / (maxDistToTarget - minDistToTarget)
            )
        else:
            J1 = np.ones((self._agents, 1)) * self._J

        # J2 = 1*(abs(distToTarget-maxDistToTarget))/(maxDistToTarget-minDistToTarget)
        # J2 = np.zeros((15, 1))

        # Calculate x_j - x_i and |x_j = x_i|
        vectors = positions[:, :2] - positions[:, :2][:, np.newaxis]
        distances = np.linalg.norm(vectors, axis=2)

        np.fill_diagonal(distances, 1e-6)  # Avoid division by zero

        # Calculate the phase difference
        # Note: Multiply by -1 since we are doing x_i = x_j but we want x_j - x_i
        phase_difference = -1 * np.subtract.outer(
            self.phase_state[:, 1], self.phase_state[:, 1]
        )

        # Calculate Q terms
        natural_frequencies = self.phase_state[:, 0]
        phase_normalized = natural_frequencies / np.absolute(natural_frequencies)
        phase_normalized = np.nan_to_num(phase_normalized)

        Q_x = (np.pi / 2) * np.absolute(
            np.subtract.outer(phase_normalized, phase_normalized)
        )
        Q_theta = (np.pi / 4) * np.absolute(
            np.subtract.outer(phase_normalized, phase_normalized)
        )

        if not self._chiral:
            Q_x = 0
            Q_theta = 0

        # Calculate cos and sin terms

        phase_cos_difference = np.cos(phase_difference - Q_x)
        phase_sin_difference = np.sin(phase_difference - Q_theta)

        dX = np.zeros((self._agents))
        dY = np.zeros((self._agents))

        for ii in range(self._agents):
            for jj in range(self._agents):
                if ii != jj:
                    dist = math.sqrt(
                        ((positions[jj, 0] - positions[ii, 0]) ** 2)
                        + ((positions[jj, 1] - positions[ii, 1]) ** 2)
                    )
                    dX[ii] += (
                        self._A
                        + J1[ii]
                        * math.cos(self.phase_state[jj, 1] - self.phase_state[ii, 1])
                    ) * (positions[jj, 0] - positions[ii, 0]) / dist - self._B * (
                        positions[jj, 0] - positions[ii, 0]
                    ) / (
                        dist**2
                    )
                    dY[ii] += (
                        self._A
                        + J1[ii]
                        * math.cos(self.phase_state[jj, 1] - self.phase_state[ii, 1])
                    ) * (positions[jj, 1] - positions[ii, 1]) / dist - self._B * (
                        positions[jj, 1] - positions[ii, 1]
                    ) / (
                        dist**2
                    )

        # Calculate velocity contributions
        # velocity_contributions = (self._A + J1*phase_cos_difference[:, :, np.newaxis]) * vectors / distances[:, :, np.newaxis] - vectors / np.square(distances[:, :, np.newaxis]) * (self._B)

        # Calculate chiral contribution
        chiral_contribtuion = self.c * np.stack(
            [
                np.cos(self.phase_state[:, 1] + np.pi / 2),
                np.sin(self.phase_state[:, 1] + np.pi / 2),
            ],
            axis=1,
        )

        if not self._chiral:
            chiral_contribtuion = 0

        # Calculate velocity and delta_phase
        # velocity = chiral_contribtuion + 1/self._agents * np.sum(velocity_contributions, axis=1)
        velocity = np.zeros((self._agents, 2))
        velocity[:, 0] = 1 / self._agents * dX
        velocity[:, 1] = 1 / self._agents * dY

        magnitudeVelocity = np.linalg.norm(velocity, axis=1)
        factorVal = 0.5
        for ii in range(self._agents):
            if magnitudeVelocity[ii] >= factorVal:
                xComponent = (
                    np.cos(math.atan2(velocity[ii, 1], velocity[ii, 0])) * factorVal
                )
                yComponent = (
                    np.sin(math.atan2(velocity[ii, 1], velocity[ii, 0])) * factorVal
                )
                velocity[ii, 0] = xComponent
                velocity[ii, 1] = yComponent

        delta_phase = self.phase_state[:, 0] + (self._K / self._agents) * np.sum(
            phase_sin_difference / distances, axis=1
        )

        # print(phase_sin_difference)
        # print(distToTarget)

        # Update phase and velocity
        dt = 0.01
        self.phase_state[:, 1] += delta_phase * (time.time() - self._updated)
        self.velocity = velocity

        self._updated = time.time()
        self.phase_state[:, 1] %= 2 * np.pi

    def get_phase_state(self):
        return self.phase_state[:, 1]

    def get_velocity(self):
        return self.velocity

    def set_target(self, target):
        """
        Set the target position for the agents
        """
        n_target = np.array([target])
        assert n_target.shape == (
            1,
            2,
        ), "Target must be of shape (1, 2), is {}".format(n_target.shape)
        self._target = n_target

    def update_phase(self, deltaT):
        # Apply the updated phase
        pass
