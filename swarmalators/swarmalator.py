import numpy as np


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
        phase: np.ndarray
            Initial phase of the agents
        natural_frequencies: np.ndarray
            Natural frequencies of the agents
        chiral: bool
            Whether to include chiral contributions
        target: np.ndarray
            Target position for the agents
        """
        np.random.seed(0)  # Debug have the same random numbers

        self._agents = agents

        # Setup base simulation parameters
        self._K = K
        self._J = J
        self._A = 1  # Ensures that agents do not dissipate infinitely
        self._B = 1  # Ensures that agents do not aggregate at a single point in space.

        # Agents have an inherent velocity
        self._inherent_velocities = np.zeros((self._agents, 2))

        # Agents have a natural frequency
        self._natural_frequencies = natural_frequencies

        # Agents might have a chiral contribution
        self._chiral = chiral
        self._c = np.ones((self._agents, 1))

        # Agents might have a target
        if target is not None:
            self._target = np.array([target])
            assert self._target.shape == (1, 2), "Target must be of shape (1, 2)"
        else:
            self._target = None

        # All agents start stationary
        self._velocity = np.zeros((self._agents, 2))

        # All agents start with a phase
        self._phase = phase
        self._delta_phase = np.zeros(self._agents)

    def update(self, positions):
        """
        Perform one tick update of swarmalator model

        Unlike the Matlab implementation we use numpy matrix operations for faster computation. This allows
        for higher performance and scalability especially when dealing with large number of agents.

        However, it makes the code harder to understand and debug.
        """

        Js = self._J if self._target is None else self._get_J_values(positions)

        # Calculate x_j - x_i, mulitply by -1 since we are doing x_i = x_j but we want x_j - x_i
        vectors = positions - positions[:, np.newaxis]

        # Calculate |x_j = x_i|
        distances = np.linalg.norm(vectors, axis=2)
        np.fill_diagonal(distances, 1)  # Avoid division by zero

        # Calculate the phase difference, multiply by -1 since we are doing x_i = x_j but we want x_j - x_i
        phase_difference = -1 * np.subtract.outer(self._phase, self._phase)

        # If chiral is enabled calculate Q_x and Q_theta
        Q_x, Q_theta = self._get_phase_offset_terms() if self._chiral else (0, 0)

        # Calculate cos and sin terms
        phase_cos_difference = np.cos(phase_difference - Q_x)
        phase_sin_difference = np.sin(phase_difference - Q_theta)

        # Calculate the updated velocity
        self._velocity = self._calculate_velocity(
            vectors, distances, phase_cos_difference, Js
        )

        # Calculate the updated phase
        self._delta_phase = self._calculate_delta_phase(phase_sin_difference, distances)

    def update_phase(self, deltaT):
        # Apply the updated phase
        self._phase += self._delta_phase * deltaT
        self._phase %= 2 * np.pi

    def _calculate_delta_phase(self, phase_sin_difference, distances):
        """
        Calculates the delta phase as:

        θ_i = ω_i + (K/N) * Σ_j[ sin(θ_j - θ_i - Q_theta) / |(x_j - x_i)| ]
        """

        delta_phase = self._natural_frequencies + (self._K / self._agents) * np.sum(
            phase_sin_difference / distances, axis=1
        )

        return delta_phase

    def _calculate_velocity(self, vectors, distances, phase_cos_difference, Js):
        """
        Calculates the velocity as:

        ẋ_i = v_i + 1/N * Σ_j[ ((x_j - x_i) / |(x_j - x_i)|) * (A + J * cos(θ_j - θ_i - Q_x)) - B * (x_j - x_i) / |(x_j - x_i)|^2 ]
        """

        # Calculate normalized_vectors and normalized_vectors_squared
        normalized_vectors = vectors / distances[:, :, np.newaxis]
        normalized_vectors_squared = vectors / np.square(distances[:, :, np.newaxis])

        # Scale the phase_cos_difference by J and add A
        scaled_phase_cos_difference = self._A + Js * phase_cos_difference

        # Get the velocity contributions
        velocity_contributions = (
            scaled_phase_cos_difference[:, :, np.newaxis] * normalized_vectors
            - self._B * normalized_vectors_squared
        )

        # If chiral is enabled calculate the chiral contribution as inherent velocities
        inherent_velocities = self._inherent_velocities

        if self._chiral:
            inherent_velocities = self._c * np.stack(
                [
                    np.cos(self._phase + np.pi / 2),
                    np.sin(self._phase + np.pi / 2),
                ],
                axis=1,
            )

        velocity = inherent_velocities + 1 / self._agents * np.sum(
            velocity_contributions, axis=1
        )

        return velocity

    def _get_J_values(self, positions: np.ndarray):
        """
        If agents have a target position the J1 values are calculated based on the distance to the target
        """

        # Target is (2,) while positions is (agents, 2) so we need to broadcast to perform the subtraction
        distToTargetVector = self._target[0, :2] - positions[:, :2][:, np.newaxis]

        # Calculate the distance to the target
        distToTarget = np.linalg.norm(distToTargetVector, axis=2)

        # Calculate the min and max distance to the target
        minDistToTarget = np.min(distToTarget)
        maxDistToTarget = np.max(distToTarget)

        J_values = (
            self._A
            * (np.absolute(distToTarget - minDistToTarget))
            / (maxDistToTarget - minDistToTarget)
        )

        return J_values

    def _get_phase_offset_terms(self):
        """
        Calculates Q_x and Q_theta which enable frequency coupling
        """

        natural_frequency_normalized = np.nan_to_num(
            self._natural_frequencies / np.absolute(self._natural_frequencies)
        )

        Q_x = (np.pi / 2) * np.absolute(
            np.subtract.outer(
                natural_frequency_normalized, natural_frequency_normalized
            )
        )

        Q_theta = (np.pi / 4) * np.absolute(
            np.subtract.outer(
                natural_frequency_normalized, natural_frequency_normalized
            )
        )

        return Q_x, Q_theta

    def get_phase_state(self):
        return self._phase

    def get_velocity(self):
        return self._velocity
