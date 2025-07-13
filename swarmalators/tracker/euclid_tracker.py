import math
import numpy as np
from scipy.optimize import linear_sum_assignment


class EuclideanDistTracker:
    def __init__(self):
        self.center_points = {}
        self.id_count = 0

    def init(self, inital_positions):
        """
        Initalize the center points

        Args:
            inital_positions (list): The inital positions of the objects in form [x, y]
        """
        try:
            for position in inital_positions:
                x, y = position

                self.center_points[self.id_count] = (x, y)
                self.id_count += 1
        except Exception as e:
            print(inital_positions)
            raise RuntimeError("Inital positions must be in form [x, y, w, h]")

    def _get_closest_id(self, center_x, center_y):
        """
        Gets the closest id to the center point

        Args:
            center_x (int): The x coordinate of the center point
            center_y (int): The y coordinate of the center point

        Returns:
            int: The id of the closest point
        """
        closest_id = -1
        min_dist = math.inf

        for id, point in self.center_points.items():
            dist = math.sqrt((center_x - point[0]) ** 2 + (center_y - point[1]) ** 2)

            if dist < min_dist:
                min_dist = dist
                closest_id = id

        return closest_id

    def update(self, object_coords):
        """
        Update the center points of the objects using the Hungarian algorithm
        to minimize the total distance.

        Args:
            object_coords (list): The center points of the objects in form [x, y]

        Returns:
            list: The new center points in form [id, (x, y)]
        """
        if len(object_coords) != len(self.center_points):
            raise RuntimeError("Number of objects must match number of center points")

        # Create a cost matrix based on distances
        cost_matrix = np.zeros((len(self.center_points), len(object_coords)))

        for i, (id, point) in enumerate(self.center_points.items()):
            for j, coord in enumerate(object_coords):
                cost_matrix[i, j] = np.linalg.norm(np.array(point) - np.array(coord))

        # Solve the assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Update the center points with the new matched coordinates
        new_center_points = {}
        for row, col in zip(row_ind, col_ind):
            id = list(self.center_points.keys())[row]
            new_center_points[id] = tuple(object_coords[col])

        self.center_points = new_center_points

        # Return the new center points sorted by id
        return sorted(new_center_points.items(), key=lambda x: x[0])


# import math
# import numpy as np
# from scipy.optimize import linear_sum_assignment


# class EuclideanDistTracker:
#     def __init__(self):
#         self.center_points = {}
#         self.id_count = 0

#     def init(self, inital_positions):
#         """
#         Initalize the center points

#         Args:
#             inital_positions (list): The inital positions of the objects in form [x, y]
#         """
#         try:
#             for position in inital_positions:
#                 x, y = position

#                 self.center_points[self.id_count] = (x, y)
#                 self.id_count += 1
#         except Exception as e:
#             print(inital_positions)
#             raise RuntimeError("Inital positions must be in form [x, y, w, h]")

#     def update(self, object_coords):
#         """
#         Update the center points of the objects

#         Args:
#             objects_rect (list): The center points of the objects in form [x, y]

#         Returns:
#             list: The new center points in form [id, (x, y)]
#         """
#         if len(object_coords) != len(self.center_points):
#             raise RuntimeError("Number of objects must match number of center points")

#         old_positions = list(self.center_points.values())

#         cost_matrix = np.linalg.norm(
#             np.array(old_positions)[:, None] - np.array(object_coords), axis=2
#         )

#         row_ind, col_ind = linear_sum_assignment(cost_matrix)

#         # Update the center points with the new matched coordinates
#         new_center_points = {}
#         for row, col in zip(row_ind, col_ind):
#             new_center_points[row] = object_coords[col]

#         # Update self.center_points with new values
#         self.center_points = new_center_points

#         # Return the new center points sorted by id
#         return sorted(new_center_points.items(), key=lambda x: x[0])
