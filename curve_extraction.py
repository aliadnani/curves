import time
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import scipy.spatial as spatial
import vg


# == FUNCTIONS ========================================================================================================


def thin_line(
    points: npt.NDArray[np.float64],
    point_cloud_thickness: float = 0.53,
    iterations: int = 1,
    sample_points: int = 0
) -> Tuple[npt.NDArray[np.float64], List[List[List[float]]]]:
    """Takes points in [[x1, y1, z1], [x2, y2, z2]...] Numpy Array format."""
    total_start_time = time.perf_counter()
    if sample_points:
        points = points[:sample_points]

    # Sort points into KDTree for nearest neighbors computation later
    point_tree = spatial.cKDTree(points)

    # Empty arrays for transformed points and regression lines
    new_points: List[List[float]] = []
    regression_lines: List[List[List[float]]] = []
    
    # Timing variables
    nearest_neighbor_time = regression_line_time = projection_time = 0.0
    
    for point in point_tree.data:
        # Get list of points within specified radius {point_cloud_thickness}
        start_time = time.perf_counter()
        points_in_radius: npt.NDArray[np.float64] = point_tree.data[
            point_tree.query_ball_point(point, point_cloud_thickness)
        ]
        nearest_neighbor_time += time.perf_counter() - start_time

        # Get mean of points within radius
        start_time = time.perf_counter()
        data_mean: npt.NDArray[np.float64] = points_in_radius.mean(axis=0)

        # Calculate 3D regression line/principal component in point form with 2 coordinates
        _, _, principal_components = np.linalg.svd(points_in_radius - data_mean)
        line_points: npt.NDArray[np.float64] = principal_components[0] * np.mgrid[-1:1:2j][:, np.newaxis]
        line_points += data_mean
        regression_lines.append(line_points.tolist())
        regression_line_time += time.perf_counter() - start_time

        # Project original point onto 3D regression line
        start_time = time.perf_counter()
        point_to_line_start: npt.NDArray[np.float64] = point - line_points[0]
        line_direction: npt.NDArray[np.float64] = line_points[1] - line_points[0]
        point_moved: npt.NDArray[np.float64] = (
            line_points[0] + np.dot(point_to_line_start, line_direction) / np.dot(line_direction, line_direction) * line_direction
        )
        projection_time += time.perf_counter() - start_time

        new_points.append(point_moved.tolist())
    
    elapsed_time = time.perf_counter() - total_start_time
    print(f"--- {elapsed_time:.6f} seconds to thin points ---")
    print(f"Finding nearest neighbors for calculating regression lines: {nearest_neighbor_time:.6f}")
    print(f"Calculating regression lines: {regression_line_time:.6f}")
    print(f"Projecting original points on regression lines: {projection_time:.6f}\n")
    
    return np.array(new_points), regression_lines


def sort_points(
    points: npt.NDArray[np.float64],
    regression_lines: List[List[List[float]]],
    sorted_point_distance: float = 0.2
) -> npt.NDArray[np.float64]:
    """Sorts points output from thin_line()."""
    sort_points_time = time.perf_counter()
    
    # Sort points into KDTree for nearest neighbors computation later
    point_tree = spatial.cKDTree(points)
    
    def traverse_direction(start_index: int, direction: int = 1) -> List[npt.NDArray[np.float64]]:
        """Traverse points in one direction from start index."""
        result: List[npt.NDArray[np.float64]] = []
        current_index = start_index
        previous_regression_vector: npt.NDArray[np.float64] = (
            np.array(regression_lines[current_index][1]) - np.array(regression_lines[current_index][0])
        )
        
        while True:
            # Calculate regression line vector; ensure similar direction as previous
            current_regression_vector: npt.NDArray[np.float64] = (
                np.array(regression_lines[current_index][1]) - np.array(regression_lines[current_index][0])
            )
            cos_angle: np.float64 = np.dot(previous_regression_vector, current_regression_vector) / (
                np.linalg.norm(previous_regression_vector) * np.linalg.norm(current_regression_vector)
            )
            if cos_angle < 0:
                current_regression_vector = -current_regression_vector
            previous_regression_vector = current_regression_vector
            
            # Find point on regression line at specified distance
            target_point: npt.NDArray[np.float64] = (
                points[current_index] + direction * (current_regression_vector / np.linalg.norm(current_regression_vector)) * sorted_point_distance
            )
            
            # Search nearest neighbors
            points_in_radius: npt.NDArray[np.float64] = point_tree.data[
                point_tree.query_ball_point(target_point, sorted_point_distance / 3)
            ]
            if len(points_in_radius) < 1:
                break
            
            # Find neighbor with smallest angle to regression line vector
            # CAN BE OPTIMIZED
            target_point_vector: npt.NDArray[np.float64] = target_point - points[current_index]
            nearest_point: npt.NDArray[np.float64] = min(
                points_in_radius,
                key=lambda x: vg.angle(target_point_vector, x - points[current_index])
            )
            
            current_index = np.where(np.all(points == nearest_point, axis=1))[0][0]
            result.append(nearest_point)
        
        return result
    
    # Start from first point and traverse both directions
    start_index = 0
    sort_points_left: List[npt.NDArray[np.float64]] = (
        [points[start_index]] + traverse_direction(start_index, direction=1)
    )
    sort_points_right: List[npt.NDArray[np.float64]] = traverse_direction(start_index, direction=-1)
    
    # Combine both directions: right (reversed) + left
    sorted_points_list: List[npt.NDArray[np.float64]] = sort_points_right[::-1] + sort_points_left
    
    elapsed_time = time.perf_counter() - sort_points_time
    print(f"--- {elapsed_time:.6f} seconds to sort points ---")
    
    return np.array(sorted_points_list)
