import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.spatial as spatial
from sklearn.neighbors import KDTree
import vg
import time


# == FUNCTIONS ========================================================================================================

# Takes points in [[x1, y1, z1], [x2, y2, z2]...] Numpy Array format
def thin_line(points, point_cloud_thickness=0.53, iterations=1,sample_points=0):
    total_start_time =  time.perf_counter()
    if sample_points != 0:
        points = points[:sample_points]
    
    # Sort points into KDTree for nearest neighbors computation later
    point_tree = spatial.cKDTree(points)

    # Empty array for transformed points
    new_points = []
    # Empty array for regression lines corresponding ^^ points
    regression_lines = []
    nn_time = 0
    rl_time = 0
    prj_time = 0
    for point in point_tree.data:
        # Get list of points within specified radius {point_cloud_thickness}
        start_time = time.perf_counter()
        points_in_radius = point_tree.data[point_tree.query_ball_point(point, point_cloud_thickness)]
        nn_time += time.perf_counter()- start_time

        # Get mean of points within radius
        start_time = time.perf_counter()
        data_mean = points_in_radius.mean(axis=0)

        # Calulate 3D regression line/principal component in point form with 2 coordinates
        uu, dd, vv = np.linalg.svd(points_in_radius - data_mean)
        linepts = vv[0] * np.mgrid[-1:1:2j][:, np.newaxis]
        linepts += data_mean
        regression_lines.append(list(linepts))
        rl_time += time.perf_counter() - start_time

        # Project original point onto 3D regression line
        start_time = time.perf_counter()
        ap = point - linepts[0]
        ab = linepts[1] - linepts[0]
        point_moved = linepts[0] + np.dot(ap,ab) / np.dot(ab,ab) * ab
        prj_time += time.perf_counter()- start_time

        new_points.append(list(point_moved))
    print("--- %s seconds to thin points ---" % (time.perf_counter() - total_start_time))
    print(f"Finding nearest neighbors for calculating regression lines: {nn_time}")
    print(f"Calculating regression lines: {rl_time}")
    print(f"Projecting original points on  regression lines: {prj_time}\n")
    return np.array(new_points), regression_lines

# Sorts points outputed from thin_points()
def sort_points(points, regression_lines, sorted_point_distance=0.2):
    sort_points_time = time.perf_counter()
    # Index of point to be sorted
    index = 0

    # sorted points array for left and right of intial point to be sorted
    sort_points_left = [points[index]]
    sort_points_right = []

    # Regression line of previously sorted point
    regression_line_prev = regression_lines[index][1] - regression_lines[index][0]

    # Sort points into KDTree for nearest neighbors computation later
    point_tree = spatial.cKDTree(points)


    # Iterative add points sequentially to the sort_points_left array
    while 1:
        # Calulate regression line vector; makes sure line vector is similar direction as previous regression line
        v = regression_lines[index][1] - regression_lines[index][0]
        if np.dot(regression_line_prev, v ) / (np.linalg.norm(regression_line_prev) * np.linalg.norm(v))  < 0:
            v = regression_lines[index][0] - regression_lines[index][1]
        regression_line_prev = v

        # Find point {distR_point} on regression line distance {sorted_point_distance} from original point 
        distR_point = points[index] + ((v / np.linalg.norm(v)) * sorted_point_distance)

        # Search nearest neighbors of distR_point within radius {sorted_point_distance / 3}
        points_in_radius = point_tree.data[point_tree.query_ball_point(distR_point, sorted_point_distance / 3)]
        if len(points_in_radius) < 1:
            break

        # Neighbor of distR_point with smallest angle to regression line vector is selected as next point in order
        # 
        # CAN BE OPTIMIZED
        # 
        nearest_point = points_in_radius[0]
        distR_point_vector = distR_point - points[index]
        nearest_point_vector = nearest_point - points[index]
        for x in points_in_radius: 
            x_vector = x - points[index]
            if vg.angle(distR_point_vector, x_vector) < vg.angle(distR_point_vector, nearest_point_vector):
                nearest_point_vector = nearest_point - points[index]
                nearest_point = x
        index = np.where(points == nearest_point)[0][0]

        # Add nearest point to 'sort_points_left' array
        sort_points_left.append(nearest_point)

    # Do it again but in the other direction of initial starting point 
    index = 0
    regression_line_prev = regression_lines[index][1] - regression_lines[index][0]
    while 1:
        # Calulate regression line vector; makes sure line vector is similar direction as previous regression line
        v = regression_lines[index][1] - regression_lines[index][0]
        if np.dot(regression_line_prev, v ) / (np.linalg.norm(regression_line_prev) * np.linalg.norm(v))  < 0:
            v = regression_lines[index][0] - regression_lines[index][1]
        regression_line_prev = v

        # Find point {distR_point} on regression line distance {sorted_point_distance} from original point 
        # 
        # Now vector is substracted from the point to go in other direction
        # 
        distR_point = points[index] - ((v / np.linalg.norm(v)) * sorted_point_distance)

        # Search nearest neighbors of distR_point within radius {sorted_point_distance / 3}
        points_in_radius = point_tree.data[point_tree.query_ball_point(distR_point, sorted_point_distance / 3)]
        if len(points_in_radius) < 1:
            break

        # Neighbor of distR_point with smallest angle to regression line vector is selected as next point in order
        # 
        # CAN BE OPTIMIZED
        # 
        nearest_point = points_in_radius[0]
        distR_point_vector = distR_point - points[index]
        nearest_point_vector = nearest_point - points[index]
        for x in points_in_radius: 
            x_vector = x - points[index]
            if vg.angle(distR_point_vector, x_vector) < vg.angle(distR_point_vector, nearest_point_vector):
                nearest_point_vector = nearest_point - points[index]
                nearest_point = x
        index = np.where(points == nearest_point)[0][0]

        # Add next point to 'sort_points_right' array
        sort_points_right.append(nearest_point)

    # Combine 'sort_points_right' and 'sort_points_left'
    sort_points_right.extend(sort_points_left[::-1])
    print("--- %s seconds to sort points ---" % (time.perf_counter() - sort_points_time))
    return np.array(sort_points_right)


# == RUN SCRIPT ========================================================================================================

start_time = time.perf_counter()

# Generate spiral points with noise 
total_rad = 10
z_factor = 3
noise = 0.06

num_true_pts = 200
s_true = np.linspace(0, total_rad, num_true_pts)
x_true = np.cos(s_true)
y_true = np.sin(s_true)
z_true = s_true/z_factor

num_sample_pts = 1200
s_sample = np.linspace(0, total_rad, num_sample_pts)
x_sample = np.cos(s_sample) + noise * np.random.randn(num_sample_pts)
y_sample = np.sin(s_sample) + noise * np.random.randn(num_sample_pts)
z_sample = s_sample/z_factor + noise * np.random.randn(num_sample_pts)

# Make a [(x1, y1, z1), (x2, y2, z2)...] Numpy Array
points = np.vstack((x_sample, y_sample, z_sample )).T

# Shuffle points
np.random.shuffle(points)

generate_points_time = time.perf_counter()
print(f'\nFor {len(x_sample)} points:')
print("\n--- %s seconds to generate sample points ---\n" % (generate_points_time - start_time))

# Thin & sort points
thinned_points, regression_lines = thin_line(points)


sorted_points = sort_points(thinned_points, regression_lines)

# Run thinning and sorting algorithms

# Plotting
fig2 = plt.figure(2)
ax3d = fig2.add_subplot(111, projection='3d')

# Plot unordedered point cloud
ax3d.plot(x_sample, y_sample, z_sample, 'm*')

# Plot sorted points
ax3d.plot(sorted_points.T[0], sorted_points.T[1], sorted_points.T[2], 'bo')

# Plot line going through sorted points 
ax3d.plot(sorted_points.T[0], sorted_points.T[1], sorted_points.T[2], '-b')

# Plot thinned points
# ax3d.plot(thinned_points.T[0], thinned_points.T[1], thinned_points.T[2], 'go')

fig2.show()
plt.show()