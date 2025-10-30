import time
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - Required for 3D projection
from matplotlib.figure import Figure

from curve_extraction import thin_line, sort_points


# == VISUALIZATION FUNCTIONS ==========================================================================================


def plot_curve_reconstruction(
    original_points: npt.NDArray[np.float64],
    sorted_points: npt.NDArray[np.float64],
    thinned_points: Optional[npt.NDArray[np.float64]] = None,
    figsize: tuple[int, int] = (10, 8),
    show_thinned: bool = False
) -> Figure:
    """Plot the curve reconstruction results in 3D.
    
    Args:
        original_points: Original noisy point cloud as (x, y, z) arrays
        sorted_points: Sorted points after reconstruction
        thinned_points: Optional thinned points to display
        figsize: Figure size as (width, height)
        show_thinned: Whether to show thinned points
        
    Returns:
        The matplotlib Figure object
    """
    # Plotting
    fig: Figure = plt.figure(figsize=figsize)
    ax3d = fig.add_subplot(111, projection="3d")
    
    # Plot unordered point cloud
    ax3d.plot(
        original_points[:, 0],
        original_points[:, 1],
        original_points[:, 2],
        "m*",
        label="Original noisy points"
    )
    
    # Plot sorted points
    ax3d.plot(
        sorted_points.T[0], sorted_points.T[1], sorted_points.T[2],
        "bo", label="Sorted points"
    )
    
    # Plot line going through sorted points
    ax3d.plot(
        sorted_points.T[0], sorted_points.T[1], sorted_points.T[2],
        "-b", linewidth=2, label="Reconstructed curve"
    )
    
    # Plot thinned points (optional)
    if show_thinned and thinned_points is not None:
        ax3d.plot(
            thinned_points.T[0], thinned_points.T[1], thinned_points.T[2],
            'go', label="Thinned points"
        )
    
    ax3d.legend()
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.set_title("3D Curve Reconstruction from Point Cloud")
    
    return fig


def show_plot() -> None:
    """Display all matplotlib plots."""
    plt.show()


# == DEMO FUNCTIONS ====================================================================================================


def generate_spiral_points(
    total_rad: float = 10,
    z_factor: float = 3,
    noise: float = 0.10,
    num_sample_pts: int = 3000
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Generate noisy spiral points for testing."""
    s_sample: npt.NDArray[np.float64] = np.linspace(0, total_rad, num_sample_pts)
    x_sample: npt.NDArray[np.float64] = np.cos(s_sample) + noise * np.random.randn(num_sample_pts)
    y_sample: npt.NDArray[np.float64] = np.sin(s_sample) + noise * np.random.randn(num_sample_pts)
    z_sample: npt.NDArray[np.float64] = s_sample / z_factor + noise * np.random.randn(num_sample_pts)
    
    # Make a [(x1, y1, z1), (x2, y2, z2)...] Numpy Array
    points: npt.NDArray[np.float64] = np.vstack((x_sample, y_sample, z_sample)).T
    
    return points, x_sample, y_sample, z_sample


# == RUN SCRIPT ========================================================================================================


def main() -> None:
    """Main execution function."""
    start_time = time.perf_counter()
    
    # Generate spiral points with noise
    points, x_sample, y_sample, z_sample = generate_spiral_points()
    
    # Shuffle points
    np.random.shuffle(points)
    
    generate_points_time = time.perf_counter()
    elapsed: float = generate_points_time - start_time
    print(f"\nFor {len(x_sample)} points:")
    print(f"\n--- {elapsed:.6f} seconds to generate sample points ---\n")
    
    # Thin & sort points
    thinned_points, regression_lines = thin_line(points)
    sorted_points = sort_points(thinned_points, regression_lines)
    
    # Create visualization
    original_points: npt.NDArray[np.float64] = np.vstack((x_sample, y_sample, z_sample)).T
    plot_curve_reconstruction(
        original_points=original_points,
        sorted_points=sorted_points,
        thinned_points=thinned_points,
        show_thinned=False
    )
    
    show_plot()


if __name__ == "__main__":
    main()
