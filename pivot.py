import numpy as np
from scipy.linalg import lstsq


def pivot_calibration_multiple(Rj_list, Pj_list, pj_list):
    """
    Perform pivot calibration using the least squares method for multiple frames.

    Parameters:
    Rj_list (list of numpy.array): List of rotation matrices from sensor to probe.
    Pj_list (list of numpy.array): List of translation vectors from sensor to probe.
    pj_list (list of numpy.array): List of probe tip coordinates.

    Returns:
    list of tuples: Calibrated coordinates of the tip and pivot point for each frame.
    """

    calibrated_results = []

    # Iterate through each set of transformation components and probe tip coordinates
    for Rj, Pj, pj in zip(Rj_list, Pj_list, pj_list):
        # Construct the linear system of equations: [Rj | I] [Pt | P_pivot]^T = pj
        A = np.hstack((Rj, np.eye(4)))  # [Rj | I]
        B = pj  # pj

        # Solve the least squares problem: A * [Pt; P_pivot] = B
        result, _, _, _ = lstsq(A, B)

        # Extract Pt and P_pivot
        Pt = result[:4]  # Pt as a 4x1 vector
        Ppivot = result[4:]  # P_pivot as a 4x1 vector

        calibrated_results.append((Pt, Ppivot))

    return calibrated_results

# Example usage Replace with actual rotation matrices, translation vectors, and probe tip coordinates for each frame
# Rj_list = [np.eye(4),  # Identity matrix np.array([[0.866, -0.5, 0, 0], [0.5, 0.866, 0, 0], [0, 0, 1, 0], [0, 0, 0,
# 1]])]  # Example rotation matrices for each frame
#
# Pj_list = [np.array([0, 0, 0, 1]), np.array([1, 2, 3, 1])]  # Example translation vectors for each frame
#
# pj_list = [np.array([10, 20, 30, 1]), np.array([40, 50, 60, 1])]  # Example probe tip coordinates for each frame
#
# # Perform pivot calibration for each frame
# calibrated_results = pivot_calibration_multiple(Rj_list, Pj_list, pj_list)

# Print calibrated results for each frame
# for i, (calibrated_pt, calibrated_pivot) in enumerate(calibrated_results):
#     print(f"Calibrated Pt for frame {i+1}:", calibrated_pt)
#     print(f"Calibrated Ppivot for frame {i+1}:", calibrated_pivot)
