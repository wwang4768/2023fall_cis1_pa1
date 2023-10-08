import numpy as np


def rotation_z(theta):
    """Return a rotation matrix for a rotation about the z-axis by theta radians."""
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    return R


def pivot_calibration(R_list, P_list):
    """
    Perform pivot calibration using the least square method.

    Parameters:
    - R_list: A list of rotation matrices R_j.
    - P_list: A list of translation vectors P_j.

    Returns:
    - P_t: Position of the tip in the probe frame.
    - P_pivot: Pivot point in the sensor frame.
    """

    n = len(R_list)

    A = np.zeros((3 * n, 6))
    b = np.zeros((3 * n, 1))

    for i in range(n):
        R = R_list[i]
        P = P_list[i].reshape(3, 1)

        A[3 * i:3 * (i + 1), :3] = R
        A[3 * i:3 * (i + 1), 3:] = np.eye(3)

        b[3 * i:3 * (i + 1)] = P

    # Use numpy's lstsq function to solve the problem
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    P_t = x[:3]
    P_pivot = x[3:]

    return P_t, P_pivot


# Creating three rotation matrices for rotations of 30, 60, and 90 degrees about the z-axis
R_list = [rotation_z(np.radians(30)), rotation_z(np.radians(60)), rotation_z(np.radians(90))]

# Creating three translation vectors
P_list = [
    np.array([1, 2, 3]),
    np.array([4, 5, 6]),
    np.array([7, 8, 9])
]

# Now you can pass the R_list and P_list to the pivot_calibration function.
P_t, P_pivot = pivot_calibration(R_list, P_list)
print("P_t:", P_t)
print("P_pivot:", P_pivot)


