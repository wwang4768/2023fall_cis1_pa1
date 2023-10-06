import numpy as np
from scipy.spatial import KDTree
from calibration_library import *

if __name__ == "__main__":
    point = Point3D(1, 2, 3)
    rotation = Rotation3D(np.pi/4, np.pi/6, np.pi/8)
    frame = Frame3D(Point3D(10, 20, 30), rotation)

    transformed_point = frame.transform_point(point)

    print("Original Point:", point)
    print("Transformed Point:", transformed_point)
    print("Frame Rotation:", frame.rotation)
    print("Frame Origin:", frame.origin)

    """
    # Create an instance of the setRegistration class
    registration = setRegistration()

    # Define source and target point clouds
    source_points = np.array(...)  # Replace with your source point cloud
    target_points = np.array(...)  # Replace with your target point cloud

    # Perform ICP registration
    transformation_matrix = registration.icp(source_points, target_points)

    # Print the transformation matrix
    print("Transformation Matrix:")
    print(transformation_matrix)
    """

    # Generate synthetic source and target points (for demonstration purposes).
    np.random.seed(0)
    source_points = np.random.rand(100, 3)
    target_points = np.copy(source_points)

    # Apply a random transformation to the target points.
    rotation_matrix = np.array([[0.866, -0.5, 0],
                                [0.5, 0.866, 0],
                                [0, 0, 1]])
    translation_vector = np.array([0.1, 0.2, 0.3])
    target_points = np.dot(rotation_matrix, target_points.T).T + translation_vector

    # Perform pivot calibration.
    registration = setRegistration()

    # Example usage of pivot_calibration
    source_points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    target_points = np.array([[2, 3, 4], [5, 6, 7], [8, 9, 10]])

    transformation_matrix = registration.pivot_calibration(source_points, target_points)

    print("Estimated Transformation:")
    print(transformation_matrix)


