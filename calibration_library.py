import numpy as np
from scipy.spatial import KDTree
import random

class Point3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"

    def translate(self, dx, dy, dz):
        self.x += dx
        self.y += dy
        self.z += dz

class Rotation3D:
    def __init__(self, yaw=0, pitch=0, roll=0):
        self.yaw = yaw  # Rotation around Z-axis
        self.pitch = pitch  # Rotation around Y-axis
        self.roll = roll  # Rotation around X-axis

    def __str__(self):
        return f"Yaw: {self.yaw}, Pitch: {self.pitch}, Roll: {self.roll}"

    def rotate(self, yaw, pitch, roll):
        self.yaw += yaw
        self.pitch += pitch
        self.roll += roll

    def matrix(self):
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(self.roll), -np.sin(self.roll)],
                        [0, np.sin(self.roll), np.cos(self.roll)]])

        R_y = np.array([[np.cos(self.pitch), 0, np.sin(self.pitch)],
                        [0, 1, 0],
                        [-np.sin(self.pitch), 0, np.cos(self.pitch)]])

        R_z = np.array([[np.cos(self.yaw), -np.sin(self.yaw), 0],
                        [np.sin(self.yaw), np.cos(self.yaw), 0],
                        [0, 0, 1]])

        return np.dot(R_z, np.dot(R_y, R_x))

class Frame3D:
    def __init__(self, origin=Point3D(0, 0, 0), rotation=Rotation3D()):
        self.origin = origin
        self.rotation = rotation

    def transform_point(self, point):
        # Apply rotation
        rotated_point = np.dot(self.rotation.matrix(), np.array([point.x, point.y, point.z]))

        # Apply translation
        transformed_point = Point3D(rotated_point[0] + self.origin.x,
                                    rotated_point[1] + self.origin.y,
                                    rotated_point[2] + self.origin.z)

        return transformed_point

class setRegistration:
    def icp(self, source_points, target_points, correspondences, max_iterations=100, tolerance=1e-6):
        """
        Perform 3D point set registration using the Iterative Closest Point (ICP) algorithm with known correspondences.

        Args:
            source_points (numpy.ndarray): Source 3D point set (Nx3).
            target_points (numpy.ndarray): Target 3D point set (Nx3).
            correspondences (numpy.ndarray): Correspondence indices from source to target (Nx1).
            max_iterations (int): Maximum number of iterations.
            tolerance (float): Convergence tolerance (change in transformation).

        Returns:
            numpy.ndarray: The transformation matrix (4x4) that aligns the source points with the target points.
        """
        transformation = np.identity(4)

        for iteration in range(max_iterations):
            # Use the known correspondences to select the corresponding points.
            correspondences_source = source_points
            correspondences_target = target_points[correspondences]

            # Calculate the transformation that minimizes the distance between corresponding points.
            transformation_update = self.compute_rigid_transform(correspondences_source, correspondences_target)

            # Apply the transformation to the source point set.
            source_points = self.apply_transformation(source_points, transformation_update)

            # Update the overall transformation matrix.
            transformation = np.dot(transformation_update, transformation)

            # Check for convergence based on the change in transformation.
            if np.linalg.norm(np.identity(4) - transformation_update) < tolerance:
                break

        return transformation


    def compute_rigid_transform(self, source_points, target_points):
        """
        Compute the rigid transformation matrix (4x4) that aligns source points with target points.

        Args:
            source_points (numpy.ndarray): Source 3D point set (Nx3).
            target_points (numpy.ndarray): Target 3D point set (Nx3).

        Returns:
            numpy.ndarray: The transformation matrix (4x4) that aligns the source points with the target points.
        """
        source_centered = source_points - np.mean(source_points, axis=0)
        target_centered = target_points - np.mean(target_points, axis=0)

        # Compute the covariance matrix.
        covariance_matrix = np.dot(source_centered.T, target_centered)

        # Compute the Singular Value Decomposition (SVD).
        U, _, Vt = np.linalg.svd(covariance_matrix)

        # Ensure proper rotation (handle reflections).
        if np.linalg.det(np.dot(U, Vt)) < 0:
            Vt[-1, :] *= -1

        # Compute the rotation matrix and translation vector.
        rotation_matrix = np.dot(U, Vt)
        translation_vector = np.mean(target_points, axis=0) - np.dot(rotation_matrix, np.mean(source_points, axis=0))

        # Construct the transformation matrix.
        transformation = np.identity(4)
        transformation[:3, :3] = rotation_matrix
        transformation[:3, 3] = translation_vector

        return transformation

import numpy as np

class Calibration:
    def pivot_calibration(self, source_points, target_points, max_iterations=100, tolerance=1e-6):
        """
        Perform pivot calibration to determine the transformation matrix between two point sets.

        Args:
            source_points (numpy.ndarray): 3D source points (Nx3).
            target_points (numpy.ndarray): Corresponding 3D target points (Nx3).
            max_iterations (int): Maximum number of iterations.
            tolerance (float): Convergence tolerance (change in transformation).

        Returns:
            numpy.ndarray: Transformation matrix (4x4) that aligns the source points with the target points.
        """
        transformation = np.identity(4)
        prev_error = float('inf')

        for iteration in range(max_iterations):
            # Calculate the transformation error metric using the current transformation
            error = self.compute_error(source_points, target_points, transformation)

            if abs(prev_error - error) < tolerance:
                break  # Convergence criteria met

            # Optimize the transformation to minimize the error metric
            self.update_transformation(transformation, source_points, target_points)

            prev_error = error

        return transformation

    def compute_error(self, source_points, target_points, transformation):
        """
        Compute the error metric between source and target points using a given transformation.

        Args:
            source_points (numpy.ndarray): 3D source points (Nx3).
            target_points (numpy.ndarray): Corresponding 3D target points (Nx3).
            transformation (numpy.ndarray): Transformation matrix (4x4).

        Returns:
            float: Error metric between the transformed source points and target points.
        """
        # Transform the source points using the current transformation
        transformed_source = self.apply_transformation(source_points, transformation)

        # Calculate the sum of squared distances between transformed source and target points
        squared_distances = np.sum((transformed_source - target_points) ** 2)
        error = np.sqrt(squared_distances)

        return error

    def update_transformation(self, transformation, source_points, target_points, learning_rate=0.01):
            """
            Update the transformation to minimize the error metric using gradient descent.

            Args:
                transformation (numpy.ndarray): Transformation matrix (4x4) to be updated.
                source_points (numpy.ndarray): 3D source points (Nx3).
                target_points (numpy.ndarray): Corresponding 3D target points (Nx3).
                learning_rate (float): Learning rate for gradient descent.

            Note: This implementation updates both the rotation matrix and translation vector.
            """
            num_points = source_points.shape[0]

            # Calculate the error between transformed source and target points
            transformed_source = self.apply_transformation(source_points, transformation)
            error = transformed_source - target_points

            # Calculate the gradients with respect to the transformation parameters
            gradient_rotation = np.zeros((3, 3))
            gradient_translation = np.zeros(3)

            for i in range(num_points):
                # Gradient for rotation (using the Jacobian of the transformation)
                gradient_rotation += 2 * np.outer(source_points[i], error[i])

                # Gradient for translation
                gradient_translation += 2 * error[i]

            # Update the rotation matrix using gradient descent
            new_rotation_matrix = transformation[:3, :3] - learning_rate * gradient_rotation

            # Update the translation vector using gradient descent
            new_translation_vector = transformation[:3, 3] - learning_rate * gradient_translation

            # Update the transformation matrix
            new_transformation = np.identity(4)
            new_transformation[:3, :3] = new_rotation_matrix
            new_transformation[:3, 3] = new_translation_vector

            # Update the original transformation matrix in-place
            transformation[:] = new_transformation[:]


        def apply_transformation(self, points, transformation):
            """
            Apply a 4x4 transformation matrix to a set of 3D points.

            Args:
                points (numpy.ndarray): 3D points to be transformed (Nx3).
                transformation (numpy.ndarray): Transformation matrix (4x4).

            Returns:
                numpy.ndarray: Transformed 3D points (Nx3).
            """
            homogeneous_points = np.column_stack((points, np.ones(points.shape[0])))
            transformed_points = np.dot(transformation, homogeneous_points.T).T
            return transformed_points[:, :3]

# if __name__ == "__main__":
#     point = Point3D(1, 2, 3)
#     rotation = Rotation3D(np.pi/4, np.pi/6, np.pi/8)
#     frame = Frame3D(Point3D(10, 20, 30), rotation)

#     transformed_point = frame.transform_point(point)

#     print("Original Point:", point)
#     print("Transformed Point:", transformed_point)
#     print("Frame Rotation:", frame.rotation)
#     print("Frame Origin:", frame.origin)
