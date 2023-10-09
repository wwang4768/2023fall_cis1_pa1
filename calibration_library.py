import numpy as np
from scipy.optimize import least_squares

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
    def calculate_3d_transformation(self, source_points, target_points):
        """
        Calculate the 4x4 transformation matrix using quaternions for 3D rigid body transformation.

        Args:
            source_points (numpy.ndarray): Source 3D point set (Nx3).
            target_points (numpy.ndarray): Target 3D point set (Nx3).

        Returns:
            numpy.ndarray: The 4x4 transformation matrix.
        """
        # Compute the centroids of source and target points
        source_centroid = np.mean(source_points, axis=0)
        target_centroid = np.mean(target_points, axis=0)

        # Center the points by subtracting centroids
        centered_source = source_points - source_centroid
        centered_target = target_points - target_centroid

        # Compute the covariance matrix
        H = np.dot(centered_source.T, centered_target)

        # Use singular value decomposition (SVD) to find rotation matrix R
        U, _, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        # Ensure proper rotation (handle reflections)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)

        # Calculate translation vector t
        t = target_centroid - np.dot(R, source_centroid)

        # Create a 4x4 transformation matrix
        transformation_matrix = np.identity(4)
        transformation_matrix[:3, :3] = R
        transformation_matrix[:3, 3] = t

        return transformation_matrix

    def icp(self, source_points, target_points, max_iterations=100, tolerance=1e-6):
        """
        Perform 3D point set registration using the Iterative Closest Point (ICP) algorithm without known correspondences.

        Args:
            source_points (numpy.ndarray): Source 3D point set (Nx3).
            target_points (numpy.ndarray): Target 3D point set (Nx3).
            max_iterations (int): Maximum number of iterations.
            tolerance (float): Convergence tolerance (change in transformation).

        Returns:
            numpy.ndarray: The transformation matrix (4x4) that aligns the source points with the target points.
        """
        transformation = np.identity(4)

        for iteration in range(max_iterations):
            # Calculate the transformation that minimizes the distance between points.
            transformation_update = self.compute_rigid_transform(source_points, target_points)

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

    def update_transformation(self, transformation, source_points, target_points, learning_rate=0.01, max_iterations=100, tolerance=1e-6):
        """
        Update the transformation to minimize the error metric using gradient descent.

        Args:
            transformation (numpy.ndarray): Transformation matrix (4x4) to be updated.
            source_points (numpy.ndarray): 3D source points (Nx3).
            target_points (numpy.ndarray): Corresponding 3D target points (Nx3).
            learning_rate (float): Learning rate for gradient descent.
            max_iterations (int): Maximum number of optimization iterations.
            tolerance (float): Convergence tolerance for stopping optimization.
        """
        num_points = source_points.shape[0]

        for iteration in range(max_iterations):
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

            # Clip gradients to prevent numerical instability
            max_gradient_value = 1e6  # Adjust as needed
            gradient_rotation = np.clip(gradient_rotation, -max_gradient_value, max_gradient_value)
            gradient_translation = np.clip(gradient_translation, -max_gradient_value, max_gradient_value)

            # Update the rotation matrix using gradient descent
            new_rotation_matrix = transformation[:3, :3] - learning_rate * gradient_rotation

            # Update the translation vector using gradient descent
            new_translation_vector = transformation[:3, 3] - learning_rate * gradient_translation

            # Update the transformation matrix
            new_transformation = np.identity(4)
            new_transformation[:3, :3] = new_rotation_matrix
            new_transformation[:3, 3] = new_translation_vector

            # Check for convergence based on the change in transformation
            if np.linalg.norm(new_transformation - transformation) < tolerance:
                break


            # Update the original transformation matrix in-place
            transformation[:] = new_transformation[:]

    def apply_transformation(self, points, transformation):
        """
        Apply a 4x4 transformation matrix to a set of 3D points and normalize the result.

        Args:
            points (numpy.ndarray): 3D points to be transformed (Nx3).
            transformation (numpy.ndarray): Transformation matrix (4x4).

        Returns:
            numpy.ndarray: Normalized transformed 3D points (Nx3).
        """
        # Ensure points are in homogeneous coordinates (add a column of ones)
        homogeneous_points = np.column_stack((points, np.ones((points.shape[0], 1))))
        #print(homogeneous_points.shape)
        
        # Perform the transformation using matrix multiplication
        transformed_points = np.dot(homogeneous_points, transformation.T)

        # Normalize the points by dividing by the last column of the result
        normalized_points = transformed_points[:, :3] / transformed_points[:, 3, np.newaxis]

        return normalized_points

    def pivot_calibration(self, transformation_matrices):
        # Convert transformation_matrices to numpy array
        transformation_matrices = np.array(transformation_matrices)

        # Number of frames
        num_frames = len(transformation_matrices)

        # Initialize parameters (p_tip/p_pivot) with an initial guess
        # 3 for p_tip and 3 for p_pivot
        initial_guess = np.zeros(6)

        # Define the objective function for the least squares optimization
        def objective_function(parameters):
            p_tip = parameters[:3].reshape(3, 1)
            p_pivot = parameters[3:].reshape(3, 1)

            # Create an empty array to store the transformed measurements
            transformed_frames = np.zeros((num_frames, 3))

            # Iterate through each frame
            for j in range(num_frames):
                R_j = transformation_matrices[j, :3, :3]
                error_matrix = np.hstack([R_j, -np.eye(3)])
                concatenated_points = np.vstack([p_tip, p_pivot])
                transformed_frames[j] = np.dot(error_matrix, concatenated_points).flatten()

            # Calculate the error as the difference between transformed and -p_j
            error = transformed_frames + transformation_matrices[:, :3, 3]

            return error.flatten()

        # Perform the least squares optimization to find the parameters
        result = least_squares(objective_function, initial_guess)

        # Extract the solution for p_tip and p_pivot
        p_tip_solution = result.x[:3]
        p_pivot_solution = result.x[3:]

        return p_tip_solution, p_pivot_solution

# if __name__ == "__main__":
#     point = Point3D(1, 2, 3)
#     rotation = Rotation3D(np.pi/4, np.pi/6, np.pi/8)
#     frame = Frame3D(Point3D(10, 20, 30), rotation)

#     transformed_point = frame.transform_point(point)

#     print("Original Point:", point)
#     print("Transformed Point:", transformed_point)
#     print("Frame Rotation:", frame.rotation)
#     print("Frame Origin:", frame.origin)


# def pivot_calibration(self, source_points, target_points, max_iterations=100, tolerance=1e-6):
#         """
#         Perform pivot calibration to determine the transformation matrix between two point sets.

#         Args:
#             source_points (numpy.ndarray): 3D source points (Nx3).
#             target_points (numpy.ndarray): Corresponding 3D target points (Nx3).
#             max_iterations (int): Maximum number of iterations.
#             tolerance (float): Convergence tolerance (change in transformation).

#         Returns:
#             numpy.ndarray: Transformation matrix (4x4) that aligns the source points with the target points.
#         """
#         transformation = np.identity(4)
#         prev_error = float('inf')

#         for iteration in range(max_iterations):
#             # Calculate the transformation error metric using the current transformation
#             error = self.compute_error(source_points, target_points, transformation)

#             if abs(prev_error - error) < tolerance:
#                 break  # Convergence criteria met

#             # Optimize the transformation to minimize the error metric
#             self.update_transformation(transformation, source_points, target_points)

#             prev_error = error

#         return transformation