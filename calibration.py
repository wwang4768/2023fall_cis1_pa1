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

if __name__ == "__main__":
    point = Point3D(1, 2, 3)
    rotation = Rotation3D(np.pi/4, np.pi/6, np.pi/8)
    frame = Frame3D(Point3D(10, 20, 30), rotation)

    transformed_point = frame.transform_point(point)

    print("Original Point:", point)
    print("Transformed Point:", transformed_point)
    print("Frame Rotation:", frame.rotation)
    print("Frame Origin:", frame.origin)
