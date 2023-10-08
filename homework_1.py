import numpy as np
from scipy.spatial import KDTree
from calibration_library import *


def parseData(input_file):
    # Initialize an empty list to store the 3D coordinates
    points = [] 

    # Open the file for reading (use 'r' before the file path to specify a raw string)
    with open(input_file, 'r') as file:
        # Skip the first line
        next(file)

        for line in file:
            # Split each line into X, Y, and Z coordinates using ',' as the separator
            x, y, z = map(float, line.strip().split(','))
            point = (x, y, z)
            #point = Point3D(x, y, z)
            # Append the coordinates as a tuple to the list
            points.append(point)

    # Convert the list of tuples to a NumPy array for easier manipulation
    point_cloud = np.array(points)
    return point_cloud

def parseCalbody(point_cloud):
    # Number of optical markers on EM base
    d = point_cloud[:8]
    # number of optical markers on calibration object
    a = point_cloud[8:16]
    # number EM markers on calibration object
    c = point_cloud[-27:]
    return d, a, c

def parseFrame(point_cloud, frame_chunk):
    frames = []
    for i in range(0, len(point_cloud), frame_chunk):
        row = point_cloud[i:i+frame_chunk]
        frames.append(row)
    return frames

if __name__ == "__main__":
    # parse input data to consume
    calbody = 'C:\\Users\\Esther Wang\\Documents\\2023_CS655_CIS1\\2023fall_cis1\\pa1-debug-a-calbody.txt'
    calbody_point_cloud = parseData(calbody)
    d0, a0, c0 = parseCalbody(calbody_point_cloud)

    calreading = 'C:\\Users\\Esther Wang\\Documents\\2023_CS655_CIS1\\2023fall_cis1\\pa1-debug-a-calreadings.txt'
    calreading_point_cloud = parseData(calreading)
    #f1, f2, f3, f4, f5, f6, f7, f8 = parseCalreading(calbody_point_cloud, 8+8+27)

    # stores the list of 8 frames, each of which contains data of 8 optical markers on EM base, 
    # 8 optical markers on calibration object and 27 EM markers on calibration object
    calreading_frames = parseFrame(calreading_point_cloud, 8+8+27) 

    empivot = 'C:\\Users\\Esther Wang\\Documents\\2023_CS655_CIS1\\2023fall_cis1\\pa1-debug-a-empivot.txt'
    empivot_point_cloud = parseData(empivot)
    # stores the list of 12 frames, each of which contains data of 6 EM markers on probe 
    empivot_frames = parseFrame(empivot_point_cloud, 6) 
    
    optpivot = 'C:\\Users\\Esther Wang\\Documents\\2023_CS655_CIS1\\2023fall_cis1\\pa1-debug-a-optpivot.txt'
    optpivot_point_cloud = parseData(optpivot)
    # stores the list of 12 frames, each of which contains data of 8 optical markers on EM base 
    # and 6 EM markers on probe
    optpivot_frames = parseFrame(optpivot_point_cloud, 8+6) 

    # Perform pivot calibration.
    registration = setRegistration()

    # Example usage of pivot_calibration
    source_points = calreading_frames[0]
    target_points = calreading_frames[1]

    transformation_matrix = registration.pivot_calibration(source_points, target_points)

    print("Estimated Transformation:")
    print(transformation_matrix)

    transformed_points = registration.apply_transformation(source_points,transformation_matrix)
    print(transformed_points)

    #4a 
    #source_points = calreading_frames[0]
    #target_points = calreading_frames[1]

    """
    point = Point3D(1, 2, 3)
    rotation = Rotation3D(np.pi/4, np.pi/6, np.pi/8)
    frame = Frame3D(Point3D(10, 20, 30), rotation)

    transformed_point = frame.transform_point(point)

    print("Original Point:", point)
    print("Transformed Point:", transformed_point)
    print("Frame Rotation:", frame.rotation)
    print("Frame Origin:", frame.origin)


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
    """
    

