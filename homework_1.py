import numpy as np
from scipy.spatial import KDTree
from calibration_library import *
import copy


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

def parseOptpivot(point_cloud, len_chunk_d, len_chunk_h):
    frames_d = []
    frames_h = []

    # Define the chunk sizes
    chunk_size_d = len_chunk_d
    chunk_size_h = len_chunk_h

    # Initialize the chunk index and current list
    current_list = 'D'
    temp = []

    # Iterate through all items
    for p in point_cloud:
        temp.append(p)
        
        # Check if we have reached the chunk size
        if len(temp) == chunk_size_d and current_list == 'D':
            frames_d.append(temp)
            temp = []
            current_list = 'H'
        elif len(temp) == chunk_size_h and current_list == 'H':
            frames_h.append(temp)
            temp = []
            current_list = 'D'
    return frames_d, frames_h

def parseFrame(point_cloud, frame_chunk):
    frames = []
    for i in range(0, len(point_cloud), frame_chunk):
        row = point_cloud[i:i+frame_chunk]
        frames.append(row)
    return frames

if __name__ == "__main__":
    # parse input data to consume
    base_path = 'C:\\Users\\Esther Wang\\Documents\\2023_CS655_CIS1\\2023fall_cis1\\pa1_student_data\\PA1 Student Data\\pa1-debug-' 
    choose_set = 'a'

    calbody = base_path + choose_set + '-calbody.txt'
    calbody_point_cloud = parseData(calbody)
    d0, a0, c0 = parseCalbody(calbody_point_cloud)

    calreading = base_path + choose_set + '-calreadings.txt'
    calreading_point_cloud = parseData(calreading)
    
    # stores the list of 8 frames, each of which contains data of 8 optical markers on EM base, 
    # 8 optical markers on calibration object and 27 EM markers on calibration object
    calreading_frames = parseFrame(calreading_point_cloud, 8+8+27) 
    
    empivot = base_path + choose_set + '-empivot.txt'
    empivot_point_cloud = parseData(empivot)
    # stores the list of 12 frames, each of which contains data of 6 EM markers on probe 
    empivot_frames = parseFrame(empivot_point_cloud, 6) 
    
    optpivot = base_path + choose_set + '-optpivot.txt'
    optpivot_point_cloud = parseData(optpivot)
    # stores the list of 12 frames, each of which contains data of 8 optical markers on EM base 
    # and 6 EM markers on probe
    optpivot_em_frames, optpivot_opt_frames = parseOptpivot(optpivot_point_cloud, 8, 6) 
    
    # Perform pivot calibration.
    registration = setRegistration()

    #4a 
    source_points_d = d0
    # print(source_points_d)
    trans_matrix_d = []
    target_points = []

    for i in range(8):
        target_points = calreading_frames[i][:8]
        transformation_matrix = registration.calculate_3d_transformation(source_points_d, target_points)
        trans_matrix_d.append(transformation_matrix)
    #print(trans_matrix_d[0])

    #4b 
    source_points_a = a0
    trans_matrix_a = []
    target_points = []

    for i in range(8):
        target_points = calreading_frames[i][8:16]
        transformation_matrix = registration.calculate_3d_transformation(source_points_a, target_points)
        trans_matrix_a.append(transformation_matrix)
    
    np.set_printoptions(formatter={'float': '{:.2f}'.format})
    #print(trans_matrix_a)
    """
    print("Estimated Transformation:")
    print(transformation_matrix)

    transformed_points = registration.apply_transformation(source_points,transformation_matrix)
    if(transformed_points.all() == target_points.all()):
        print("true")
    """

    # 4d
    source_points_c = c0
    transformation_matrix = np.dot(np.linalg.inv(trans_matrix_d[0]), trans_matrix_a[0])
    transformed_point = registration.apply_transformation(source_points_c, transformation_matrix)
    #print(transformation_matrix)
    #print(transformed_point)

    # 4e
    # Initalize the set for gj = Gj - G0
    translated_points = copy.deepcopy(empivot_frames)
    # Find centroid of Gj (the original position of 6 EM markers on the probe)
    mid_pts = np.mean(empivot_frames, axis=1)
    # Find transformation matrix for 12 frames
    trans_matrix_e = []

    for i in range(12):
        for j in range(6):
        # fill out gj 
            p = empivot_frames[i][j] - mid_pts[i]
            translated_points[i][j] = p
    
    # fix gj as the original starting positions
    source_points = translated_points[0]
    for i in range(12):
        target_points = empivot_frames[i]
        transformation_matrix = registration.calculate_3d_transformation(source_points, target_points)
        trans_matrix_e.append(transformation_matrix)
    #print(trans_matrix_e)

    p_tip, p_pivot = registration.pivot_calibration(trans_matrix_e)
    #print(p_pivot)

    # 4f
    # Initalize the set for gj = Gj - G0
    translated_points = copy.deepcopy(optpivot_opt_frames)
    H_prime = []
    # Find centroid of Gj (the original position of 6 EM markers on the probe)
    mid_pts = np.mean(optpivot_opt_frames, axis=1)
    # Find transformation matrix for 12 frames
    trans_matrix_f = []

    # Calculate Fd
    source_points_d = optpivot_em_frames[0]
    transformation_matrix_Fd = []
    target_points = []

    for i in range(12):
        target_points = optpivot_em_frames[i]
        transformation_matrix = registration.calculate_3d_transformation(source_points_d, target_points)
        transformation_matrix_Fd.append(transformation_matrix)

    for i in range(12):
        for j in range(6):
        # fill out gj 
            p = optpivot_opt_frames[i][j] - mid_pts[i]
            translated_points[i][j] = p
    """
    for chunk in translated_points:
        chunk_array = np.vstack(chunk)
        transformed_chunk = registration.apply_transformation(chunk_array, transformation_matrix_Fd[0])
        transformed_translated_points.append(transformed_chunk)
    """
    #apply Fd to H
    for i in range(12):
        chunk_array = np.vstack(optpivot_opt_frames[i])
        transformed_chunk = registration.apply_transformation(chunk_array, transformation_matrix_Fd[i])
        H_prime.append(transformed_chunk)

    # fix gj as the original starting positions
    # source_points = translated_points[0]
    source_points = translated_points[0]
    # source_points = registration.apply_transformation(np.vstack(translated_points[0]), transformation_matrix_Fd[0])
    for i in range(12):
        target_points = H_prime[i]
        transformation_matrix = registration.calculate_3d_transformation(source_points, target_points)
        trans_matrix_f.append(transformation_matrix)

    p_tip, p_pivot = registration.pivot_calibration(trans_matrix_f)
    print(p_pivot)
    