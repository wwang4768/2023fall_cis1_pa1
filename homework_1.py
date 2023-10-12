import numpy as np
from calibration_library import *
from dataParsing_library import *
import copy
import os

def main(): 
    # Read in input dataset
    script_directory = os.path.dirname(__file__)
    choose_set = 'a'
    base_path = os.path.join(script_directory, 'pa1_student_data\\PA1 Student Data\\pa1-debug-')

    calbody = base_path + choose_set + '-calbody.txt'
    calbody_point_cloud = parseData(calbody)
    d0, a0, c0 = parseCalbody(calbody_point_cloud)

    calreading = base_path + choose_set + '-calreadings.txt'
    calreading_point_cloud = parseData(calreading)
    calreading_frames = parseFrame(calreading_point_cloud, 8+8+27) # 8 optical markers on calibration object and 27 EM markers on calibration object
    
    empivot = base_path + choose_set + '-empivot.txt'
    empivot_point_cloud = parseData(empivot)
    empivot_frames = parseFrame(empivot_point_cloud, 6) # stores the list of 12 frames, each of which contains data of 6 EM markers on probe 
    
    optpivot = base_path + choose_set + '-optpivot.txt'
    optpivot_point_cloud = parseData(optpivot) # stores the list of 12 frames, each of which contains data of 8 optical markers on EM base & 6 EM markers on probe
    optpivot_em_frames, optpivot_opt_frames = parseOptpivot(optpivot_point_cloud, 8, 6) 
    

    registration = setRegistration()
    np.set_printoptions(formatter={'float': '{:.2f}'.format})

    # Q4
    # Part A 
    source_points_d = d0
    trans_matrix_d = []
    target_points = []

    for i in range(8):
        target_points = calreading_frames[i][:8]
        transformation_matrix = registration.calculate_3d_transformation(source_points_d, target_points)
        trans_matrix_d.append(transformation_matrix)

    # Part B
    source_points_a = a0
    trans_matrix_a = []
    target_points = []

    for i in range(8):
        target_points = calreading_frames[i][8:16]
        transformation_matrix = registration.calculate_3d_transformation(source_points_a, target_points)
        trans_matrix_a.append(transformation_matrix)
    
    # Part C
    source_points_c = c0
    transformation_matrix = []
    transformed_point = []
    for i in range(8):
        transformation_matrix = np.dot(np.linalg.inv(trans_matrix_d[i]), trans_matrix_a[i])
        transformed_point.append(registration.apply_transformation(source_points_c, transformation_matrix))
    
    # Part D
    # print(transformed_point)

    # Q5
    # Initalize the set for gj = Gj - G0
    translated_points_Gj = copy.deepcopy(empivot_frames)
    # Find centroid of Gj (the original position of 6 EM markers on the probe)
    midpoint = np.mean(empivot_frames, axis=1)
    trans_matrix_FG = []

    for i in range(12):
        for j in range(6):
            p = empivot_frames[i][j] - midpoint[i]
            translated_points_Gj[i][j] = p
    
    # fix gj as the original starting positions
    source_points = translated_points_Gj[0]
    for i in range(12):
        target_points = empivot_frames[i]
        transformation_matrix = registration.calculate_3d_transformation(source_points, target_points)
        trans_matrix_FG.append(transformation_matrix)
    p_tip, p_pivot = registration.pivot_calibration(trans_matrix_FG)
    print(p_pivot)

    # Q6
    # Initalize the set for gj = Gj - G0
    translated_points = copy.deepcopy(optpivot_opt_frames)
    H_prime = []
    # Find centroid of Gj (the original position of 6 EM markers on the probe)
    midpoint = np.mean(optpivot_opt_frames, axis=1)
    # Find transformation matrix for 12 frames
    trans_matrix_f = []

    # Calculate Fd
    source_points_d = d0 #optpivot_em_frames[0]
    transformation_matrix_Fd = []
    target_points = []

    for i in range(12):
        target_points = optpivot_em_frames[i]
        transformation_matrix = registration.calculate_3d_transformation(source_points_d, target_points)
        transformation_matrix = np.linalg.inv(transformation_matrix)
        transformation_matrix_Fd.append(transformation_matrix)

    for i in range(12):
        for j in range(6):
        # fill out gj 
            p = optpivot_opt_frames[i][j] - midpoint[i]
            translated_points[i][j] = p
    
    #apply Fd to H
    for i in range(12):
        chunk_array = np.vstack(optpivot_opt_frames[i])
        transformed_chunk = registration.apply_transformation(chunk_array, transformation_matrix_Fd[i])
        H_prime.append(transformed_chunk)

    # fix gj as the original starting positions
    source_points = translated_points[0]

    for i in range(12):
        target_points = H_prime[i]
        transformation_matrix = registration.calculate_3d_transformation(source_points, target_points)
        trans_matrix_f.append(transformation_matrix)
    p_tip, p_pivot = registration.pivot_calibration(trans_matrix_f)
    print(p_pivot)

if __name__ == "__main__":
    main()