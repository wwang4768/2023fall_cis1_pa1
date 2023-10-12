import numpy as np

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