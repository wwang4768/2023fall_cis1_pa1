import numpy as np

def parseData(input_file):
    points = [] 

    with open(input_file, 'r') as file:
        # skip the first line
        next(file)

        for line in file:
            x, y, z = map(float, line.strip().split(','))
            point = (x, y, z)
            points.append(point)

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

    chunk_size_d = len_chunk_d
    chunk_size_h = len_chunk_h

    current_list = 'D'
    temp = []

    for p in point_cloud:
        temp.append(p)
        
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