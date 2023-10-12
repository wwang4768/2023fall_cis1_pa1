import unittest
import numpy 
from calibration_library import *
from dataParsing_library import *

class validate(unittest.TestCase):
    # Test registration
    def setUp(self):
        self.source_points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.target_points = np.array([[2, 3, 4], [5, 6, 7], [8, 9, 10]])

        self.set_registration = setRegistration()

    def test_calculate_3d_transformation(self):
        transformation_matrix = self.set_registration.calculate_3d_transformation(self.source_points, self.target_points)
        self.assertEqual(transformation_matrix.shape, (4, 4))

    def test_compute_error(self):
        transformation_matrix = self.set_registration.calculate_3d_transformation(self.source_points, self.target_points)
        error = self.set_registration.compute_error(self.source_points, self.target_points, transformation_matrix)
        self.assertLess(error, 1.0)

    def test_apply_transformation(self):
        transformation_matrix = self.set_registration.calculate_3d_transformation(self.source_points, self.target_points)
        transformed_points = self.set_registration.apply_transformation(self.source_points, transformation_matrix)
        self.assertEqual(transformed_points.shape, self.source_points.shape)

    # Test parsing
    def test_parseData(self):
        input_file = 'pa1_student_data\PA1 Student Data\pa1-debug-a-calreadings.txt'
        point_cloud = parseData(input_file)
        self.assertIsInstance(point_cloud, np.ndarray)
        self.assertEqual(point_cloud.shape[1], 3)
        self.assertGreater(point_cloud.shape[0], 0)

    def test_parseCalbody(self):
        input_file = 'pa1_student_data\PA1 Student Data\pa1-debug-a-calbody.txt'
        point_cloud = parseData(input_file)
        d, a, c = parseCalbody(point_cloud)
        self.assertEqual(len(d), 8)
        self.assertEqual(len(a), 8)
        self.assertEqual(len(c), 27)

    def test_parseOptpivot(self):
        input_file = 'pa1_student_data\PA1 Student Data\pa1-debug-a-optpivot.txt'
        point_cloud = parseData(input_file)
        frames_d, frames_h = parseOptpivot(point_cloud, 8, 6)
        self.assertTrue(len(frames_d) > 0)
        self.assertTrue(len(frames_h) > 0)

    def test_parseFrame(self):
        test_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        frame_chunk = 4
        frames = parseFrame(test_data, frame_chunk)
        self.assertEqual(len(frames), len(test_data) // frame_chunk)

    def calculate_error_from_sample(self, file1, file2, use_reference=0):
        with open(file1, 'r') as f1:
            next(f1)
            data1 = [list(map(float, line.strip().split(','))) for line in f1]

        with open(file2, 'r') as f2:
            next(f2)
            data2 = [list(map(float, line.strip().split(','))) for line in f2]

        reference_data = data1 if use_reference == 0 else data2

        percentage_differences = []
        for row1, row2 in zip(data1, data2):
            percentage_diff_row = []
            for val1, val2 in zip(row1, row2):
                percentage_diff = ((val2 - val1) / val1)
                percentage_diff_row.append(percentage_diff)
            percentage_differences.append(percentage_diff_row)

        return percentage_differences

if __name__ == '__main__':
    unittest.main()
    