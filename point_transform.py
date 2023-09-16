import os
import time
import math
import numpy as np


class Quaternion:
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def normalize(self):
        magnitude = math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        self.w /= magnitude
        self.x /= magnitude
        self.y /= magnitude
        self.z /= magnitude

    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def __mul__(self, other):
        w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
        x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
        y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
        z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
        return Quaternion(w, x, y, z)
    
    def get_coordinates(self):
        return self.x, self.y, self.z
    
    @classmethod
    def from_rotation_vector(cls, axis, angle):
        half_angle = math.radians(angle) / 2
        sin_half_angle = math.sin(half_angle)
        return cls(math.cos(half_angle), axis[0] * sin_half_angle, axis[1] * sin_half_angle, axis[2] * sin_half_angle)


def rotate_point(point, rotation):
    """Rotate a point around the origin"""
    if hasattr(point, "x"):
        point = np.array((point.x, point.y, point.z))
    else:
        try:
            point = np.array(point)
        except Exception as e:
            raise Exception(f"\nUnable to convert received value for arg point into numpy array.\ntype(point)=={type(point)}\nexception=={e}\n")
    q_point = Quaternion(0, point[0], point[1], point[2])
    q_conjugate = rotation.conjugate()
    rotated_point = (rotation * q_point * q_conjugate).get_coordinates()
    return rotated_point


def rotate_points_with_vectors(points, from_vector, to_vector=(0, 1, 0), allow_antiparallel=False, antiparallels_rotation_axis=np.array([1, 0, 0])):
    """Rotate the coordinate system from the from_vector to the to_vector"""
    # Turn received floor_normal into a numpy array
    if hasattr(from_vector, "x"):
        from_vector = np.array((from_vector.x, from_vector.y, from_vector.z))
    else:
        try:
            from_vector = np.array(from_vector)
        except Exception as e:
            raise Exception(f"\nUnable to convert received value for arg from_vector into numpy array.\ntype(from_vector)=={type(from_vector)}\nexception=={e}\n")
        
    # Normalize the from_vector vector
    from_vector = from_vector / np.linalg.norm(from_vector)
    # Calculate axis and angle of rotation
    dot = np.dot(from_vector, to_vector)
    if np.isclose(dot, 1.0):
        # from_vector and to_vector vectors are parallel. No rotation needed.
        return points
        # # handle this special case by rotating around an arbitrary axis
        # axis = np.array([1, 0, 0])  # choose an arbitrary axis, e.g., [1, 0, 0]
        # angle = 0.0  # no rotation needed
    elif np.isclose(dot, -1.0):
        # from_vector and to_vector vectors are anti-parallel
        # handle this special case by rotating around the perpendicular axis to from_vector
        if not allow_antiparallel:
            raise Exception(f"\nfrom_vector and to_vector vectors are anti-parallel. \"Infinite\" number of rotation axis to choose from and "
                            f"each one would give a different result. Set arg allow_antiparallel=True to allow it anyways and optionally "
                            f"set arg antiparallels_rotation_axis."
                            f"\nfrom_vector=={from_vector}  (normalized)"
                            f"\nto_vector=={to_vector}  (normalized)\n"
                            )
        if hasattr(antiparallels_rotation_axis, "x"):
            antiparallels_rotation_axis = np.array((antiparallels_rotation_axis.x, antiparallels_rotation_axis.y, antiparallels_rotation_axis.z))
        else:
            try:
                antiparallels_rotation_axis = np.array(antiparallels_rotation_axis)
                assert len(antiparallels_rotation_axis) >= 3, f"len(antiparallels_rotation_axis) must be at least 3 but it's just of len {len(antiparallels_rotation_axis)}"
            except Exception as e:
                raise Exception(f"\nUnable to convert received value for arg antiparallels_rotation_axis into numpy array."
                                f"\ntype(antiparallels_rotation_axis)=={type(antiparallels_rotation_axis)}"
                                f"\nexception=={e}\n")
        axis = np.cross(from_vector, np.array([1, 0, 0]))  # choose an arbitrary perpendicular axis
        axis /= np.linalg.norm(axis)  # normalize the axis vector
        angle = 180.0  # rotate by 180 degrees
    else:
        # regular case, calculate axis and angle of rotation as before
        axis = np.cross(from_vector, to_vector)
        axis /= np.linalg.norm(axis)
        angle = np.degrees(np.arccos(dot))

    # Normalize the axis vector
    axis_magnitude = math.sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2)
    axis = (axis[0] / axis_magnitude, axis[1] / axis_magnitude, axis[2] / axis_magnitude)

    # Calculate quaternion for rotation
    floor_quat = Quaternion.from_rotation_vector(axis, angle)
    floor_quat.normalize()

    # Rotate the points
    rotated_points = [rotate_point(p, floor_quat) for p in points]
    
    return rotated_points


def test_rotate_points_with_vectors():
    # Test case 1
    point = (1, 1, 1)
    # floor_plane = (0, 0.99999, 0.00001)
    floor_plane = (0, 1, 0)
    up = (0, 1, 0)
    expected_rotated_point = (1, 1, 1)
    rotated_point = rotate_points_with_vectors(point, floor_plane, up)
    print("Test case 1 - parallel vectors:")
    print("Point:", point)
    print("Floor plane normal:", floor_plane)
    print("Up vector:", up)
    print("Expected rotated point:", expected_rotated_point)
    print("Rotated point:", rotated_point)
    print()

    # Test case 1.5
    point = (1, 1, 1)
    floor_plane = (0, -1, 0)
    up = (0, 1, 0)
    expected_rotated_point = (-1, -1, 1)
    rotated_point = rotate_points_with_vectors(point, floor_plane, up, allow_antiparallel=True, antiparallels_rotation_axis=np.array([1, 0, 0]))
    print("Test case 1.5 - anti-parallel vectors:")
    print("Point:", point)
    print("Floor plane normal:", floor_plane)
    print("Up vector:", up)
    print("Expected rotated point:", expected_rotated_point)
    print("Rotated point:", rotated_point)
    print()

    # Test case 2
    point = (-1, 2, -3)
    floor_plane = (0, 0, 1)
    up = (0, 1, 0)
    expected_rotated_point = (-1, -3, -2)
    rotated_point = rotate_points_with_vectors(point, floor_plane, up)
    print("Test case 2:")
    print("Point:", point)
    print("Floor plane normal:", floor_plane)
    print("Up vector:", up)
    print("Expected rotated point:", expected_rotated_point)
    print("Rotated point:", rotated_point)
    print()

    # Test case 3
    point = (-1, -1, -1)
    floor_plane = (1.0, 1.0, 1.0)
    up = (0, 1, 0)
    expected_rotated_point = (0, -math.sqrt(3), 0)
    rotated_point = rotate_points_with_vectors(point, floor_plane, up)
    print("Test case 3:")
    print("Point:", point)
    print("Floor plane normal:", floor_plane)
    print("Up vector:", up)
    print("Expected rotated point:", expected_rotated_point)
    print("Rotated point:", rotated_point)
    print()

    # Test case 4
    point = (123, 456, 789)
    floor_plane = (1.0, 1.0, 1.0)
    up = (0, 1, 0)
    expected_rotated_point = ("?", "?", "?")
    rotated_point = rotate_points_with_vectors(point, floor_plane, up)
    print("Test case 4:")
    print("Point:", point)
    print("Floor plane normal:", floor_plane)
    print("Up vector:", up)
    print("Expected rotated point:", expected_rotated_point)
    print("Rotated point:", rotated_point)
    print()


def testy1():
    # Example usage
    point = (1, 2, 3)  # Point to rotate
    angle = math.pi / 4  # Rotation angle in radians
    axis = (0, 0, 1)  # Rotation axis (unit vector)

    # Calculate quaternion representing the rotation
    half_angle = angle / 2
    w = math.cos(half_angle)
    x = axis[0] * math.sin(half_angle)
    y = axis[1] * math.sin(half_angle)
    z = axis[2] * math.sin(half_angle)
    rotation_quaternion = Quaternion(w, x, y, z)
    rotation_quaternion.normalize()

    # Rotate the point
    rotated_point = rotate_point(point, rotation_quaternion)
    print("Original Point:", point)
    print("Rotated Point:", rotated_point)



if __name__ == "__main__":
    # testy1()
    test_align_point_with_floor_plane()

if __name__ == "__main__":
    print(f"\n\n    Finished Script '{os.path.basename(__file__)}' at {time.strftime('%Y-%m-%d_%H-%M-%S')}    \n\n")
