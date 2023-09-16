import os
import time
import math

import numpy as np
import cv2
from pykinect2 import PyKinectV2, PyKinectRuntime

from show_kinect_data import show_camera
import point_transform


kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body)


class KinectDetector:
    """Detects gestures using the Kinect and returns the corresponding action"""
    def __init__(self):
        self.kinect = kinect
        self.body = None
        self.floor_plane = None

        # Timestamp of last gesture triggers (can be reseted to allow repeat gestures quickly)
        self.t_last_consume = 0.0
        self.t_last_jump = 0.0
        self.t_left_foot_active = 0.0
        self.t_right_foot_active = 0.0
        self.t_start_run = 0.0
        self.t_last_grab_open_state = 0.0
        self.t_last_grab_closed_state = 0.0
        self.t_last_light_attack_shoulder_state = 0.0
        self.t_last_light_attack_hip_state = 0.0
        self.t_last_parry_start = 0.0

        # Gesture readyness flags - require reset of gesture before being able to trigger again to prevent unwanted triggers
        self.roll_gesture_ready = True
        self.lock_on_gesture_ready = True
        self.kick_gesture_ready = True


    def get_actions(self, update_body_frame=True):
        """Returns the actions that are currently detected"""
        if update_body_frame:
            self.update_body_frame()
        actions = {}
        actions["lean"] = self.get_lean()
        actions["camera_move"] = self.get_camera_move()
        actions["is_consuming_item"] = self.is_consuming_item()
        actions["triggered_jump"] = self.triggered_jump()
        actions["triggered_run"] = self.triggered_run()
        actions["triggered_roll"] = self.triggered_roll()
        actions["triggered_interaction"] = self.triggered_interaction()
        actions["triggered_light_attack"] = self.triggered_light_attack()
        actions["triggered_block"] = self.triggered_block()
        actions["triggered_parry"] = self.triggered_parry()
        actions["triggered_lock_on"] = self.triggered_lock_on()
        actions["triggered_kick"] = self.triggered_kick()
        # not implemented yet (NIY)
        actions["triggered_heavy_attack"] = self.triggered_heavy_attack()  # NIY

        return actions
    

    def update_body_frame(self):
        """Updates the body_frame, body and the floor_plane"""
        # Get body frame
        if not self.kinect.has_new_body_frame():
            return False
        body_frame = self.kinect.get_last_body_frame()

        # Get tracked body
        tracked_bodies = [body for body in body_frame.bodies if body.is_tracked]
        if len(tracked_bodies) == 0:
            return False
        self.body = tracked_bodies[0]

        # Get floor plane
        self.floor_plane = body_frame.floor_clip_plane

        return True
    
    
    def get_rel_joint_distances(self, joint1, joint2, with_direct_distance=True):
        """Return the relative offsets between two joints in the x, y and z direction and the whole distance"""
        # Get xyz-coordinates of joint1 (depending on the type of joint)
        if hasattr(joint1, "x"):
            joint1_x, joint1_y, joint1_z = joint1.x, joint1.y, joint1.z
        elif hasattr(joint1, "Position"):
            joint1_x, joint1_y, joint1_z = joint1.Position.x, joint1.Position.y, joint1.Position.z
        elif isinstance(joint1, (list, tuple, np.ndarray)):
            assert len(joint1) == 3, f"Joint must be a list, tuple or np.ndarray of length 3, but is of len {len(joint1)}"
            joint1_x, joint1_y, joint1_z = joint1
        else:
            raise ValueError(f"Joint must be a list, tuple, np.ndarray or have attributes x, y and z, but is of type {type(joint1)}")
        
        # Get xyz-coordinates of joint2 (depending on the type of joint)
        if hasattr(joint2, "x"):
            joint2_x, joint2_y, joint2_z = joint2.x, joint2.y, joint2.z
        elif hasattr(joint2, "Position"):
            joint2_x, joint2_y, joint2_z = joint2.Position.x, joint2.Position.y, joint2.Position.z
        elif isinstance(joint2, (list, tuple, np.ndarray)):
            assert len(joint2) == 3, f"Joint must be a list, tuple or np.ndarray of length 3, but is of len {len(joint2)}"
            joint2_x, joint2_y, joint2_z = joint2
        else:
            raise ValueError(f"Joint must be a list, tuple, np.ndarray or have attributes x, y and z, but is of type {type(joint2)}")
        
        rel_x, rel_y, rel_z = joint1_x - joint2_x, joint1_y - joint2_y, joint1_z - joint2_z

        if not with_direct_distance:
            return rel_x, rel_y, rel_z

        direct_distance = math.sqrt(rel_x * rel_x + rel_y * rel_y + rel_z * rel_z)
        return rel_x, rel_y, rel_z, direct_distance
    
    
    def get_joint_distance(self, joint1, joint2):
        """Return the direct distance between two joints"""
        distances = self.get_rel_joint_distances(joint1, joint2)
        if distances is None:
            return None
        return distances[-1]
    
    
    def check_distances(self, joint1, joint2, min_x=None, max_x=None, min_y=None, max_y=None, min_z=None, max_z=None, min_dist=None, max_dist=None):
        """Check if the relative distances between two joints are within the given bounds"""
        rel_x, rel_y, rel_z, distance = self.get_rel_joint_distances(joint1, joint2)
        # X
        if min_x is not None and rel_x < min_x:
            return False
        if max_x is not None and rel_x > max_x:
            return False
        # Y
        if min_y is not None and rel_y < min_y:
            return False
        if max_y is not None and rel_y > max_y:
            return False
        # Z
        if min_z is not None and rel_z < min_z:
            return False
        if max_z is not None and rel_z > max_z:
            return False
        # Distance
        if min_dist is not None and distance < min_dist:
            return False
        if max_dist is not None and distance > max_dist:
            return False
        
        return True  # Passed all checks
    

    def check_distances_log(self, joint1, joint2, log_name:str,
                            min_x=None, max_x=None, min_y=None, max_y=None,
                            min_z=None, max_z=None, min_dist=None, max_dist=None):
        """Check if the relative distances between two joints are within the given bounds"""
        passed_all_checks = True
        rel_x, rel_y, rel_z, distance = self.get_rel_joint_distances(joint1, joint2)
        print(f"[{log_name}] Only first 3 decimal points in log")
        # X
        if min_x is not None:
            print(f"[{log_name}] rel_x({rel_x:.3f}) > min_x({min_x:.3f}) == {rel_x > min_x}")
            if not rel_x > min_x: passed_all_checks = False
        if max_x is not None:
            print(f"[{log_name}] rel_x({rel_x:.3f}) < max_x({max_x:.3f}) == {rel_x < max_x}")
            if not rel_x < max_x: passed_all_checks = False
        # Y
        if min_y is not None:
            print(f"[{log_name}] rel_y({rel_y:.3f}) > min_y({min_y:.3f}) == {rel_y > min_y}")
            if not rel_y > min_y: passed_all_checks = False
        if max_y is not None:
            print(f"[{log_name}] rel_y({rel_y:.3f}) < max_y({max_y:.3f}) == {rel_y < max_y}")
            if not rel_y < max_y: passed_all_checks = False
        # Z
        if min_z is not None:
            print(f"[{log_name}] rel_z({rel_z:.3f}) > min_z({min_z:.3f}) == {rel_z > min_z}")
            if not rel_z > min_z: passed_all_checks = False
        if max_z is not None:
            print(f"[{log_name}] rel_z({rel_z:.3f}) < max_z({max_z:.3f}) == {rel_z < max_z}")
            if not rel_z < max_z: passed_all_checks = False
        # Distance
        if min_dist is not None:
            print(f"[{log_name}] distance({distance:.3f}) > min_dist({min_dist:.3f}) == {distance > min_dist}")
            if not distance > min_dist: passed_all_checks = False
        if max_dist is not None:
            print(f"[{log_name}] distance({distance:.3f}) < max_dist({max_dist:.3f}) == {distance < max_dist}")
            if not distance < max_dist: passed_all_checks = False
        print(f"[{log_name}] passed_all_checks == {passed_all_checks}")
        
        return passed_all_checks

    
    def get_floor_distance(self, joint):
        """Return the shortest distance between a given joint and the floor plane"""
        if self.floor_plane is None:
            return None
        # Get xyz-coordinates of joint (depending on the type of joint)
        if hasattr(joint, "x"):
            joint_x, joint_y, joint_z = joint.x, joint.y, joint.z
        elif hasattr(joint, "Position"):
            joint_x, joint_y, joint_z = joint.Position.x, joint.Position.y, joint.Position.z
        elif isinstance(joint, (list, tuple, np.ndarray)):
            assert len(joint) == 3, f"Joint must be a list, tuple or np.ndarray of length 3, but is of len {len(joint)}"
            joint_x, joint_y, joint_z = joint
        else:
            raise ValueError(f"Joint must be a list, tuple, np.ndarray or have attributes x, y and z, but is of type {type(joint)}")
        
        # Get floor plane
        floor_x, floor_y, floor_z, floor_w = self.floor_plane.x, self.floor_plane.y, self.floor_plane.z, self.floor_plane.w

        # Calculate the distance between the joint and the floor plane
        numerator = floor_x * joint_x + floor_y * joint_y + floor_z * joint_z + floor_w
        denominator = math.sqrt(floor_x * floor_x + floor_y * floor_y + floor_z * floor_z)
        distance_to_floor = numerator / denominator

        return distance_to_floor
    

    @staticmethod
    def single_position_to_numpy(point):
        """Convert a single point to a numpy array"""
        if isinstance(point, np.ndarray):
            return point
        elif isinstance(point, (list, tuple)):
            return np.array(point)
        elif hasattr(point, "x"):
            return np.array([point.x, point.y, point.z])
        elif hasattr(point, "Position"):
            return np.array([point.Position.x, point.Position.y, point.Position.z])
        else:
            raise ValueError(f"Point must be a list, tuple, np.ndarray or have attributes x, y and z, but is of type {type(point)}")
        

    @classmethod
    def positions_to_numpy(cls, points):
        """Convert a multiple points to a numpy array"""
        new_points = []
        for p in points:
            new_points.append(cls.single_position_to_numpy(p))
        new_points = np.array(new_points)
        return new_points
    

    @staticmethod
    def move_origin(points, new_origin):
        """Move the origin of the given points to the given new origin"""
        new_points = []
        new_origin = np.array(new_origin)
        for p in points:
            p = np.array(p)
            new_points.append(p - new_origin)
        return new_points


    def align_points_with_floor(self, points):
        """Align the given points with the floor such that the x and z axis are parallel to the floor plane"""
        return point_transform.rotate_points_with_vectors(points, from_vector=self.floor_plane, to_vector=(0, 1, 0))


    def align_points_with_upper_body(self, points):
        """
            Idea is to compare joint positions in a coordinate system relative to the upper-body's rotation and lean.
            Make the SpineShoulder joint the new origin and align the new coordinate axis with the upper-body's rotation and lean.
            X-Axis: From ShoulderLeft to ShoulderRight. Increasing to the users right.
            Y-Axis: From SpineMid to SpineShoulder. Increasing to the users right.
            Z-Axis: Ortogonal  to the X and Y axis (cross-product of y_vector x x_vector = z_vector). Increasing to the users front.
            We act as if the new x and y are always orthogonal to each other, even though they are not.
        """
        if self.body is None:
            return points
        # Move origin to SpineShoulder
        new_origin_joint = self.body.joints[PyKinectV2.JointType_SpineMid].Position
        new_origin = np.array([new_origin_joint.x, new_origin_joint.y, new_origin_joint.z])
        aligned_points = self.positions_to_numpy(points)
        aligned_points = self.move_origin(aligned_points, new_origin)

        # Get joint positions as np arrays
        shoulder_left = self.body.joints[PyKinectV2.JointType_ShoulderLeft].Position  # x-axis vector tail position
        shoulder_left = np.array([shoulder_left.x, shoulder_left.y, shoulder_left.z])
        shoulder_right = self.body.joints[PyKinectV2.JointType_ShoulderRight].Position  # x-axis vector head position
        shoulder_right = np.array([shoulder_right.x, shoulder_right.y, shoulder_right.z])
        spine_mid = self.body.joints[PyKinectV2.JointType_SpineMid].Position  # y-axis vector tail position
        spine_mid = np.array([spine_mid.x, spine_mid.y, spine_mid.z])
        spine_shoulder = self.body.joints[PyKinectV2.JointType_SpineShoulder].Position  # y-axis vector head position
        spine_shoulder = np.array([spine_shoulder.x, spine_shoulder.y, spine_shoulder.z])
        # Move origins to SpineShoulder
        shoulder_left -= new_origin
        shoulder_right -= new_origin
        spine_mid -= new_origin
        spine_shoulder -= new_origin
        
        # Align y-axis
        # Make y-axis go through the SpineMid and SpineShoulder joints
        y_axis_vector = np.array(spine_shoulder) - np.array(spine_mid)  # Vector from left to right shoulder
        aligned_points = point_transform.rotate_points_with_vectors(points, from_vector=y_axis_vector, to_vector=np.array([0, 1, 0]))
        shoulder_right, shoulder_left = point_transform.rotate_points_with_vectors(np.array([shoulder_right, shoulder_left]), from_vector=y_axis_vector, to_vector=np.array([0, 1, 0]))

        # Align x-axis
        # Make x-axis go through the left and right shoulder joints
        x_axis_vector = np.array(shoulder_right) - np.array(shoulder_left)  # Vector from left to right shoulder
        aligned_points = point_transform.rotate_points_with_vectors(points, from_vector=x_axis_vector, to_vector=np.array([1, 0, 0]))

        # Print the new axis directions in the original coordinate system 
        print()
        new_x_axis = x_axis_vector / np.linalg.norm(x_axis_vector)
        print(f"new_x_axis:          x : {new_x_axis[0]:.3f}  |  y : {new_x_axis[1]:.3f}  |  z : {new_x_axis[2]:.3f}")
        original_spine_mid = self.body.joints[PyKinectV2.JointType_SpineMid].Position  # y-axis vector tail position
        original_spine_mid = np.array([original_spine_mid.x, original_spine_mid.y, original_spine_mid.z])
        original_spine_shoulder = self.body.joints[PyKinectV2.JointType_SpineShoulder].Position  # y-axis vector head position
        original_spine_shoulder = np.array([original_spine_shoulder.x, original_spine_shoulder.y, original_spine_shoulder.z])
        new_y_axis = original_spine_shoulder - original_spine_mid
        new_y_axis = new_y_axis / np.linalg.norm(new_y_axis)
        print(f"new_y_axis:          x : {new_y_axis[0]:.3f}  |  y : {new_y_axis[1]:.3f}  |  z : {new_y_axis[2]:.3f}")
        new_z_axis = np.cross(new_y_axis, new_x_axis)
        new_z_axis = new_z_axis / np.linalg.norm(np.cross(new_y_axis, new_x_axis))
        print(f"new_z_axis:          x : {new_z_axis[0]:.3f}  |  y : {new_z_axis[1]:.3f}  |  z : {new_z_axis[2]:.3f}")

        return aligned_points

    
    def get_lean(self):
        """
            Gesture detection for player movement:
            Returns the lean of the body in xy-coordinates
        """
        if self.body is None:
            return (0.0, 0.0)
        return (self.body.lean.x, self.body.lean.y)
    

    def get_camera_move(self):
        if self.body is None:
            return (0.0, 0.0)

        # Get the position of the left hand and wrist joints
        left_hand_direction = np.array([0.0, 0.0])
        if self.body.hand_left_state == PyKinectV2.HandState_Lasso:
            # Get the position of the hands and wrist joints
            left_hand_position = self.body.joints[PyKinectV2.JointType_HandLeft].Position
            left_wrist_position = self.body.joints[PyKinectV2.JointType_WristLeft].Position
            # Calculate the direction of the right hand
            left_hand_direction = np.array([
                left_hand_position.x - left_wrist_position.x,
                left_hand_position.y - left_wrist_position.y,
            ])
            # Scale to normal vector
            left_hand_direction = left_hand_direction / np.linalg.norm(left_hand_direction)

        # Get the position of the right hand and wrist joints
        right_hand_direction = np.array([0.0, 0.0])
        if self.body.hand_right_state == PyKinectV2.HandState_Lasso:
            # Get the position of the hands and wrist joints
            right_hand_position = self.body.joints[PyKinectV2.JointType_HandRight].Position
            right_wrist_position = self.body.joints[PyKinectV2.JointType_WristRight].Position
            # Calculate the direction of the right hand
            right_hand_direction = np.array([
                right_hand_position.x - right_wrist_position.x,
                right_hand_position.y - right_wrist_position.y,
            ])
            # Scale to normal vector
            right_hand_direction = right_hand_direction / np.linalg.norm(right_hand_direction)

        # Combine and normalize the hand directions
        hand_direction = left_hand_direction + right_hand_direction
        hand_direction_magnitude = np.linalg.norm(hand_direction)
        if hand_direction_magnitude > 0.0:
            hand_direction = hand_direction / hand_direction_magnitude

        return hand_direction
    
    
    def is_consuming_item(self) -> bool:
        """
            Gesture Detection for item consumption. Mainly for consuming an Estus Flask:
            Returns True if the player is drinking an Estus, False otherwise
        """
        if self.body is None:
            return False
        
        # Detection parameters
        t_last_consume_threshold = 1.2  # seconds
        reset_distance = 0.5
        hand_head_bounds = {
            "min_x" : -0.11, "max_x" : 0.15,
            "min_y" : -0.3, "max_y" : 0.0,
            "min_z" : -0.2, "max_z" : 0.0,
            "min_dist" : 0.01, "max_dist" : 0.35,
        }

        # Get joints
        right_hand_pos = self.body.joints[PyKinectV2.JointType_HandRight].Position
        head_pos = self.body.joints[PyKinectV2.JointType_Head].Position
        
        # Check parameters
        seconds_since_last_consume = time.time() - self.t_last_consume
        *_, head_hand_distance = self.get_rel_joint_distances(right_hand_pos, head_pos)
        if head_hand_distance > reset_distance:  # if right hand and head are far away again, look for consume gesture again
            self.t_last_consume = 0.0
            return False
        if self.body.hand_right_state != PyKinectV2.HandState_Closed:  # Check if right hand is closed
            return False
        if not self.check_distances(right_hand_pos, head_pos, **hand_head_bounds):
            return False
        if seconds_since_last_consume < t_last_consume_threshold:
            self.t_last_consume = time.time()
            return False
        
        self.t_last_consume = time.time()
        return True  # All checks for gesture detection passed
    
    
    def triggered_jump(self):
        """Gesture Detection for jumping"""
        if self.body is None:
            return False
        t_last_trigger_threshold = 0.5
        floor_distance_reset = 0.2
        left_foot, right_foot = self.body.joints[PyKinectV2.JointType_FootLeft], self.body.joints[PyKinectV2.JointType_FootRight]
        left_foot_height, right_foot_height = self.get_floor_distance(left_foot), self.get_floor_distance(right_foot)
        seconds_since_trigger = time.time() - self.t_last_jump

        # # Check if feet are off the floor
        if left_foot_height < floor_distance_reset or right_foot_height < floor_distance_reset:
            self.t_last_jump = 0.0
            return False
        if left_foot_height < 0.3 or right_foot_height < 0.3:
            return False
        if left_foot_height < 0.4 or right_foot_height < 0.4:
            return False
        if abs(left_foot_height - right_foot_height) > 0.3:
            return False
        if seconds_since_trigger < t_last_trigger_threshold:
            self.t_last_jump = time.time()
            return False
        self.t_last_jump = time.time()
        return True
    

    @staticmethod
    def feet_are_neutral(left_foot_height, right_foot_height) -> bool:
        if left_foot_height < 0.12 and right_foot_height < 0.12:
            return True
        else:
            return False
        

    @staticmethod
    def left_foot_active(left_foot_height, right_foot_height) -> bool:
        if left_foot_height > 0.20 and right_foot_height < 0.15:
            return True
        else:
            return False
        

    @staticmethod
    def right_foot_active(left_foot_height, right_foot_height) -> bool:
        if right_foot_height > 0.20 and left_foot_height < 0.15:
            return True
        else:
            return False
        

    def triggered_run(self):
        if self.body is None:
            return False
        min_run_duration = 0.6  # Duration the button has to be presseed to run and not roll
        delta_min = 0.7  # 0.7==slow marching, 0.6==Fast Marching, 0.5==Close to Jogging, 0.4 Jogging, 0.3 Running

        t_now = time.time()
        # Update the time the feet were last lifted/active
        left_foot, right_foot = self.body.joints[PyKinectV2.JointType_FootLeft], self.body.joints[PyKinectV2.JointType_FootRight]
        left_foot_height, right_foot_height = self.get_floor_distance(left_foot), self.get_floor_distance(right_foot)
        if self.left_foot_active(left_foot_height, right_foot_height):
            self.t_left_foot_active = t_now
        if self.right_foot_active(left_foot_height, right_foot_height):
            self.t_right_foot_active = t_now
        # Check if running
        feet_active_time_delta = abs(self.t_left_foot_active - self.t_right_foot_active)
        seconds_since_last_foot_active = t_now - max(self.t_left_foot_active, self.t_right_foot_active) 
        if feet_active_time_delta > delta_min:
            if t_now - self.t_start_run < min_run_duration:  # Make sure the run button is pressed long enough to trigger runnign and not rolling
                return True
            return False
        if seconds_since_last_foot_active > delta_min:
            if t_now - self.t_start_run < min_run_duration:  # Make sure the run button is pressed long enough to trigger runnign and not rolling
                return True
            return False
        self.t_start_run = t_now
        return True
    
    
    def triggered_roll(self):
        """
            Squating to trigger Gesture Detection for rolling. Returns True if squatting, False otherwise.
            Due to problems with false-positive when only True if all conditions are met, it now is voted
            on and only returns True if the portion of true detections is above the vote_true_threshold.
        """
        if self.body is None:
            return False
        vote_true_threshold = 0.66
        vote_results = []
        req_hip_floor_distance = 0.40  # minimal required closeness to floor to allow squat detection
        req_feet_hip_distance = 0.60  # minimal required closeness from feet to hips to allow squat detection
        or_below_sum_distance = 2.00  # Also trigger roll if the sum of distances is below this value

        # Get floor and joints
        left_foot = self.body.joints[PyKinectV2.JointType_FootLeft]
        right_foot = self.body.joints[PyKinectV2.JointType_FootRight]
        left_hip = self.body.joints[PyKinectV2.JointType_HipLeft]
        right_hip = self.body.joints[PyKinectV2.JointType_HipRight]

        # Log values
        left_hip_distance_to_floor = self.get_floor_distance(left_hip)
        right_hip_distance_to_floor = self.get_floor_distance(right_hip)
        left_foot_hip_distance = self.get_joint_distance(left_foot, left_hip)
        right_foot_hip_distance = self.get_joint_distance(right_foot, right_hip)

        # Check the roll/squat parameter conditions
        vote_results.append(left_hip_distance_to_floor > req_hip_floor_distance)
        vote_results.append(right_hip_distance_to_floor > req_hip_floor_distance)
        vote_results.append(left_foot_hip_distance > req_feet_hip_distance)
        vote_results.append(right_foot_hip_distance > req_feet_hip_distance)

        # Make decision
        true_votes = vote_results.count(True)
        # Gesture has not been reset/made ready yet. Return False.
        if not self.roll_gesture_ready:
            # Check if gesture can be reset to be triggered again
            if true_votes == 0:
                self.roll_gesture_ready = True
            return False
        # Calculate and Evaluate number of conditions that are met
        true_votes_portion = true_votes / len(vote_results)
        if true_votes_portion > vote_true_threshold:
            self.roll_gesture_ready = False
            return True
        # Trigger anyways if the sum of distances is low
        elif sum([left_hip_distance_to_floor, right_hip_distance_to_floor, left_foot_hip_distance, right_foot_hip_distance]) < or_below_sum_distance:
            self.roll_gesture_ready = False
            return True
        return False
    

    def triggered_interaction(self):
        """
            Track the right hand for a grab gesture. Grab is detected as the right hand
            held forwards changing from opened to closed.
        """
        if self.body is None:
            return False
        target_hand_shoulder_distance = 0.40

        # Get joints and states
        right_hand_position = self.body.joints[PyKinectV2.JointType_HandRight].Position
        right_shoulder_position = self.body.joints[PyKinectV2.JointType_ShoulderRight].Position
        right_hand_state = self.body.hand_right_state

        # Check if hand is held towards kinect
        hand_shoulder_z_distance = right_shoulder_position.z - right_hand_position.z
        if hand_shoulder_z_distance > target_hand_shoulder_distance:
            # Check if hand is closed
            if right_hand_state == PyKinectV2.HandState_Closed:
                # Check if hand was previously open
                if self.t_last_grab_open_state > self.t_last_grab_closed_state:
                    # Hand is closed and was previously open
                    self.t_last_grab_closed_state = time.time()
                    return True
                self.t_last_grab_closed_state = time.time()
            # Check if hand is open
            elif right_hand_state == PyKinectV2.HandState_Open:
                self.t_last_grab_open_state = time.time()
        # Hand is not held towards kinect or is not open or closed
        return False


    def triggered_light_attack(self):
        """
            Track the right hand for a light attack gesture. Gesture motion starts with the right hand above the
            right shoulder and ends with the right hand close to the left hip with little time inbetween.
        """
        if self.body is None:
            return False

        # Position parameters
        max_shoulder_to_hip_motion_duration = 0.4
        max_hip_to_shoulder_motion_duration = 0.33
        hand_shoulder_bounds = {
            "min_x" : -0.05, "max_x" : 0.20,
            "min_y" : 0.0, "max_y" : 0.35,
            "min_z" :  -0.30, "max_z" : 0.1,
            "min_dist" : 0.1, "max_dist" : 0.45,
        }
        hand_hip_bounds = {
            "min_x" : -0.2, "max_x" : 0.05,
            "min_y" : -0.15, "max_y" : 0.15,
            "min_z" :  -0.3, "max_z" : 0.0,
            "min_dist" : 0.1, "max_dist" : 0.3,
        }

        # Get joint and distances
        right_hand_position = self.body.joints[PyKinectV2.JointType_HandRight].Position
        right_shoulder_position = self.body.joints[PyKinectV2.JointType_ShoulderRight].Position
        left_hip_position = self.body.joints[PyKinectV2.JointType_HipLeft].Position

        # Check if closed right-hand above right-shoulder and handle it
        if self.body.hand_right_state == PyKinectV2.HandState_Closed:
            if self.check_distances(right_hand_position, right_shoulder_position, **hand_shoulder_bounds):
                self.t_last_light_attack_shoulder_state = time.time()  # Last time that the right hand was above the right shoulder is right now
                seconds_since_other_state = self.t_last_light_attack_shoulder_state - self.t_last_light_attack_hip_state

                if seconds_since_other_state < max_hip_to_shoulder_motion_duration:
                    self.t_last_light_attack_hip_state = 0.0  # To require other state to be detected again before triggering again
                    return True 
                else:
                    return False

        # Check if closed right-hand on left-hip and handle it
        if self.check_distances(right_hand_position, left_hip_position, **hand_hip_bounds):
            self.t_last_light_attack_hip_state = time.time()  # Last time that the right hand was on the left hip is right now
            seconds_since_other_state = self.t_last_light_attack_hip_state - self.t_last_light_attack_shoulder_state

            if seconds_since_other_state < max_shoulder_to_hip_motion_duration:
                self.t_last_light_attack_shoulder_state = 0.0  # To require other state to be detected again before triggering again
                return True
            else:
                return False

        return False


    def triggered_heavy_attack(self):
        """ 
            Look for stabbing movements.
            Right hand fist on right waist -> to right hand fist forwards
        """
        if self.body is None:
            return False
        
        # Get joint and distances
        right_hand_position = self.body.joints[PyKinectV2.JointType_HandRight].Position
        right_shoulder_position = self.body.joints[PyKinectV2.JointType_ShoulderRight].Position
        right_hip_position = self.body.joints[PyKinectV2.JointType_HipRight].Position
        hand_shoulder_x, hand_shoulder_y, hand_shoulder_z = right_hand_position.x - right_shoulder_position.x, right_hand_position.y - right_shoulder_position.y, right_hand_position.z - right_shoulder_position.z
        hand_shoulder_dist = math.sqrt(hand_shoulder_x**2 + hand_shoulder_y**2 + hand_shoulder_z**2)
        hand_hip_x, hand_hip_y, hand_hip_z = right_hand_position.x - right_hip_position.x, right_hand_position.y - right_hip_position.y, right_hand_position.z - right_hip_position.z
        hand_hip_dist = math.sqrt(hand_hip_x**2 + hand_hip_y**2 + hand_hip_z**2)

        return False


    def triggered_block(self):
        """
            Left hand in closed state in front of torso
        """
        if self.body is None:
            return False
        
        # distance parameters
        hand_spine_bounds = {
            "min_x" : -0.1, "max_x" : 0.1,
            "min_y" : -0.32, "max_y" : -0.03,
            "min_z" :  -0.35, "max_z" : 0.0,
            "min_dist" : 0.1, "max_dist" : 0.4,
        }
        
        # Get joint and distances
        spine_shoulder_pos = self.body.joints[PyKinectV2.JointType_SpineShoulder].Position
        left_hand_pos = self.body.joints[PyKinectV2.JointType_HandLeft].Position

        distances_in_bounds = self.check_distances(left_hand_pos, spine_shoulder_pos, **hand_spine_bounds)

        return distances_in_bounds


    def triggered_parry(self):
        """
            Left Hand on right hip -> to left hand above left shoulder
        """
        if self.body is None:
            return False
        
        # Position parameters
        target_start_end_duration = 0.6
        hand_hip_bounds = {
            "min_x" : 0.0, "max_x" : 0.15,
            "min_y" : -0.01, "max_y" : 0.12,
            "min_z" :  -0.2, "max_z" : -0.03,
            "min_dist" : None, "max_dist" : 0.2,
        }
        hand_shoulder_bounds = {
            "min_x" : -0.40, "max_x" : -0.05,
            "min_y" : 0.18, "max_y" : 0.35,
            "min_z" :  -0.21, "max_z" : 0.18,
            "min_dist" : 0.10, "max_dist" : 0.50,
        }

        # Get joint and distances
        left_hand_pos = self.body.joints[PyKinectV2.JointType_HandLeft].Position
        left_shoulder_pos = self.body.joints[PyKinectV2.JointType_ShoulderLeft].Position
        right_hip_pos = self.body.joints[PyKinectV2.JointType_HipRight].Position

        # Parry start (hand on hip)
        if self.check_distances(left_hand_pos, right_hip_pos, **hand_hip_bounds):
            self.t_last_parry_start = time.time()
            return False

        # Parry end (hand over shoulder)
        if self.body.hand_left_state == PyKinectV2.HandState_Closed:  # Hand closed
            seconds_since_parry_start = time.time() - self.t_last_parry_start
            if seconds_since_parry_start < target_start_end_duration:  # Parry start within time limit
                if not self.check_distances(left_hand_pos, left_shoulder_pos, **hand_shoulder_bounds):  # Hand over shoulder
                    self.t_last_parry_start = 0.0
                    return True

        return False


    def triggered_lock_on(self):
        """
            Both open hands stretched out forwards. Hands fairly close together.
            check_distances(self, joint1, joint2, min_x=None, max_x=None, min_y=None, max_y=None, min_z=None, max_z=None, min_dist=None, max_dist=None)
        """
        if self.body is None:
            return False
        
        # distance parameters
        hand_shoulder_bounds = {
            "min_x" : -0.15, "max_x" : 0.15,
            "min_y" : -0.15, "max_y" : 0.15,
            "min_z" :  None, "max_z" : -0.35,
        }
        hand_bounds = {
            "min_x" : -0.45, "max_x" : 0.0,
            "min_y" : -0.1, "max_y" : 0.1,
            "min_z" :  -0.1, "max_z" : 0.1,
            "min_dist" :  None, "max_dist" : 0.45,
        }
        
        # Get joint positions and hand states
        # Left side
        left_hand_position = self.body.joints[PyKinectV2.JointType_HandLeft].Position
        left_hand_open = self.body.hand_left_state == PyKinectV2.HandState_Open
        left_shoulder_position = self.body.joints[PyKinectV2.JointType_ShoulderLeft].Position
        # Right side
        right_hand_position = self.body.joints[PyKinectV2.JointType_HandRight].Position
        right_hand_open = self.body.hand_right_state == PyKinectV2.HandState_Open
        right_shoulder_position = self.body.joints[PyKinectV2.JointType_ShoulderRight].Position

        # Handle gesture reset
        if not self.lock_on_gesture_ready:
            hands_x, hands_y, hands_z, hands_dist = self.get_rel_joint_distances(left_hand_position, right_hand_position)
            if abs(hands_x) > 0.50:
                self.lock_on_gesture_ready = True
                return False
            if abs(hands_y) > 0.35:
                self.lock_on_gesture_ready = True
                return False
            if abs(hands_z) > 0.35:
                self.lock_on_gesture_ready = True
                return False
            if abs(hands_dist) > 0.60:
                self.lock_on_gesture_ready = True
                return False
            left_hand_shoulder_x, left_hand_shoulder_y, left_hand_shoulder_z, left_hand_shoulder_dist = self.get_rel_joint_distances(left_hand_position, left_shoulder_position)
            right_hand_shoulder_x, right_hand_shoulder_y, right_hand_shoulder_z, right_hand_shoulder_dist = self.get_rel_joint_distances(right_hand_position, right_shoulder_position)
            if abs(left_hand_shoulder_y) > 0.25 or abs(right_hand_shoulder_y) > 0.25:
                self.lock_on_gesture_ready = True
                return False
            if left_hand_shoulder_z > -0.25 or right_hand_shoulder_z > -0.25:
                self.lock_on_gesture_ready = True
                return False
            return False
            

        if not left_hand_open or not right_hand_open:
            return False

        # Check if all the criteria for the lock-on gesture are met
        # Left-Hand to Left-Shoulder
        if not self.check_distances(left_hand_position, left_shoulder_position, **hand_shoulder_bounds):
            return False
        # Right-Hand to Right-Shoulder
        if not self.check_distances(right_hand_position, right_shoulder_position, **hand_shoulder_bounds):
            return False
        # Left-Hand to Right-Hand
        if not self.check_distances(left_hand_position, right_hand_position, **hand_bounds):
            return False

        self.lock_on_gesture_ready = False
        return True  # All the criteria for the lock-on gesture are met
    

    def triggered_kick(self):
        """
            Right foot raised above floor and held towards kinect away from right
            hip with the left foot on the floor.
        """
        if self.body is None:
            return False
        
        right_foot_pos = self.body.joints[PyKinectV2.JointType_FootRight].Position
        left_foot_pos = self.body.joints[PyKinectV2.JointType_FootLeft].Position
        right_hip_pos = self.body.joints[PyKinectV2.JointType_HipRight].Position

        right_foot_floor_dist = self.get_floor_distance(right_foot_pos)
        left_foot_floor_dist = self.get_floor_distance(left_foot_pos)
        feet_x, feet_y, feet_z, feet_dist = self.get_rel_joint_distances(right_foot_pos, left_foot_pos)

        if not self.kick_gesture_ready:
            # Check if the right foot is raised above the floor
            if right_foot_floor_dist < 0.4:
                self.kick_gesture_ready = True
                return False
            # Check if the left foot is on the floor
            if feet_dist < 0.5:
                self.kick_gesture_ready = True
                return False
            return False

        # distance parameters
        right_foot_floor_dist_min = 0.6
        left_foot_floor_dist_max = 0.1
        feet_bounds = {
            # "min_z" : None, "max_z" : -0.4,
            # "min_dist" : 0.7, "max_dist" : None,
        }
        foot_hip_bounds = {
            "min_y" : -0.35, "max_y" : 0.35,
            "min_dist" :  0.5, "max_dist" : None,
        }

        # Check if all the criteria for the lock-on gesture are met
        if right_foot_floor_dist < right_foot_floor_dist_min:
            return False  # Kick foot too low
        if left_foot_floor_dist > left_foot_floor_dist_max:
            return False  # Stand foot in air
        
        if not self.check_distances(right_foot_pos, left_foot_pos, **feet_bounds):  # pointless?
            return False
        if not self.check_distances(right_foot_pos, right_hip_pos, **foot_hip_bounds):
            return False
        
        self.kick_gesture_ready = False
        return True  # All the criteria for the lock-on gesture are met

    
    def test_align_points_with_floor(self):
        """Test if we can convert our points into a stable coordinate system where the xz-plane forms a plane parallel to the floor."""
        if self.body is None:
            return False
        
        # Get original hand positions
        original_left_hand_position = self.body.joints[PyKinectV2.JointType_HandLeft].Position
        original_left_hand_position = original_left_hand_position.x, original_left_hand_position.y, original_left_hand_position.z
        original_right_hand_position = self.body.joints[PyKinectV2.JointType_HandRight].Position
        original_right_hand_position = original_right_hand_position.x, original_right_hand_position.y, original_right_hand_position.z
        original_points = [original_left_hand_position, original_right_hand_position]
        original_rel_x, original_rel_y, original_rel_z, original_rel_dist = self.get_rel_joint_distances(original_left_hand_position, original_right_hand_position)

        # Get rotated hand positions (hopefully aligned with the floor plane)
        rotated_points = self.align_points_with_floor(original_points)
        rotated_left_hand_position, rotated_right_hand_position = rotated_points
        rotated_rel_x, rotated_rel_y, rotated_rel_z, rotated_rel_dist = self.get_rel_joint_distances(rotated_left_hand_position, rotated_right_hand_position)

        # Print results 
        print()
        print("Rotating points (hand joints) to hopefully align coordinate system with floor plane")
        print()
        print(f"self.floor_plane:                  x : {self.floor_plane.x:.3f}  |  y : {self.floor_plane.y:.3f}  |  z : {self.floor_plane.z:.3f}  |  w : {self.floor_plane.w:.3f}")
        print()
        print(f"original right hand:               x : {original_right_hand_position[0]:.3f}  |  y : {original_right_hand_position[1]:.3f}  |  z : {original_right_hand_position[2]:.3f}")
        print(f"original left hand:                x : {original_left_hand_position[0]:.3f}  |  y : {original_left_hand_position[1]:.3f}  |  z : {original_left_hand_position[2]:.3f}")
        print(f"original distances left to right:  x : {original_rel_x:.3f}  |  y : {original_rel_y:.3f}  |  z : {original_rel_z:.3f}  |  dist : {original_rel_dist:.3f}")
        print()
        print(f"rotated right hand:                x : {rotated_right_hand_position[0]:.3f}  |  y : {rotated_right_hand_position[1]:.3f}  |  z : {rotated_right_hand_position[2]:.3f}")
        print(f"rotated left hand:                 x : {rotated_left_hand_position[0]:.3f}  |  y : {rotated_left_hand_position[1]:.3f}  |  z : {rotated_left_hand_position[2]:.3f}")
        print(f"rotated distances left to right:   x : {rotated_rel_x:.3f}  |  y : {rotated_rel_y:.3f}  |  z : {rotated_rel_z:.3f}  |  dist : {rotated_rel_dist:.3f}")
        print()  
        print("-" * 100)


    def test_align_points_with_upper_body(self):
        """Test if we can convert our points into a stable coordinate system where the xz-plane forms a plane parallel to the floor."""
        if self.body is None:
            return False
        
        # Get original hand positions
        original_right_shoulder_position = self.body.joints[PyKinectV2.JointType_ShoulderRight].Position
        original_right_shoulder_position = original_right_shoulder_position.x, original_right_shoulder_position.y, original_right_shoulder_position.z
        original_right_hand_position = self.body.joints[PyKinectV2.JointType_HandRight].Position
        original_right_hand_position = original_right_hand_position.x, original_right_hand_position.y, original_right_hand_position.z
        original_points = [original_right_shoulder_position, original_right_hand_position]
        original_rel_x, original_rel_y, original_rel_z, original_rel_dist = self.get_rel_joint_distances(original_right_hand_position, original_right_shoulder_position)

        # Get rotated hand positions (hopefully aligned with the floor plane)
        aligned_points = self.align_points_with_upper_body(original_points)
        aligned_right_shoulder_position, aligned_right_hand_position = aligned_points
        aligned_rel_x, aligned_rel_y, aligned_rel_z, aligned_rel_dist = self.get_rel_joint_distances(aligned_right_hand_position, aligned_right_shoulder_position)

        # Print results
        lean_x, lean_y = self.get_lean()
        from_original_to_aligned = np.array([aligned_rel_x-original_rel_x, aligned_rel_y-original_rel_y, aligned_rel_z-original_rel_z, aligned_rel_dist-original_rel_dist])
        print()
        print("test_align_points_with_upper_body (align with spine first and then with shoulders)")
        print()
        print(f"self.lean:                            x : {lean_x:.3f}  |  y : {lean_y:.3f}")
        print()
        print(f"original right hand:                  x : {original_right_hand_position[0]:.3f}  |  y : {original_right_hand_position[1]:.3f}  |  z : {original_right_hand_position[2]:.3f}")
        print(f"original right shoulder:              x : {original_right_shoulder_position[0]:.3f}  |  y : {original_right_shoulder_position[1]:.3f}  |  z : {original_right_shoulder_position[2]:.3f}")
        print(f"original distances shoulder to hand:  x : {original_rel_x:.3f}  |  y : {original_rel_y:.3f}  |  z : {original_rel_z:.3f}  |  dist : {original_rel_dist:.3f}")
        print()
        print(f"aligned right hand:                   x : {aligned_right_hand_position[0]:.3f}  |  y : {aligned_right_hand_position[1]:.3f}  |  z : {aligned_right_hand_position[2]:.3f}")
        print(f"aligned right shoulder:               x : {aligned_right_shoulder_position[0]:.3f}  |  y : {aligned_right_shoulder_position[1]:.3f}  |  z : {aligned_right_shoulder_position[2]:.3f}")
        print(f"aligned distances shoulder to hand:   x : {aligned_rel_x:.3f}  |  y : {aligned_rel_y:.3f}  |  z : {aligned_rel_z:.3f}  |  dist : {aligned_rel_dist:.3f}")
        print()
        print(f"from_original_to_aligned:             x : {from_original_to_aligned[0]:.3f}  |  y : {from_original_to_aligned[1]:.3f}  |  z : {from_original_to_aligned[2]:.3f}  |  dist : {from_original_to_aligned[3]:.3f}")
        print()
        print("-" * 100)

        


# ==================================================================================================



def testy1():
    import threading
    args = {}
    kwargs = {
        "display_type": "pygame",
        "single_skeleton_color": (120, 20, 120),
        # "window_size": (1280, 720),
        "window_size": (1024, 576),
        "show_fps": True,
    }
    cam_thread = threading.Thread(target=show_camera, args=args, kwargs=kwargs)
    cam_thread.start()
    kinect_detector = KinectDetector()
    while True:
        time.sleep(0.05)
        if not cam_thread.is_alive():
            exit()
        kinect_detector.get_actions(update_body_frame=True)


if __name__ == "__main__":
    testy1()

if __name__ == "__main__":
    print(f"\n\n    Finished Script '{os.path.basename(__file__)}' at {time.strftime('%Y-%m-%d_%H-%M-%S')}    \n\n")































































































# import os
# import time
# import math

# import numpy as np
# import cv2
# from pykinect2 import PyKinectV2, PyKinectRuntime

# from show_kinect_data import show_camera
# import point_transform


# kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body)


# class KinectDetector:
#     """Detects gestures using the Kinect and returns the corresponding action"""
#     def __init__(self):
#         # self.kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body)
#         self.kinect = kinect
#         self.body = None
#         self.floor_plane = None

#         # Timestamp of last gesture triggers (can be reseted to allow repeat gestures quickly)
#         self.t_last_consume = 0.0
#         self.t_last_jump = 0.0
#         self.t_left_foot_active = 0.0
#         self.t_right_foot_active = 0.0
#         self.t_start_run = 0.0
#         self.t_last_grab_open_state = 0.0
#         self.t_last_grab_closed_state = 0.0
#         # self.t_last_light_attack_start = 0.0
#         self.t_last_light_attack_shoulder_state = 0.0
#         self.t_last_light_attack_hip_state = 0.0
#         self.t_last_parry_start = 0.0

#         # Gesture readyness flags - require reset of gesture before being able to trigger again to prevent unwanted triggers
#         self.roll_gesture_ready = True
#         self.lock_on_gesture_ready = True
#         self.kick_gesture_ready = True


#     def get_actions(self, update_body_frame=True):
#         """Returns the actions that are currently detected"""
#         if update_body_frame:
#             self.update_body_frame()
#         actions = {}
#         actions["lean"] = self.get_lean()
#         actions["camera_move"] = self.get_camera_move()
#         actions["is_consuming_item"] = self.is_consuming_item()
#         actions["triggered_jump"] = self.triggered_jump()
#         actions["triggered_run"] = self.triggered_run()
#         actions["triggered_roll"] = self.triggered_roll()
#         actions["triggered_interaction"] = self.triggered_interaction()
#         actions["triggered_light_attack"] = self.triggered_light_attack()
#         actions["triggered_block"] = self.triggered_block()
#         actions["triggered_parry"] = self.triggered_parry()
#         actions["triggered_lock_on"] = self.triggered_lock_on()
#         actions["triggered_kick"] = self.triggered_kick()
#         # not implemented yet (NIY)
#         actions["triggered_heavy_attack"] = self.triggered_heavy_attack()  # NIY

#         # self.test_align_points_with_floor()  # DELETEME: Testing Point rotation
#         # self.test_align_points_with_upper_body()  # DELETEME: Testing Point rotation

#         return actions
    

#     def update_body_frame(self):
#         """Updates the body_frame, body and the floor_plane"""
#         # Get body frame
#         if not self.kinect.has_new_body_frame():
#             return False
#         body_frame = self.kinect.get_last_body_frame()

#         # Get tracked body
#         tracked_bodies = [body for body in body_frame.bodies if body.is_tracked]
#         if len(tracked_bodies) == 0:
#             return False
#         self.body = tracked_bodies[0]

#         # Get floor plane
#         self.floor_plane = body_frame.floor_clip_plane

#         return True
    
    
#     def get_rel_joint_distances(self, joint1, joint2, with_direct_distance=True):
#         """Return the relative offsets between two joints in the x, y and z direction and the whole distance"""
#         # Get xyz-coordinates of joint1 (depending on the type of joint)
#         if hasattr(joint1, "x"):
#             joint1_x, joint1_y, joint1_z = joint1.x, joint1.y, joint1.z
#         elif hasattr(joint1, "Position"):
#             joint1_x, joint1_y, joint1_z = joint1.Position.x, joint1.Position.y, joint1.Position.z
#         elif isinstance(joint1, (list, tuple, np.ndarray)):
#             assert len(joint1) == 3, f"Joint must be a list, tuple or np.ndarray of length 3, but is of len {len(joint1)}"
#             joint1_x, joint1_y, joint1_z = joint1
#         else:
#             raise ValueError(f"Joint must be a list, tuple, np.ndarray or have attributes x, y and z, but is of type {type(joint1)}")
        
#         # Get xyz-coordinates of joint2 (depending on the type of joint)
#         if hasattr(joint2, "x"):
#             joint2_x, joint2_y, joint2_z = joint2.x, joint2.y, joint2.z
#         elif hasattr(joint2, "Position"):
#             joint2_x, joint2_y, joint2_z = joint2.Position.x, joint2.Position.y, joint2.Position.z
#         elif isinstance(joint2, (list, tuple, np.ndarray)):
#             assert len(joint2) == 3, f"Joint must be a list, tuple or np.ndarray of length 3, but is of len {len(joint2)}"
#             joint2_x, joint2_y, joint2_z = joint2
#         else:
#             raise ValueError(f"Joint must be a list, tuple, np.ndarray or have attributes x, y and z, but is of type {type(joint2)}")
        
#         rel_x, rel_y, rel_z = joint1_x - joint2_x, joint1_y - joint2_y, joint1_z - joint2_z

#         if not with_direct_distance:
#             return rel_x, rel_y, rel_z

#         direct_distance = math.sqrt(rel_x * rel_x + rel_y * rel_y + rel_z * rel_z)
#         return rel_x, rel_y, rel_z, direct_distance
    
    
#     def get_joint_distance(self, joint1, joint2):
#         """Return the direct distance between two joints"""
#         distances = self.get_rel_joint_distances(joint1, joint2)
#         if distances is None:
#             return None
#         return distances[-1]
    
    
#     def check_distances(self, joint1, joint2, min_x=None, max_x=None, min_y=None, max_y=None, min_z=None, max_z=None, min_dist=None, max_dist=None):
#         """Check if the relative distances between two joints are within the given bounds"""
#         rel_x, rel_y, rel_z, distance = self.get_rel_joint_distances(joint1, joint2)
#         # X
#         if min_x is not None and rel_x < min_x:
#             return False
#         if max_x is not None and rel_x > max_x:
#             return False
#         # Y
#         if min_y is not None and rel_y < min_y:
#             return False
#         if max_y is not None and rel_y > max_y:
#             return False
#         # Z
#         if min_z is not None and rel_z < min_z:
#             return False
#         if max_z is not None and rel_z > max_z:
#             return False
#         # Distance
#         if min_dist is not None and distance < min_dist:
#             return False
#         if max_dist is not None and distance > max_dist:
#             return False
        
#         return True  # Passed all checks
    

#     def check_distances_log(self, joint1, joint2, log_name:str,
#                             min_x=None, max_x=None, min_y=None, max_y=None,
#                             min_z=None, max_z=None, min_dist=None, max_dist=None):
#         """Check if the relative distances between two joints are within the given bounds"""
#         passed_all_checks = True
#         rel_x, rel_y, rel_z, distance = self.get_rel_joint_distances(joint1, joint2)
#         print(f"[{log_name}] Only first 3 decimal points in log")
#         # X
#         if min_x is not None:
#             print(f"[{log_name}] rel_x({rel_x:.3f}) > min_x({min_x:.3f}) == {rel_x > min_x}")
#             if not rel_x > min_x: passed_all_checks = False
#         if max_x is not None:
#             print(f"[{log_name}] rel_x({rel_x:.3f}) < max_x({max_x:.3f}) == {rel_x < max_x}")
#             if not rel_x < max_x: passed_all_checks = False
#         # Y
#         if min_y is not None:
#             print(f"[{log_name}] rel_y({rel_y:.3f}) > min_y({min_y:.3f}) == {rel_y > min_y}")
#             if not rel_y > min_y: passed_all_checks = False
#         if max_y is not None:
#             print(f"[{log_name}] rel_y({rel_y:.3f}) < max_y({max_y:.3f}) == {rel_y < max_y}")
#             if not rel_y < max_y: passed_all_checks = False
#         # Z
#         if min_z is not None:
#             print(f"[{log_name}] rel_z({rel_z:.3f}) > min_z({min_z:.3f}) == {rel_z > min_z}")
#             if not rel_z > min_z: passed_all_checks = False
#         if max_z is not None:
#             print(f"[{log_name}] rel_z({rel_z:.3f}) < max_z({max_z:.3f}) == {rel_z < max_z}")
#             if not rel_z < max_z: passed_all_checks = False
#         # Distance
#         if min_dist is not None:
#             print(f"[{log_name}] distance({distance:.3f}) > min_dist({min_dist:.3f}) == {distance > min_dist}")
#             if not distance > min_dist: passed_all_checks = False
#         if max_dist is not None:
#             print(f"[{log_name}] distance({distance:.3f}) < max_dist({max_dist:.3f}) == {distance < max_dist}")
#             if not distance < max_dist: passed_all_checks = False
#         print(f"[{log_name}] passed_all_checks == {passed_all_checks}")
        
#         return passed_all_checks

    
#     def get_floor_distance(self, joint):
#         """Return the shortest distance between a given joint and the floor plane"""
#         if self.floor_plane is None:
#             return None
#         # Get xyz-coordinates of joint (depending on the type of joint)
#         if hasattr(joint, "x"):
#             joint_x, joint_y, joint_z = joint.x, joint.y, joint.z
#         elif hasattr(joint, "Position"):
#             joint_x, joint_y, joint_z = joint.Position.x, joint.Position.y, joint.Position.z
#         elif isinstance(joint, (list, tuple, np.ndarray)):
#             assert len(joint) == 3, f"Joint must be a list, tuple or np.ndarray of length 3, but is of len {len(joint)}"
#             joint_x, joint_y, joint_z = joint
#         else:
#             raise ValueError(f"Joint must be a list, tuple, np.ndarray or have attributes x, y and z, but is of type {type(joint)}")
        
#         # Get floor plane
#         floor_x, floor_y, floor_z, floor_w = self.floor_plane.x, self.floor_plane.y, self.floor_plane.z, self.floor_plane.w

#         # Calculate the distance between the joint and the floor plane
#         numerator = floor_x * joint_x + floor_y * joint_y + floor_z * joint_z + floor_w
#         denominator = math.sqrt(floor_x * floor_x + floor_y * floor_y + floor_z * floor_z)
#         distance_to_floor = numerator / denominator

#         return distance_to_floor
    

#     @staticmethod
#     def single_position_to_numpy(point):
#         """Convert a single point to a numpy array"""
#         if isinstance(point, np.ndarray):
#             return point
#         elif isinstance(point, (list, tuple)):
#             return np.array(point)
#         elif hasattr(point, "x"):
#             return np.array([point.x, point.y, point.z])
#         elif hasattr(point, "Position"):
#             return np.array([point.Position.x, point.Position.y, point.Position.z])
#         else:
#             raise ValueError(f"Point must be a list, tuple, np.ndarray or have attributes x, y and z, but is of type {type(point)}")
        

#     @classmethod
#     def positions_to_numpy(cls, points):
#         """Convert a multiple points to a numpy array"""
#         new_points = []
#         for p in points:
#             new_points.append(cls.single_position_to_numpy(p))
#         new_points = np.array(new_points)
#         return new_points
    

#     @staticmethod
#     def move_origin(points, new_origin):
#         """Move the origin of the given points to the given new origin"""
#         new_points = []
#         new_origin = np.array(new_origin)
#         for p in points:
#             p = np.array(p)
#             new_points.append(p - new_origin)
#         return new_points


#     def align_points_with_floor(self, points):
#         """Align the given points with the floor such that the x and z axis are parallel to the floor plane"""
#         return point_transform.rotate_points_with_vectors(points, from_vector=self.floor_plane, to_vector=(0, 1, 0))


#     def align_points_with_upper_body(self, points):
#         """
#             Idea is to compare joint positions in a coordinate system relative to the upper-body's rotation and lean.
#             Make the SpineShoulder joint the new origin and align the new coordinate axis with the upper-body's rotation and lean.
#             X-Axis: From ShoulderLeft to ShoulderRight. Increasing to the users right.
#             Y-Axis: From SpineMid to SpineShoulder. Increasing to the users right.
#             Z-Axis: Ortogonal  to the X and Y axis (cross-product of y_vector x x_vector = z_vector). Increasing to the users front.
#             We act as if the new x and y are always orthogonal to each other, even though they are not.
#         """
#         if self.body is None:
#             return points
#         # Move origin to SpineShoulder
#         new_origin_joint = self.body.joints[PyKinectV2.JointType_SpineMid].Position
#         new_origin = np.array([new_origin_joint.x, new_origin_joint.y, new_origin_joint.z])
#         aligned_points = self.positions_to_numpy(points)
#         aligned_points = self.move_origin(aligned_points, new_origin)

#         # Get joint positions as np arrays
#         shoulder_left = self.body.joints[PyKinectV2.JointType_ShoulderLeft].Position  # x-axis vector tail position
#         shoulder_left = np.array([shoulder_left.x, shoulder_left.y, shoulder_left.z])
#         shoulder_right = self.body.joints[PyKinectV2.JointType_ShoulderRight].Position  # x-axis vector head position
#         shoulder_right = np.array([shoulder_right.x, shoulder_right.y, shoulder_right.z])
#         spine_mid = self.body.joints[PyKinectV2.JointType_SpineMid].Position  # y-axis vector tail position
#         spine_mid = np.array([spine_mid.x, spine_mid.y, spine_mid.z])
#         spine_shoulder = self.body.joints[PyKinectV2.JointType_SpineShoulder].Position  # y-axis vector head position
#         spine_shoulder = np.array([spine_shoulder.x, spine_shoulder.y, spine_shoulder.z])
#         # Move origins to SpineShoulder
#         shoulder_left -= new_origin
#         shoulder_right -= new_origin
#         spine_mid -= new_origin
#         spine_shoulder -= new_origin
        
#         # Align y-axis
#         # Make y-axis go through the SpineMid and SpineShoulder joints
#         # y_axis_vector = (spine_shoulder.x - spine_mid.x, spine_shoulder.y - spine_mid.y, spine_shoulder.z - spine_mid.z)  # Vector from left to right shoulder
#         y_axis_vector = np.array(spine_shoulder) - np.array(spine_mid)  # Vector from left to right shoulder
#         aligned_points = point_transform.rotate_points_with_vectors(points, from_vector=y_axis_vector, to_vector=np.array([0, 1, 0]))
#         shoulder_right, shoulder_left = point_transform.rotate_points_with_vectors(np.array([shoulder_right, shoulder_left]), from_vector=y_axis_vector, to_vector=np.array([0, 1, 0]))

#         # Align x-axis
#         # Make x-axis go through the left and right shoulder joints
#         # x_axis_vector = (shoulder_right.x - shoulder_left.x, shoulder_right.y - shoulder_left.y, shoulder_right.z - shoulder_left.z)  # Vector from left to right shoulder
#         x_axis_vector = np.array(shoulder_right) - np.array(shoulder_left)  # Vector from left to right shoulder
#         aligned_points = point_transform.rotate_points_with_vectors(points, from_vector=x_axis_vector, to_vector=np.array([1, 0, 0]))
#         # spine_mid, spine_shoulder = point_transform.rotate_points_with_vectors(np.array([spine_mid, spine_shoulder]), from_vector=x_axis_vector, to_vector=np.array([1, 0, 0]))

#         # Print the new axis directions in the original coordinate system 
#         print()
#         new_x_axis = x_axis_vector / np.linalg.norm(x_axis_vector)
#         print(f"new_x_axis:          x : {new_x_axis[0]:.3f}  |  y : {new_x_axis[1]:.3f}  |  z : {new_x_axis[2]:.3f}")
#         original_spine_mid = self.body.joints[PyKinectV2.JointType_SpineMid].Position  # y-axis vector tail position
#         original_spine_mid = np.array([original_spine_mid.x, original_spine_mid.y, original_spine_mid.z])
#         original_spine_shoulder = self.body.joints[PyKinectV2.JointType_SpineShoulder].Position  # y-axis vector head position
#         original_spine_shoulder = np.array([original_spine_shoulder.x, original_spine_shoulder.y, original_spine_shoulder.z])
#         new_y_axis = original_spine_shoulder - original_spine_mid
#         new_y_axis = new_y_axis / np.linalg.norm(new_y_axis)
#         print(f"new_y_axis:          x : {new_y_axis[0]:.3f}  |  y : {new_y_axis[1]:.3f}  |  z : {new_y_axis[2]:.3f}")
#         new_z_axis = np.cross(new_y_axis, new_x_axis)
#         new_z_axis = new_z_axis / np.linalg.norm(np.cross(new_y_axis, new_x_axis))
#         print(f"new_z_axis:          x : {new_z_axis[0]:.3f}  |  y : {new_z_axis[1]:.3f}  |  z : {new_z_axis[2]:.3f}")

#         return aligned_points

    
#     def get_lean(self):
#         """
#             Gesture detection for player movement:
#             Returns the lean of the body in xy-coordinates
#         """
#         if self.body is None:
#             return (0.0, 0.0)
#         return (self.body.lean.x, self.body.lean.y)
    

#     def get_camera_move(self):
#         if self.body is None:
#             return (0.0, 0.0)

#         # Get the position of the left hand and wrist joints
#         left_hand_direction = np.array([0.0, 0.0])
#         if self.body.hand_left_state == PyKinectV2.HandState_Lasso:
#             # Get the position of the hands and wrist joints
#             left_hand_position = self.body.joints[PyKinectV2.JointType_HandLeft].Position
#             left_wrist_position = self.body.joints[PyKinectV2.JointType_WristLeft].Position
#             # Calculate the direction of the right hand
#             left_hand_direction = np.array([
#                 left_hand_position.x - left_wrist_position.x,
#                 left_hand_position.y - left_wrist_position.y,
#             ])
#             # Scale to normal vector
#             left_hand_direction = left_hand_direction / np.linalg.norm(left_hand_direction)

#         # Get the position of the right hand and wrist joints
#         right_hand_direction = np.array([0.0, 0.0])
#         if self.body.hand_right_state == PyKinectV2.HandState_Lasso:
#             # Get the position of the hands and wrist joints
#             right_hand_position = self.body.joints[PyKinectV2.JointType_HandRight].Position
#             right_wrist_position = self.body.joints[PyKinectV2.JointType_WristRight].Position
#             # Calculate the direction of the right hand
#             right_hand_direction = np.array([
#                 right_hand_position.x - right_wrist_position.x,
#                 right_hand_position.y - right_wrist_position.y,
#             ])
#             # Scale to normal vector
#             right_hand_direction = right_hand_direction / np.linalg.norm(right_hand_direction)

#         # Combine and normalize the hand directions
#         hand_direction = left_hand_direction + right_hand_direction
#         hand_direction_magnitude = np.linalg.norm(hand_direction)
#         if hand_direction_magnitude > 0.0:
#             hand_direction = hand_direction / hand_direction_magnitude

#         return hand_direction
    
    
#     def is_consuming_item(self) -> bool:
#         """
#             Gesture Detection for item consumption. Mainly for consuming an Estus Flask:
#             Returns True if the player is drinking an Estus, False otherwise
#         """
#         if self.body is None:
#             return False
        
#         # Detection parameters
#         t_last_consume_threshold = 1.2  # seconds
#         reset_distance = 0.5
#         # hand_head_bounds = {
#         #     "min_x" : -0.11, "max_x" : 0.015,
#         #     "min_y" : -0.3, "max_y" : 0.0,
#         #     "min_z" : -0.2, "max_z" : 0.0,
#         #     "min_dist" : 0.01, "max_dist" : 0.35,
#         # }
#         hand_head_bounds = {
#             "min_x" : -0.11, "max_x" : 0.15,
#             "min_y" : -0.3, "max_y" : 0.0,
#             "min_z" : -0.2, "max_z" : 0.0,
#             "min_dist" : 0.01, "max_dist" : 0.35,
#         }

#         # Get joints
#         right_hand_pos = self.body.joints[PyKinectV2.JointType_HandRight].Position
#         head_pos = self.body.joints[PyKinectV2.JointType_Head].Position
        
#         # Check parameters
#         seconds_since_last_consume = time.time() - self.t_last_consume
#         *_, head_hand_distance = self.get_rel_joint_distances(right_hand_pos, head_pos)
#         # print()
#         # print(f"right hand closed=={self.body.hand_right_state==PyKinectV2.HandState_Closed}  seconds_since_last_consumed=={seconds_since_last_consume:.3f}")
#         # self.check_distances_log(right_hand_pos, head_pos, log_name="consume RightHand-Head", **hand_head_bounds)
#         # print()
#         if head_hand_distance > reset_distance:  # if right hand and head are far away again, look for consume gesture again
#             self.t_last_consume = 0.0
#             return False
#         if self.body.hand_right_state != PyKinectV2.HandState_Closed:  # Check if right hand is closed
#             return False
#         if not self.check_distances(right_hand_pos, head_pos, **hand_head_bounds):
#             return False
#         if seconds_since_last_consume < t_last_consume_threshold:
#             self.t_last_consume = time.time()
#             return False
        
#         self.t_last_consume = time.time()
#         return True  # All checks for gesture detection passed
    
    
#     def triggered_jump(self):
#         """Gesture Detection for jumping"""
#         if self.body is None:
#             return False
#         t_last_trigger_threshold = 0.5
#         floor_distance_reset = 0.2
#         left_foot, right_foot = self.body.joints[PyKinectV2.JointType_FootLeft], self.body.joints[PyKinectV2.JointType_FootRight]
#         left_foot_height, right_foot_height = self.get_floor_distance(left_foot), self.get_floor_distance(right_foot)
#         seconds_since_trigger = time.time() - self.t_last_jump

#         # print(f"left_foot_height=={left_foot_height:.3f}  left_foot_height=={left_foot_height:.3f}")

#         # # Check if feet are off the floor
#         if left_foot_height < floor_distance_reset or right_foot_height < floor_distance_reset:
#             self.t_last_jump = 0.0
#             return False
#         if left_foot_height < 0.3 or right_foot_height < 0.3:
#             return False
#         if left_foot_height < 0.4 or right_foot_height < 0.4:
#             return False
#         if abs(left_foot_height - right_foot_height) > 0.3:
#             return False
#         if seconds_since_trigger < t_last_trigger_threshold:
#             self.t_last_jump = time.time()
#             return False
#         self.t_last_jump = time.time()
#         return True
    

#     @staticmethod
#     def feet_are_neutral(left_foot_height, right_foot_height) -> bool:
#         if left_foot_height < 0.12 and right_foot_height < 0.12:
#             return True
#         else:
#             return False
        

#     @staticmethod
#     def left_foot_active(left_foot_height, right_foot_height) -> bool:
#         if left_foot_height > 0.20 and right_foot_height < 0.15:
#             return True
#         else:
#             return False
        

#     @staticmethod
#     def right_foot_active(left_foot_height, right_foot_height) -> bool:
#         if right_foot_height > 0.20 and left_foot_height < 0.15:
#             return True
#         else:
#             return False
        

#     def triggered_run(self):
#         if self.body is None:
#             return False
#         min_run_duration = 0.6  # Duration the button has to be presseed to run and not roll
#         delta_min = 0.7  # 0.7==slow marching, 0.6==Fast Marching, 0.5==Close to Jogging, 0.4 Jogging, 0.3 Running

#         t_now = time.time()
#         # Update the time the feet were last lifted/active
#         left_foot, right_foot = self.body.joints[PyKinectV2.JointType_FootLeft], self.body.joints[PyKinectV2.JointType_FootRight]
#         left_foot_height, right_foot_height = self.get_floor_distance(left_foot), self.get_floor_distance(right_foot)
#         if self.left_foot_active(left_foot_height, right_foot_height):
#             self.t_left_foot_active = t_now
#         if self.right_foot_active(left_foot_height, right_foot_height):
#             self.t_right_foot_active = t_now
#         # Check if running
#         feet_active_time_delta = abs(self.t_left_foot_active - self.t_right_foot_active)
#         seconds_since_last_foot_active = t_now - max(self.t_left_foot_active, self.t_right_foot_active) 
#         if feet_active_time_delta > delta_min:
#             if t_now - self.t_start_run < min_run_duration:  # Make sure the run button is pressed long enough to trigger runnign and not rolling
#                 return True
#             return False
#         if seconds_since_last_foot_active > delta_min:
#             if t_now - self.t_start_run < min_run_duration:  # Make sure the run button is pressed long enough to trigger runnign and not rolling
#                 return True
#             return False
#         self.t_start_run = t_now
#         return True
    
    
#     def triggered_roll(self):
#         """
#             Squating to trigger Gesture Detection for rolling. Returns True if squatting, False otherwise.
#             Due to problems with false-positive when only True if all conditions are met, it now is voted
#             on and only returns True if the portion of true detections is above the vote_true_threshold.
#         """
#         if self.body is None:
#             return False
#         vote_true_threshold = 0.66
#         vote_results = []
#         # req_hip_floor_distance = 0.38  # minimal required closeness to floor to allow squat detection
#         # req_feet_hip_distance = 0.48  # minimal required closeness from feet to hips to allow squat detection
#         req_hip_floor_distance = 0.40  # minimal required closeness to floor to allow squat detection
#         req_feet_hip_distance = 0.60  # minimal required closeness from feet to hips to allow squat detection
#         or_below_sum_distance = 2.00  # Also trigger roll if the sum of distances is below this value

#         # Get floor and joints
#         left_foot = self.body.joints[PyKinectV2.JointType_FootLeft]
#         right_foot = self.body.joints[PyKinectV2.JointType_FootRight]
#         left_hip = self.body.joints[PyKinectV2.JointType_HipLeft]
#         right_hip = self.body.joints[PyKinectV2.JointType_HipRight]

#         # Log values
#         left_hip_distance_to_floor = self.get_floor_distance(left_hip)
#         right_hip_distance_to_floor = self.get_floor_distance(right_hip)
#         left_foot_hip_distance = self.get_joint_distance(left_foot, left_hip)
#         right_foot_hip_distance = self.get_joint_distance(right_foot, right_hip)
#         # print()
#         # print(f"self.roll_gesture_ready : {self.roll_gesture_ready}")
#         # print(f"dist: left_hip_distance_to_floor({left_hip_distance_to_floor:.3f}) < req_hip_floor_distance({req_hip_floor_distance:.3f}) == {left_hip_distance_to_floor < req_hip_floor_distance}")
#         # print(f"dist: right_hip_distance_to_floor({right_hip_distance_to_floor:.3f}) < req_hip_floor_distance({req_hip_floor_distance:.3f}) == {right_hip_distance_to_floor < req_hip_floor_distance}")
#         # print(f"dist: left_foot_hip_distance({left_foot_hip_distance:.3f}) < req_feet_hip_distance({req_feet_hip_distance:.3f}) == {left_foot_hip_distance < req_feet_hip_distance}")
#         # print(f"dist: right_foot_hip_distance({right_foot_hip_distance:.3f}) < req_feet_hip_distance({req_feet_hip_distance:.3f}) == {right_foot_hip_distance < req_feet_hip_distance}")
#         # distances_sum = sum([left_hip_distance_to_floor, right_hip_distance_to_floor, left_foot_hip_distance, right_foot_hip_distance])
#         # print(f"distances_sum({distances_sum:.3f}) < or_below_sum_distance({or_below_sum_distance:.3f}) == {distances_sum < or_below_sum_distance}")
#         # print()

#         # Check the roll/squat parameter conditions
#         vote_results.append(left_hip_distance_to_floor > req_hip_floor_distance)
#         vote_results.append(right_hip_distance_to_floor > req_hip_floor_distance)
#         vote_results.append(left_foot_hip_distance > req_feet_hip_distance)
#         vote_results.append(right_foot_hip_distance > req_feet_hip_distance)

#         # Make decision
#         true_votes = vote_results.count(True)
#         # Gesture has not been reset/made ready yet. Return False.
#         if not self.roll_gesture_ready:
#             # Check if gesture can be reset to be triggered again
#             if true_votes == 0:
#                 self.roll_gesture_ready = True
#             return False
#         # Calculate and Evaluate number of conditions that are met
#         true_votes_portion = true_votes / len(vote_results)
#         if true_votes_portion > vote_true_threshold:
#             self.roll_gesture_ready = False
#             return True
#         # Trigger anyways if the sum of distances is low
#         elif sum([left_hip_distance_to_floor, right_hip_distance_to_floor, left_foot_hip_distance, right_foot_hip_distance]) < or_below_sum_distance:
#             self.roll_gesture_ready = False
#             return True
#         return False
    

#     def triggered_interaction(self):
#         """
#             Track the right hand for a grab gesture. Grab is detected as the right hand
#             held forwards changing from opened to closed.
#         """
#         if self.body is None:
#             return False
#         target_hand_shoulder_distance = 0.40

#         # Get joints and states
#         right_hand_position = self.body.joints[PyKinectV2.JointType_HandRight].Position
#         right_shoulder_position = self.body.joints[PyKinectV2.JointType_ShoulderRight].Position
#         right_hand_state = self.body.hand_right_state

#         # Check if hand is held towards kinect
#         hand_shoulder_z_distance = right_shoulder_position.z - right_hand_position.z
#         if hand_shoulder_z_distance > target_hand_shoulder_distance:
#             # Check if hand is closed
#             if right_hand_state == PyKinectV2.HandState_Closed:
#                 # Check if hand was previously open
#                 if self.t_last_grab_open_state > self.t_last_grab_closed_state:
#                     # Hand is closed and was previously open
#                     self.t_last_grab_closed_state = time.time()
#                     return True
#                 self.t_last_grab_closed_state = time.time()
#             # Check if hand is open
#             elif right_hand_state == PyKinectV2.HandState_Open:
#                 self.t_last_grab_open_state = time.time()
#         # Hand is not held towards kinect or is not open or closed
#         return False


#     def triggered_light_attack(self):
#         """
#             Track the right hand for a light attack gesture. Gesture motion starts with the right hand above the
#             right shoulder and ends with the right hand close to the left hip with little time inbetween.
#         """
#         if self.body is None:
#             return False
        
#         # # Position parameters
#         # max_motion_duration = 0.6
#         # hand_shoulder_bounds = {
#         #     "min_x" : -0.05, "max_x" : 0.25,
#         #     "min_y" : 0.0, "max_y" : 0.35,
#         #     "min_z" :  -0.25, "max_z" : 0.1,
#         #     "min_dist" : 0.1, "max_dist" : 0.45,
#         # }
#         # hand_hip_bounds = {
#         #     "min_x" : -0.2, "max_x" : 0.05,
#         #     "min_y" : 0.0, "max_y" : 0.15,
#         #     "min_z" :  -0.3, "max_z" : 0.0,
#         #     "min_dist" : 0.1, "max_dist" : 0.3,
#         # }
#         # Position parameters
#         # max_motion_duration = 0.4
#         max_shoulder_to_hip_motion_duration = 0.4
#         max_hip_to_shoulder_motion_duration = 0.33
#         hand_shoulder_bounds = {
#             "min_x" : -0.05, "max_x" : 0.25,
#             "min_y" : 0.0, "max_y" : 0.35,
#             "min_z" :  -0.30, "max_z" : 0.1,
#             "min_dist" : 0.1, "max_dist" : 0.45,
#         }
#         hand_hip_bounds = {
#             "min_x" : -0.2, "max_x" : 0.05,
#             "min_y" : -0.15, "max_y" : 0.15,
#             "min_z" :  -0.3, "max_z" : 0.0,
#             "min_dist" : 0.1, "max_dist" : 0.3,
#         }

#         # Get joint and distances
#         right_hand_position = self.body.joints[PyKinectV2.JointType_HandRight].Position
#         right_shoulder_position = self.body.joints[PyKinectV2.JointType_ShoulderRight].Position
#         left_hip_position = self.body.joints[PyKinectV2.JointType_HipLeft].Position

#         # print()
#         # print(f"self.body.hand_right_state==PyKinectV2.HandState_Closed  ==  {self.body.hand_right_state == PyKinectV2.HandState_Closed}")
#         # print(f"t since shoulder state == {time.time() - self.t_last_light_attack_shoulder_state:.3f}")
#         # print(f"t since hip state == {time.time() - self.t_last_light_attack_hip_state:.3f}")
#         # self.check_distances_log(right_hand_position, right_shoulder_position, **hand_shoulder_bounds, log_name="right shoulder to right hand")
#         # self.check_distances_log(right_hand_position, left_hip_position, **hand_hip_bounds, log_name="left hip to right hand")
#         # print()

#         # Check if closed right-hand above right-shoulder and handle it
#         if self.body.hand_right_state == PyKinectV2.HandState_Closed:
#             if self.check_distances(right_hand_position, right_shoulder_position, **hand_shoulder_bounds):
#                 self.t_last_light_attack_shoulder_state = time.time()  # Last time that the right hand was above the right shoulder is right now
#                 seconds_since_other_state = self.t_last_light_attack_shoulder_state - self.t_last_light_attack_hip_state

#                 # DELETE THIS
#                 if seconds_since_other_state < 1.0:
#                     print(f"to shoulder duration: \t{seconds_since_other_state}")

#                 if seconds_since_other_state < max_hip_to_shoulder_motion_duration:
#                     self.t_last_light_attack_hip_state = 0.0  # To require other state to be detected again before triggering again
#                     return True 
#                 else:
#                     return False

#         # Check if closed right-hand on left-hip and handle it
#         if self.check_distances(right_hand_position, left_hip_position, **hand_hip_bounds):
#             self.t_last_light_attack_hip_state = time.time()  # Last time that the right hand was on the left hip is right now
#             seconds_since_other_state = self.t_last_light_attack_hip_state - self.t_last_light_attack_shoulder_state

#             # DELETE THIS
#             if seconds_since_other_state < 1.0:
#                 print(f"to hip duration: \t{seconds_since_other_state}")

#             if seconds_since_other_state < max_shoulder_to_hip_motion_duration:
#                 self.t_last_light_attack_shoulder_state = 0.0  # To require other state to be detected again before triggering again
#                 return True
#             else:
#                 return False

#         return False


#     def triggered_heavy_attack(self):
#         """ 
#             Look for stabbing movements.
#             Right hand fist on right waist -> to right hand fist forwards
#         """
#         if self.body is None:
#             return False
        
#         # Get joint and distances
#         right_hand_position = self.body.joints[PyKinectV2.JointType_HandRight].Position
#         right_shoulder_position = self.body.joints[PyKinectV2.JointType_ShoulderRight].Position
#         right_hip_position = self.body.joints[PyKinectV2.JointType_HipRight].Position
#         hand_shoulder_x, hand_shoulder_y, hand_shoulder_z = right_hand_position.x - right_shoulder_position.x, right_hand_position.y - right_shoulder_position.y, right_hand_position.z - right_shoulder_position.z
#         hand_shoulder_dist = math.sqrt(hand_shoulder_x**2 + hand_shoulder_y**2 + hand_shoulder_z**2)
#         hand_hip_x, hand_hip_y, hand_hip_z = right_hand_position.x - right_hip_position.x, right_hand_position.y - right_hip_position.y, right_hand_position.z - right_hip_position.z
#         hand_hip_dist = math.sqrt(hand_hip_x**2 + hand_hip_y**2 + hand_hip_z**2)
 
#         # print()
#         # print("Heavy attack")
#         # print(f"self.body.hand_right_state:{self.body.hand_right_state}")  # 3 = closed
#         # print(f"hand_shoulder_x:{hand_shoulder_x:.3f} | hand_shoulder_y:{hand_shoulder_y:.3f} | hand_shoulder_z:{hand_shoulder_z:.3f} | hand_shoulder_dist:{hand_shoulder_dist:.3f}")
#         # print(f"hand_hip_x:{hand_hip_x:.3f} | hand_hip_y:{hand_hip_y:.3f} | hand_hip_z:{hand_hip_z:.3f} | hand_hip_dist:{hand_hip_dist:.3f}")
#         # print()

#         return False


#     def triggered_block(self):
#         """
#             Left hand in closed state in front of torso
#         """
#         if self.body is None:
#             return False
        
#         # distance parameters
#         hand_spine_bounds = {
#             "min_x" : -0.1, "max_x" : 0.1,
#             "min_y" : -0.32, "max_y" : -0.03,
#             "min_z" :  -0.35, "max_z" : 0.0,
#             "min_dist" : 0.1, "max_dist" : 0.4,
#         }
        
#         # Get joint and distances
#         spine_shoulder_pos = self.body.joints[PyKinectV2.JointType_SpineShoulder].Position
#         left_hand_pos = self.body.joints[PyKinectV2.JointType_HandLeft].Position
#         # left_hand_is_closed = self.body.hand_left_state == PyKinectV2.HandState_Closed
#         # rel_x, rel_y, rel_z, dist = self.get_rel_joint_distances(left_hand_pos, spine_shoulder_pos)

#         # print(f"left-hand to spine-shoulder: x:{rel_x:.3f} | y:{rel_y:.3f} | z:{rel_z:.3f} | dist:{dist:.3f} | left_hand_is_closed:{left_hand_is_closed}")

#         distances_in_bounds = self.check_distances(left_hand_pos, spine_shoulder_pos, **hand_spine_bounds)

#         # if distances_in_bounds:
#         #     print("BLOCKING")
#         # else:
#         #     print("not blocking")

#         return distances_in_bounds


#     def triggered_parry(self):
#         """
#             Left Hand on right hip -> to left hand above left shoulder
#         """
#         if self.body is None:
#             return False
        
#         # Position parameters
#         target_start_end_duration = 0.6
#         hand_hip_bounds = {
#             "min_x" : 0.0, "max_x" : 0.15,
#             "min_y" : -0.01, "max_y" : 0.12,
#             "min_z" :  -0.2, "max_z" : -0.03,
#             "min_dist" : None, "max_dist" : 0.2,
#         }
#         hand_shoulder_bounds = {
#             "min_x" : -0.40, "max_x" : -0.05,
#             "min_y" : 0.18, "max_y" : 0.35,
#             "min_z" :  -0.21, "max_z" : 0.18,
#             "min_dist" : 0.10, "max_dist" : 0.50,
#         }

#         # Get joint and distances
#         left_hand_pos = self.body.joints[PyKinectV2.JointType_HandLeft].Position
#         left_shoulder_pos = self.body.joints[PyKinectV2.JointType_ShoulderLeft].Position
#         right_hip_pos = self.body.joints[PyKinectV2.JointType_HipRight].Position
        
#         hand_hip_x, hand_hip_y, hand_hip_z, hand_hip_dist = self.get_rel_joint_distances(left_hand_pos, right_hip_pos)
#         hand_shoulder_x, hand_shoulder_y, hand_shoulder_z, hand_shoulder_dist = self.get_rel_joint_distances(left_hand_pos, left_shoulder_pos)

#         # print()
#         # print(f"left hand state : {self.body.hand_left_state == PyKinectV2.HandState_Closed}")
#         # print(f"lHand-rHip:  x:{hand_hip_x:.3f} | y:{hand_hip_y:.3f} | z:{hand_hip_z:.3f} | dist:{hand_hip_dist:.3f}")
#         # print(f"lHand-lShoulder:  x:{hand_shoulder_x:.3f} | y:{hand_shoulder_y:.3f} | z:{hand_shoulder_z:.3f} | dist:{hand_shoulder_dist:.3f}")
#         # print()

#         # Parry start (hand on hip)
#         if self.check_distances(left_hand_pos, right_hip_pos, **hand_hip_bounds):
#             self.t_last_parry_start = time.time()
#             return False

#         # Parry end (hand over shoulder)
#         if self.body.hand_left_state == PyKinectV2.HandState_Closed:  # Hand closed
#             seconds_since_parry_start = time.time() - self.t_last_parry_start
#             if seconds_since_parry_start < target_start_end_duration:  # Parry start within time limit
#                 if not self.check_distances(left_hand_pos, left_shoulder_pos, **hand_shoulder_bounds):  # Hand over shoulder
#                     self.t_last_parry_start = 0.0
#                     return True

#         return False


#     def triggered_lock_on(self):
#         """
#             Both open hands stretched out forwards. Hands fairly close together.
#             check_distances(self, joint1, joint2, min_x=None, max_x=None, min_y=None, max_y=None, min_z=None, max_z=None, min_dist=None, max_dist=None)
#         """
#         if self.body is None:
#             return False
        
#         # # distance parameters
#         # hand_shoulder_bounds = {
#         #     "min_x" : -0.15, "max_x" : 0.15,
#         #     "min_y" : -0.15, "max_y" : 0.15,
#         #     "min_z" :  None, "max_z" : -0.35,
#         # }
#         # hand_bounds = {
#         #     "min_x" : -0.3, "max_x" : 0.0,
#         #     "min_y" : -0.1, "max_y" : 0.1,
#         #     "min_z" :  -0.1, "max_z" : 0.1,
#         #     "min_dist" :  None, "max_dist" : 0.35,
#         # }
#         # distance parameters
#         hand_shoulder_bounds = {
#             "min_x" : -0.15, "max_x" : 0.15,
#             "min_y" : -0.15, "max_y" : 0.15,
#             "min_z" :  None, "max_z" : -0.35,
#         }
#         hand_bounds = {
#             "min_x" : -0.45, "max_x" : 0.0,
#             "min_y" : -0.1, "max_y" : 0.1,
#             "min_z" :  -0.1, "max_z" : 0.1,
#             "min_dist" :  None, "max_dist" : 0.45,
#         }
        
#         # Get joint positions and hand states
#         # Left side
#         left_hand_position = self.body.joints[PyKinectV2.JointType_HandLeft].Position
#         left_hand_open = self.body.hand_left_state == PyKinectV2.HandState_Open
#         left_shoulder_position = self.body.joints[PyKinectV2.JointType_ShoulderLeft].Position
#         # Right side
#         right_hand_position = self.body.joints[PyKinectV2.JointType_HandRight].Position
#         right_hand_open = self.body.hand_right_state == PyKinectV2.HandState_Open
#         right_shoulder_position = self.body.joints[PyKinectV2.JointType_ShoulderRight].Position

#         # print("\nLock-On Gesture:")
#         # print(f"self.floor_plane xyzw :  {self.floor_plane.x:.5f}  |  {self.floor_plane.y:.5f}  |  {self.floor_plane.z:.5f}  |  {self.floor_plane.w:.5f}")
#         # self.check_distances_log(left_hand_position, left_shoulder_position, log_name="Left-Hand to Left-Shoulder", **hand_shoulder_bounds)
#         # self.check_distances_log(right_hand_position, right_shoulder_position, log_name="Right-Hand to Right-Shoulder", **hand_shoulder_bounds)
#         # self.check_distances_log(left_hand_position, right_hand_position, log_name="Left-Hand to Right-Hand", **hand_bounds)
#         # print()

#         # Handle gesture reset
#         if not self.lock_on_gesture_ready:
#             hands_x, hands_y, hands_z, hands_dist = self.get_rel_joint_distances(left_hand_position, right_hand_position)
#             if abs(hands_x) > 0.50:
#                 self.lock_on_gesture_ready = True
#                 return False
#             if abs(hands_y) > 0.35:
#                 self.lock_on_gesture_ready = True
#                 return False
#             if abs(hands_z) > 0.35:
#                 self.lock_on_gesture_ready = True
#                 return False
#             if abs(hands_dist) > 0.60:
#                 self.lock_on_gesture_ready = True
#                 return False
#             left_hand_shoulder_x, left_hand_shoulder_y, left_hand_shoulder_z, left_hand_shoulder_dist = self.get_rel_joint_distances(left_hand_position, left_shoulder_position)
#             right_hand_shoulder_x, right_hand_shoulder_y, right_hand_shoulder_z, right_hand_shoulder_dist = self.get_rel_joint_distances(right_hand_position, right_shoulder_position)
#             if abs(left_hand_shoulder_y) > 0.25 or abs(right_hand_shoulder_y) > 0.25:
#                 self.lock_on_gesture_ready = True
#                 return False
#             if left_hand_shoulder_z > -0.25 or right_hand_shoulder_z > -0.25:
#                 self.lock_on_gesture_ready = True
#                 return False
#             return False
            

#         if not left_hand_open or not right_hand_open:
#             return False

#         # Check if all the criteria for the lock-on gesture are met
#         # Left-Hand to Left-Shoulder
#         if not self.check_distances(left_hand_position, left_shoulder_position, **hand_shoulder_bounds):
#             return False
#         # Right-Hand to Right-Shoulder
#         if not self.check_distances(right_hand_position, right_shoulder_position, **hand_shoulder_bounds):
#             return False
#         # Left-Hand to Right-Hand
#         if not self.check_distances(left_hand_position, right_hand_position, **hand_bounds):
#             return False

#         self.lock_on_gesture_ready = False
#         return True  # All the criteria for the lock-on gesture are met
    

#     def triggered_kick(self):
#         """
#             Right foot raised above floor and held towards kinect away from right
#             hip with the left foot on the floor.
#         """
#         if self.body is None:
#             return False
        
#         right_foot_pos = self.body.joints[PyKinectV2.JointType_FootRight].Position
#         left_foot_pos = self.body.joints[PyKinectV2.JointType_FootLeft].Position
#         right_hip_pos = self.body.joints[PyKinectV2.JointType_HipRight].Position

#         right_foot_floor_dist = self.get_floor_distance(right_foot_pos)
#         left_foot_floor_dist = self.get_floor_distance(left_foot_pos)
#         feet_x, feet_y, feet_z, feet_dist = self.get_rel_joint_distances(right_foot_pos, left_foot_pos)
#         # right_foot_hip_x, right_foot_hip_y, right_foot_hip_z, right_foot_hip_dist = self.get_rel_joint_distances(right_foot_pos, right_hip_pos)

#         # print()
#         # print(f"feet floor dist:  L:{left_foot_floor_dist:.3f}   R:{right_foot_floor_dist:.3f}")
#         # print(f"feet:   x:{feet_x:.3f}  y:{feet_y:.3f}  z:{feet_z:.3f}  dist:{feet_dist:.3f}")
#         # print(f"right_foot_hip:   x:{right_foot_hip_x:.3f}  y:{right_foot_hip_y:.3f}  z:{right_foot_hip_z:.3f}  dist:{right_foot_hip_dist:.3f}")
#         # print()

#         if not self.kick_gesture_ready:
#             # Check if the right foot is raised above the floor
#             if right_foot_floor_dist < 0.4:
#                 self.kick_gesture_ready = True
#                 return False
#             # Check if the left foot is on the floor
#             if feet_dist < 0.5:
#                 self.kick_gesture_ready = True
#                 return False
#             return False

#         # distance parameters
#         right_foot_floor_dist_min = 0.6
#         left_foot_floor_dist_max = 0.1
#         feet_bounds = {
#             # "min_z" : None, "max_z" : -0.4,
#             # "min_dist" : 0.7, "max_dist" : None,
#         }
#         foot_hip_bounds = {
#             "min_y" : -0.35, "max_y" : 0.35,
#             "min_dist" :  0.5, "max_dist" : None,
#         }

#         # Check if all the criteria for the lock-on gesture are met
#         if right_foot_floor_dist < right_foot_floor_dist_min:
#             return False  # Kick foot too low
#         if left_foot_floor_dist > left_foot_floor_dist_max:
#             return False  # Stand foot in air
        
#         if not self.check_distances(right_foot_pos, left_foot_pos, **feet_bounds):
#             return False
#         if not self.check_distances(right_foot_pos, right_hip_pos, **foot_hip_bounds):
#             return False
        
#         self.kick_gesture_ready = False
#         return True  # All the criteria for the lock-on gesture are met

    
#     def test_align_points_with_floor(self):
#         """Test if we can convert our points into a stable coordinate system where the xz-plane forms a plane parallel to the floor."""
#         if self.body is None:
#             return False
        
#         # Get original hand positions
#         original_left_hand_position = self.body.joints[PyKinectV2.JointType_HandLeft].Position
#         original_left_hand_position = original_left_hand_position.x, original_left_hand_position.y, original_left_hand_position.z
#         original_right_hand_position = self.body.joints[PyKinectV2.JointType_HandRight].Position
#         original_right_hand_position = original_right_hand_position.x, original_right_hand_position.y, original_right_hand_position.z
#         original_points = [original_left_hand_position, original_right_hand_position]
#         original_rel_x, original_rel_y, original_rel_z, original_rel_dist = self.get_rel_joint_distances(original_left_hand_position, original_right_hand_position)

#         # Get rotated hand positions (hopefully aligned with the floor plane)
#         rotated_points = self.align_points_with_floor(original_points)
#         rotated_left_hand_position, rotated_right_hand_position = rotated_points
#         rotated_rel_x, rotated_rel_y, rotated_rel_z, rotated_rel_dist = self.get_rel_joint_distances(rotated_left_hand_position, rotated_right_hand_position)

#         # Print results 
#         print()
#         print("Rotating points (hand joints) to hopefully align coordinate system with floor plane")
#         print()
#         print(f"self.floor_plane:                  x : {self.floor_plane.x:.3f}  |  y : {self.floor_plane.y:.3f}  |  z : {self.floor_plane.z:.3f}  |  w : {self.floor_plane.w:.3f}")
#         print()
#         print(f"original right hand:               x : {original_right_hand_position[0]:.3f}  |  y : {original_right_hand_position[1]:.3f}  |  z : {original_right_hand_position[2]:.3f}")
#         print(f"original left hand:                x : {original_left_hand_position[0]:.3f}  |  y : {original_left_hand_position[1]:.3f}  |  z : {original_left_hand_position[2]:.3f}")
#         print(f"original distances left to right:  x : {original_rel_x:.3f}  |  y : {original_rel_y:.3f}  |  z : {original_rel_z:.3f}  |  dist : {original_rel_dist:.3f}")
#         print()
#         print(f"rotated right hand:                x : {rotated_right_hand_position[0]:.3f}  |  y : {rotated_right_hand_position[1]:.3f}  |  z : {rotated_right_hand_position[2]:.3f}")
#         print(f"rotated left hand:                 x : {rotated_left_hand_position[0]:.3f}  |  y : {rotated_left_hand_position[1]:.3f}  |  z : {rotated_left_hand_position[2]:.3f}")
#         print(f"rotated distances left to right:   x : {rotated_rel_x:.3f}  |  y : {rotated_rel_y:.3f}  |  z : {rotated_rel_z:.3f}  |  dist : {rotated_rel_dist:.3f}")
#         print()  
#         print("-" * 100)


#     def test_align_points_with_upper_body(self):
#         """Test if we can convert our points into a stable coordinate system where the xz-plane forms a plane parallel to the floor."""
#         if self.body is None:
#             return False
        
#         # Get original hand positions
#         original_right_shoulder_position = self.body.joints[PyKinectV2.JointType_ShoulderRight].Position
#         original_right_shoulder_position = original_right_shoulder_position.x, original_right_shoulder_position.y, original_right_shoulder_position.z
#         original_right_hand_position = self.body.joints[PyKinectV2.JointType_HandRight].Position
#         original_right_hand_position = original_right_hand_position.x, original_right_hand_position.y, original_right_hand_position.z
#         original_points = [original_right_shoulder_position, original_right_hand_position]
#         original_rel_x, original_rel_y, original_rel_z, original_rel_dist = self.get_rel_joint_distances(original_right_hand_position, original_right_shoulder_position)

#         # Get rotated hand positions (hopefully aligned with the floor plane)
#         aligned_points = self.align_points_with_upper_body(original_points)
#         aligned_right_shoulder_position, aligned_right_hand_position = aligned_points
#         aligned_rel_x, aligned_rel_y, aligned_rel_z, aligned_rel_dist = self.get_rel_joint_distances(aligned_right_hand_position, aligned_right_shoulder_position)

#         # Print results
#         lean_x, lean_y = self.get_lean()
#         from_original_to_aligned = np.array([aligned_rel_x-original_rel_x, aligned_rel_y-original_rel_y, aligned_rel_z-original_rel_z, aligned_rel_dist-original_rel_dist])
#         print()
#         print("test_align_points_with_upper_body (align with spine first and then with shoulders)")
#         print()
#         print(f"self.lean:                            x : {lean_x:.3f}  |  y : {lean_y:.3f}")
#         print()
#         print(f"original right hand:                  x : {original_right_hand_position[0]:.3f}  |  y : {original_right_hand_position[1]:.3f}  |  z : {original_right_hand_position[2]:.3f}")
#         print(f"original right shoulder:              x : {original_right_shoulder_position[0]:.3f}  |  y : {original_right_shoulder_position[1]:.3f}  |  z : {original_right_shoulder_position[2]:.3f}")
#         print(f"original distances shoulder to hand:  x : {original_rel_x:.3f}  |  y : {original_rel_y:.3f}  |  z : {original_rel_z:.3f}  |  dist : {original_rel_dist:.3f}")
#         print()
#         print(f"aligned right hand:                   x : {aligned_right_hand_position[0]:.3f}  |  y : {aligned_right_hand_position[1]:.3f}  |  z : {aligned_right_hand_position[2]:.3f}")
#         print(f"aligned right shoulder:               x : {aligned_right_shoulder_position[0]:.3f}  |  y : {aligned_right_shoulder_position[1]:.3f}  |  z : {aligned_right_shoulder_position[2]:.3f}")
#         print(f"aligned distances shoulder to hand:   x : {aligned_rel_x:.3f}  |  y : {aligned_rel_y:.3f}  |  z : {aligned_rel_z:.3f}  |  dist : {aligned_rel_dist:.3f}")
#         print()
#         print(f"from_original_to_aligned:             x : {from_original_to_aligned[0]:.3f}  |  y : {from_original_to_aligned[1]:.3f}  |  z : {from_original_to_aligned[2]:.3f}  |  dist : {from_original_to_aligned[3]:.3f}")
#         print()
#         print("-" * 100)

        


# # ==================================================================================================



# def testy1():
#     import threading
#     args = {}
#     kwargs = {
#         "display_type": "pygame",
#         "single_skeleton_color": (120, 20, 120),
#         # "window_size": (1280, 720),
#         "window_size": (1024, 576),
#         "show_fps": True,
#     }
#     cam_thread = threading.Thread(target=show_camera, args=args, kwargs=kwargs)
#     cam_thread.start()
#     kinect_detector = KinectDetector()
#     while True:
#         time.sleep(0.05)
#         if not cam_thread.is_alive():
#             exit()
#         kinect_detector.get_actions(update_body_frame=True)


# if __name__ == "__main__":
#     testy1()

# if __name__ == "__main__":
#     print(f"\n\n    Finished Script '{os.path.basename(__file__)}' at {time.strftime('%Y-%m-%d_%H-%M-%S')}    \n\n")

































































































# import os
# import time
# import math

# import numpy as np
# import cv2
# from pykinect2 import PyKinectV2, PyKinectRuntime

# import point_transform


# kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body)


# class KinectDetector:
#     """Detects gestures using the Kinect and returns the corresponding action"""
#     def __init__(self):
#         # self.kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body)
#         self.kinect = kinect
#         self.body_frame = None
#         self.tracked_bodies = None
#         self.body = None
#         self.floor_plane = None
#         self.color_frame = None

#         self.body_frame_num = 0
#         self.color_frame_num = 0

#         # Timestamp of last gesture triggers (can be reseted to allow repeat gestures quickly)
#         self.t_last_consume = 0.0
#         self.t_last_jump = 0.0
#         self.t_left_foot_active = 0.0
#         self.t_right_foot_active = 0.0
#         self.t_start_run = 0.0
#         self.t_last_grab_open_state = 0.0
#         self.t_last_grab_closed_state = 0.0
#         # self.t_last_light_attack_start = 0.0
#         self.t_last_light_attack_shoulder_state = 0.0
#         self.t_last_light_attack_hip_state = 0.0
#         self.t_last_parry_start = 0.0

#         # Gesture readyness flags - require reset of gesture before being able to trigger again to prevent unwanted triggers
#         self.roll_gesture_ready = True
#         self.lock_on_gesture_ready = True
#         self.kick_gesture_ready = True


#     def get_actions(self, update_body_frame=True, update_color_frame=True):
#         """Returns the actions that are currently detected"""
#         if update_body_frame:
#             self.update_body_frame()
#         if update_color_frame:
#             self.update_color_frame()
#         actions = {}
#         actions["lean"] = self.get_lean()
#         actions["camera_move"] = self.get_camera_move()
#         actions["is_consuming_item"] = self.is_consuming_item()
#         actions["triggered_jump"] = self.triggered_jump()
#         actions["triggered_run"] = self.triggered_run()
#         actions["triggered_roll"] = self.triggered_roll()
#         actions["triggered_interaction"] = self.triggered_interaction()
#         actions["triggered_light_attack"] = self.triggered_light_attack()
#         actions["triggered_block"] = self.triggered_block()
#         actions["triggered_parry"] = self.triggered_parry()
#         actions["triggered_lock_on"] = self.triggered_lock_on()
#         actions["triggered_kick"] = self.triggered_kick()
#         # not implemented yet (NIY)
#         actions["triggered_heavy_attack"] = self.triggered_heavy_attack()  # NIY

#         # self.test_align_points_with_floor()  # DELETEME: Testing Point rotation
#         self.test_align_points_with_upper_body()  # DELETEME: Testing Point rotation

#         return actions
    

#     def update_body_frame(self):
#         """Updates the body_frame, body and the floor_plane"""
#         # Get body frame
#         if not self.kinect.has_new_body_frame():
#             return False
#         self.body_frame = self.kinect.get_last_body_frame()
#         self.body_frame_num += 1

#         # Get tracked body
#         self.tracked_bodies = [body for body in self.body_frame.bodies if body.is_tracked]
#         if len(self.tracked_bodies) == 0:
#             return False
#         self.body = self.tracked_bodies[0]

#         # Get floor plane
#         self.floor_plane = self.body_frame.floor_clip_plane

#         return True
    

#     def update_color_frame(self):
#         """Updates the color_frame, body and the floor_plane"""
#         # Get body frame
#         if not self.kinect.has_new_color_frame():
#             return False
#         self.color_frame = self.kinect.get_last_color_frame()
#         self.color_frame_num += 1
#         return True
    
    
#     def get_rel_joint_distances(self, joint1, joint2, with_direct_distance=True):
#         """Return the relative offsets between two joints in the x, y and z direction and the whole distance"""
#         # Get xyz-coordinates of joint1 (depending on the type of joint)
#         if hasattr(joint1, "x"):
#             joint1_x, joint1_y, joint1_z = joint1.x, joint1.y, joint1.z
#         elif hasattr(joint1, "Position"):
#             joint1_x, joint1_y, joint1_z = joint1.Position.x, joint1.Position.y, joint1.Position.z
#         elif isinstance(joint1, (list, tuple, np.ndarray)):
#             assert len(joint1) == 3, f"Joint must be a list, tuple or np.ndarray of length 3, but is of len {len(joint1)}"
#             joint1_x, joint1_y, joint1_z = joint1
#         else:
#             raise ValueError(f"Joint must be a list, tuple, np.ndarray or have attributes x, y and z, but is of type {type(joint1)}")
        
#         # Get xyz-coordinates of joint2 (depending on the type of joint)
#         if hasattr(joint2, "x"):
#             joint2_x, joint2_y, joint2_z = joint2.x, joint2.y, joint2.z
#         elif hasattr(joint2, "Position"):
#             joint2_x, joint2_y, joint2_z = joint2.Position.x, joint2.Position.y, joint2.Position.z
#         elif isinstance(joint2, (list, tuple, np.ndarray)):
#             assert len(joint2) == 3, f"Joint must be a list, tuple or np.ndarray of length 3, but is of len {len(joint2)}"
#             joint2_x, joint2_y, joint2_z = joint2
#         else:
#             raise ValueError(f"Joint must be a list, tuple, np.ndarray or have attributes x, y and z, but is of type {type(joint2)}")
        
#         rel_x, rel_y, rel_z = joint1_x - joint2_x, joint1_y - joint2_y, joint1_z - joint2_z

#         if not with_direct_distance:
#             return rel_x, rel_y, rel_z

#         direct_distance = math.sqrt(rel_x * rel_x + rel_y * rel_y + rel_z * rel_z)
#         return rel_x, rel_y, rel_z, direct_distance
    
    
#     def get_joint_distance(self, joint1, joint2):
#         """Return the direct distance between two joints"""
#         distances = self.get_rel_joint_distances(joint1, joint2)
#         if distances is None:
#             return None
#         return distances[-1]
    
    
#     def check_distances(self, joint1, joint2, min_x=None, max_x=None, min_y=None, max_y=None, min_z=None, max_z=None, min_dist=None, max_dist=None):
#         """Check if the relative distances between two joints are within the given bounds"""
#         rel_x, rel_y, rel_z, distance = self.get_rel_joint_distances(joint1, joint2)
#         # X
#         if min_x is not None and rel_x < min_x:
#             return False
#         if max_x is not None and rel_x > max_x:
#             return False
#         # Y
#         if min_y is not None and rel_y < min_y:
#             return False
#         if max_y is not None and rel_y > max_y:
#             return False
#         # Z
#         if min_z is not None and rel_z < min_z:
#             return False
#         if max_z is not None and rel_z > max_z:
#             return False
#         # Distance
#         if min_dist is not None and distance < min_dist:
#             return False
#         if max_dist is not None and distance > max_dist:
#             return False
        
#         return True  # Passed all checks
    

#     def check_distances_log(self, joint1, joint2, log_name:str,
#                             min_x=None, max_x=None, min_y=None, max_y=None,
#                             min_z=None, max_z=None, min_dist=None, max_dist=None):
#         """Check if the relative distances between two joints are within the given bounds"""
#         passed_all_checks = True
#         rel_x, rel_y, rel_z, distance = self.get_rel_joint_distances(joint1, joint2)
#         print(f"[{log_name}] Only first 3 decimal points in log")
#         # X
#         if min_x is not None:
#             print(f"[{log_name}] rel_x({rel_x:.3f}) > min_x({min_x:.3f}) == {rel_x > min_x}")
#             if not rel_x > min_x: passed_all_checks = False
#         if max_x is not None:
#             print(f"[{log_name}] rel_x({rel_x:.3f}) < max_x({max_x:.3f}) == {rel_x < max_x}")
#             if not rel_x < max_x: passed_all_checks = False
#         # Y
#         if min_y is not None:
#             print(f"[{log_name}] rel_y({rel_y:.3f}) > min_y({min_y:.3f}) == {rel_y > min_y}")
#             if not rel_y > min_y: passed_all_checks = False
#         if max_y is not None and rel_y > max_y:
#             print(f"[{log_name}] rel_y({rel_y:.3f}) < max_y({max_y:.3f}) == {rel_y < max_y}")
#             if not rel_y < max_y: passed_all_checks = False
#         # Z
#         if min_z is not None:
#             print(f"[{log_name}] rel_z({rel_z:.3f}) > min_z({min_z:.3f}) == {rel_z > min_z}")
#             if not rel_z > min_z: passed_all_checks = False
#         if max_z is not None:
#             print(f"[{log_name}] rel_z({rel_z:.3f}) < max_z({max_z:.3f}) == {rel_z < max_z}")
#             if not rel_z < max_z: passed_all_checks = False
#         # Distance
#         if min_dist is not None:
#             print(f"[{log_name}] distance({distance:.3f}) > min_dist({min_dist:.3f}) == {distance > min_dist}")
#             if not distance > min_dist: passed_all_checks = False
#         if max_dist is not None:
#             print(f"[{log_name}] distance({distance:.3f}) < max_dist({max_dist:.3f}) == {distance < max_dist}")
#             if not distance < max_dist: passed_all_checks = False
#         print(f"[{log_name}] passed_all_checks == {passed_all_checks}")
        
#         return passed_all_checks

    
#     def get_floor_distance(self, joint):
#         """Return the shortest distance between a given joint and the floor plane"""
#         if self.floor_plane is None:
#             return None
#         # Get xyz-coordinates of joint (depending on the type of joint)
#         if hasattr(joint, "x"):
#             joint_x, joint_y, joint_z = joint.x, joint.y, joint.z
#         elif hasattr(joint, "Position"):
#             joint_x, joint_y, joint_z = joint.Position.x, joint.Position.y, joint.Position.z
#         elif isinstance(joint, (list, tuple, np.ndarray)):
#             assert len(joint) == 3, f"Joint must be a list, tuple or np.ndarray of length 3, but is of len {len(joint)}"
#             joint_x, joint_y, joint_z = joint
#         else:
#             raise ValueError(f"Joint must be a list, tuple, np.ndarray or have attributes x, y and z, but is of type {type(joint)}")
        
#         # Get floor plane
#         floor_x, floor_y, floor_z, floor_w = self.floor_plane.x, self.floor_plane.y, self.floor_plane.z, self.floor_plane.w

#         # Calculate the distance between the joint and the floor plane
#         numerator = floor_x * joint_x + floor_y * joint_y + floor_z * joint_z + floor_w
#         denominator = math.sqrt(floor_x * floor_x + floor_y * floor_y + floor_z * floor_z)
#         distance_to_floor = numerator / denominator

#         return distance_to_floor
    

#     @staticmethod
#     def single_position_to_numpy(point):
#         """Convert a single point to a numpy array"""
#         if isinstance(point, np.ndarray):
#             return point
#         elif isinstance(point, (list, tuple)):
#             return np.array(point)
#         elif hasattr(point, "x"):
#             return np.array([point.x, point.y, point.z])
#         elif hasattr(point, "Position"):
#             return np.array([point.Position.x, point.Position.y, point.Position.z])
#         else:
#             raise ValueError(f"Point must be a list, tuple, np.ndarray or have attributes x, y and z, but is of type {type(point)}")
        

#     @classmethod
#     def positions_to_numpy(cls, points):
#         """Convert a multiple points to a numpy array"""
#         new_points = []
#         for p in points:
#             new_points.append(cls.single_position_to_numpy(p))
#         new_points = np.array(new_points)
#         return new_points
    

#     @staticmethod
#     def move_origin(points, new_origin):
#         """Move the origin of the given points to the given new origin"""
#         new_points = []
#         new_origin = np.array(new_origin)
#         for p in points:
#             p = np.array(p)
#             new_points.append(p - new_origin)
#         return new_points


#     def align_points_with_floor(self, points):
#         """Align the given points with the floor such that the x and z axis are parallel to the floor plane"""
#         return point_transform.rotate_points_with_vectors(points, from_vector=self.floor_plane, to_vector=(0, 1, 0))


#     # def align_points_with_upper_body(self, points):
#     #     """
#     #         Idea is to compare joint positions in a coordinate system relative to the upper-body's rotation and lean.
#     #         Make the SpineShoulder joint the new origin and align the new coordinate axis with the upper-body's rotation and lean.
#     #         X-Axis: From ShoulderLeft to ShoulderRight. Increasing to the users right.
#     #         Y-Axis: From SpineMid to SpineShoulder. Increasing to the users right.
#     #         Z-Axis: Ortogonal  to the X and Y axis (cross-product of y_vector x x_vector = z_vector). Increasing to the users front.
#     #         We act as if the new x and y are always orthogonal to each other, even though they are not.
#     #     """
#     #     if self.body is None:
#     #         return points
#     #     # Move origin to SpineShoulder
#     #     new_origin_joint = self.body.joints[PyKinectV2.JointType_SpineMid].Position
#     #     new_origin = np.array([new_origin_joint.x, new_origin_joint.y, new_origin_joint.z])
#     #     aligned_points = self.positions_to_numpy(points)
#     #     aligned_points = self.move_origin(aligned_points, new_origin)

#     #     # Get joint positions as np arrays
#     #     shoulder_left = self.body.joints[PyKinectV2.JointType_ShoulderLeft].Position  # x-axis vector tail position
#     #     shoulder_left = np.array([shoulder_left.x, shoulder_left.y, shoulder_left.z])
#     #     shoulder_right = self.body.joints[PyKinectV2.JointType_ShoulderRight].Position  # x-axis vector head position
#     #     shoulder_right = np.array([shoulder_right.x, shoulder_right.y, shoulder_right.z])
#     #     spine_mid = self.body.joints[PyKinectV2.JointType_SpineMid].Position  # y-axis vector tail position
#     #     spine_mid = np.array([spine_mid.x, spine_mid.y, spine_mid.z])
#     #     spine_shoulder = self.body.joints[PyKinectV2.JointType_SpineShoulder].Position  # y-axis vector head position
#     #     spine_shoulder = np.array([spine_shoulder.x, spine_shoulder.y, spine_shoulder.z])
#     #     # Move origins to SpineShoulder
#     #     shoulder_left -= new_origin
#     #     shoulder_right -= new_origin
#     #     spine_mid -= new_origin
#     #     spine_shoulder -= new_origin

#     #     # Align x-axis
#     #     # Make x-axis go through the left and right shoulder joints
#     #     # x_axis_vector = (shoulder_right.x - shoulder_left.x, shoulder_right.y - shoulder_left.y, shoulder_right.z - shoulder_left.z)  # Vector from left to right shoulder
#     #     x_axis_vector = shoulder_right - shoulder_left  # Vector from left to right shoulder
#     #     aligned_points = point_transform.rotate_points_with_vectors(points, from_vector=x_axis_vector, to_vector=np.array([1, 0, 0]))
#     #     spine_mid, spine_shoulder = point_transform.rotate_points_with_vectors(np.array([spine_mid, spine_shoulder]), from_vector=x_axis_vector, to_vector=np.array([1, 0, 0]))
        
#     #     # Align y-axis
#     #     # Make y-axis go through the SpineMid and SpineShoulder joints
#     #     # y_axis_vector = (spine_shoulder.x - spine_mid.x, spine_shoulder.y - spine_mid.y, spine_shoulder.z - spine_mid.z)  # Vector from left to right shoulder
#     #     y_axis_vector = np.array(spine_shoulder) - np.array(spine_mid)  # Vector from left to right shoulder
#     #     aligned_points = point_transform.rotate_points_with_vectors(points, from_vector=y_axis_vector, to_vector=np.array([0, 1, 0]))

#     #     # Print the new axis directions in the original coordinate system 
#     #     print()
#     #     new_x_axis = x_axis_vector / np.linalg.norm(x_axis_vector)
#     #     print(f"new_x_axis:          x : {new_x_axis[0]:.3f}  |  y : {new_x_axis[1]:.3f}  |  z : {new_x_axis[2]:.3f}")
#     #     original_spine_mid = self.body.joints[PyKinectV2.JointType_SpineMid].Position  # y-axis vector tail position
#     #     original_spine_mid = np.array([original_spine_mid.x, original_spine_mid.y, original_spine_mid.z])
#     #     original_spine_shoulder = self.body.joints[PyKinectV2.JointType_SpineShoulder].Position  # y-axis vector head position
#     #     original_spine_shoulder = np.array([original_spine_shoulder.x, original_spine_shoulder.y, original_spine_shoulder.z])
#     #     new_y_axis = original_spine_shoulder - original_spine_mid
#     #     new_y_axis = new_y_axis / np.linalg.norm(new_y_axis)
#     #     print(f"new_y_axis:          x : {new_y_axis[0]:.3f}  |  y : {new_y_axis[1]:.3f}  |  z : {new_y_axis[2]:.3f}")
#     #     new_z_axis = np.cross(new_y_axis, new_x_axis)
#     #     new_z_axis = new_z_axis / np.linalg.norm(np.cross(new_y_axis, new_x_axis))
#     #     print(f"new_z_axis:          x : {new_z_axis[0]:.3f}  |  y : {new_z_axis[1]:.3f}  |  z : {new_z_axis[2]:.3f}")

#     #     return aligned_points


#     def align_points_with_upper_body(self, points):
#         """
#             Idea is to compare joint positions in a coordinate system relative to the upper-body's rotation and lean.
#             Make the SpineShoulder joint the new origin and align the new coordinate axis with the upper-body's rotation and lean.
#             X-Axis: From ShoulderLeft to ShoulderRight. Increasing to the users right.
#             Y-Axis: From SpineMid to SpineShoulder. Increasing to the users right.
#             Z-Axis: Ortogonal  to the X and Y axis (cross-product of y_vector x x_vector = z_vector). Increasing to the users front.
#             We act as if the new x and y are always orthogonal to each other, even though they are not.
#         """
#         if self.body is None:
#             return points
#         # Move origin to SpineShoulder
#         new_origin_joint = self.body.joints[PyKinectV2.JointType_SpineMid].Position
#         new_origin = np.array([new_origin_joint.x, new_origin_joint.y, new_origin_joint.z])
#         aligned_points = self.positions_to_numpy(points)
#         aligned_points = self.move_origin(aligned_points, new_origin)

#         # Get joint positions as np arrays
#         shoulder_left = self.body.joints[PyKinectV2.JointType_ShoulderLeft].Position  # x-axis vector tail position
#         shoulder_left = np.array([shoulder_left.x, shoulder_left.y, shoulder_left.z])
#         shoulder_right = self.body.joints[PyKinectV2.JointType_ShoulderRight].Position  # x-axis vector head position
#         shoulder_right = np.array([shoulder_right.x, shoulder_right.y, shoulder_right.z])
#         spine_mid = self.body.joints[PyKinectV2.JointType_SpineMid].Position  # y-axis vector tail position
#         spine_mid = np.array([spine_mid.x, spine_mid.y, spine_mid.z])
#         spine_shoulder = self.body.joints[PyKinectV2.JointType_SpineShoulder].Position  # y-axis vector head position
#         spine_shoulder = np.array([spine_shoulder.x, spine_shoulder.y, spine_shoulder.z])
#         # Move origins to SpineShoulder
#         shoulder_left -= new_origin
#         shoulder_right -= new_origin
#         spine_mid -= new_origin
#         spine_shoulder -= new_origin
        
#         # Align y-axis
#         # Make y-axis go through the SpineMid and SpineShoulder joints
#         # y_axis_vector = (spine_shoulder.x - spine_mid.x, spine_shoulder.y - spine_mid.y, spine_shoulder.z - spine_mid.z)  # Vector from left to right shoulder
#         y_axis_vector = np.array(spine_shoulder) - np.array(spine_mid)  # Vector from left to right shoulder
#         aligned_points = point_transform.rotate_points_with_vectors(points, from_vector=y_axis_vector, to_vector=np.array([0, 1, 0]))
#         shoulder_right, shoulder_left = point_transform.rotate_points_with_vectors(np.array([shoulder_right, shoulder_left]), from_vector=y_axis_vector, to_vector=np.array([0, 1, 0]))

#         # Align x-axis
#         # Make x-axis go through the left and right shoulder joints
#         # x_axis_vector = (shoulder_right.x - shoulder_left.x, shoulder_right.y - shoulder_left.y, shoulder_right.z - shoulder_left.z)  # Vector from left to right shoulder
#         x_axis_vector = np.array(shoulder_right) - np.array(shoulder_left)  # Vector from left to right shoulder
#         aligned_points = point_transform.rotate_points_with_vectors(points, from_vector=x_axis_vector, to_vector=np.array([1, 0, 0]))
#         # spine_mid, spine_shoulder = point_transform.rotate_points_with_vectors(np.array([spine_mid, spine_shoulder]), from_vector=x_axis_vector, to_vector=np.array([1, 0, 0]))

#         # Print the new axis directions in the original coordinate system 
#         print()
#         new_x_axis = x_axis_vector / np.linalg.norm(x_axis_vector)
#         print(f"new_x_axis:          x : {new_x_axis[0]:.3f}  |  y : {new_x_axis[1]:.3f}  |  z : {new_x_axis[2]:.3f}")
#         original_spine_mid = self.body.joints[PyKinectV2.JointType_SpineMid].Position  # y-axis vector tail position
#         original_spine_mid = np.array([original_spine_mid.x, original_spine_mid.y, original_spine_mid.z])
#         original_spine_shoulder = self.body.joints[PyKinectV2.JointType_SpineShoulder].Position  # y-axis vector head position
#         original_spine_shoulder = np.array([original_spine_shoulder.x, original_spine_shoulder.y, original_spine_shoulder.z])
#         new_y_axis = original_spine_shoulder - original_spine_mid
#         new_y_axis = new_y_axis / np.linalg.norm(new_y_axis)
#         print(f"new_y_axis:          x : {new_y_axis[0]:.3f}  |  y : {new_y_axis[1]:.3f}  |  z : {new_y_axis[2]:.3f}")
#         new_z_axis = np.cross(new_y_axis, new_x_axis)
#         new_z_axis = new_z_axis / np.linalg.norm(np.cross(new_y_axis, new_x_axis))
#         print(f"new_z_axis:          x : {new_z_axis[0]:.3f}  |  y : {new_z_axis[1]:.3f}  |  z : {new_z_axis[2]:.3f}")

#         return aligned_points

    
#     def get_lean(self):
#         """
#             Gesture detection for player movement:
#             Returns the lean of the body in xy-coordinates
#         """
#         if self.body is None:
#             return (0.0, 0.0)
#         return (self.body.lean.x, self.body.lean.y)
    

#     def get_camera_move(self):
#         if self.body is None:
#             return (0.0, 0.0)

#         # Get the position of the left hand and wrist joints
#         left_hand_direction = np.array([0.0, 0.0])
#         if self.body.hand_left_state == PyKinectV2.HandState_Lasso:
#             # Get the position of the hands and wrist joints
#             left_hand_position = self.body.joints[PyKinectV2.JointType_HandLeft].Position
#             left_wrist_position = self.body.joints[PyKinectV2.JointType_WristLeft].Position
#             # Calculate the direction of the right hand
#             left_hand_direction = np.array([
#                 left_hand_position.x - left_wrist_position.x,
#                 left_hand_position.y - left_wrist_position.y,
#             ])
#             # Scale to normal vector
#             left_hand_direction = left_hand_direction / np.linalg.norm(left_hand_direction)

#         # Get the position of the right hand and wrist joints
#         right_hand_direction = np.array([0.0, 0.0])
#         if self.body.hand_right_state == PyKinectV2.HandState_Lasso:
#             # Get the position of the hands and wrist joints
#             right_hand_position = self.body.joints[PyKinectV2.JointType_HandRight].Position
#             right_wrist_position = self.body.joints[PyKinectV2.JointType_WristRight].Position
#             # Calculate the direction of the right hand
#             right_hand_direction = np.array([
#                 right_hand_position.x - right_wrist_position.x,
#                 right_hand_position.y - right_wrist_position.y,
#             ])
#             # Scale to normal vector
#             right_hand_direction = right_hand_direction / np.linalg.norm(right_hand_direction)

#         # Combine and normalize the hand directions
#         hand_direction = left_hand_direction + right_hand_direction
#         hand_direction_magnitude = np.linalg.norm(hand_direction)
#         if hand_direction_magnitude > 0.0:
#             hand_direction = hand_direction / hand_direction_magnitude

#         return hand_direction
    
    
#     def is_consuming_item(self) -> bool:
#         """
#             Gesture Detection for item consumption. Mainly for consuming an Estus Flask:
#             Returns True if the player is drinking an Estus, False otherwise
#         """
#         if self.body is None:
#             return False
        
#         # Detection parameters
#         t_last_consume_threshold = 1.2  # seconds
#         reset_distance = 0.5
#         # hand_head_bounds = {
#         #     "min_x" : -0.11, "max_x" : 0.015,
#         #     "min_y" : -0.3, "max_y" : 0.0,
#         #     "min_z" : -0.2, "max_z" : 0.0,
#         #     "min_dist" : 0.01, "max_dist" : 0.35,
#         # }
#         hand_head_bounds = {
#             "min_x" : -0.11, "max_x" : 0.15,
#             "min_y" : -0.3, "max_y" : 0.0,
#             "min_z" : -0.2, "max_z" : 0.0,
#             "min_dist" : 0.01, "max_dist" : 0.35,
#         }

#         # Get joints
#         right_hand_pos = self.body.joints[PyKinectV2.JointType_HandRight].Position
#         head_pos = self.body.joints[PyKinectV2.JointType_Head].Position
        
#         # Check parameters
#         seconds_since_last_consume = time.time() - self.t_last_consume
#         *_, head_hand_distance = self.get_rel_joint_distances(right_hand_pos, head_pos)
#         # print()
#         # print(f"right hand closed=={self.body.hand_right_state==PyKinectV2.HandState_Closed}  seconds_since_last_consumed=={seconds_since_last_consume:.3f}")
#         # self.check_distances_log(right_hand_pos, head_pos, log_name="consume RightHand-Head", **hand_head_bounds)
#         # print()
#         if head_hand_distance > reset_distance:  # if right hand and head are far away again, look for consume gesture again
#             self.t_last_consume = 0.0
#             return False
#         if self.body.hand_right_state != PyKinectV2.HandState_Closed:  # Check if right hand is closed
#             return False
#         if not self.check_distances(right_hand_pos, head_pos, **hand_head_bounds):
#             return False
#         if seconds_since_last_consume < t_last_consume_threshold:
#             self.t_last_consume = time.time()
#             return False
        
#         self.t_last_consume = time.time()
#         return True  # All checks for gesture detection passed
    
    
#     def triggered_jump(self):
#         """Gesture Detection for jumping"""
#         if self.body is None:
#             return False
#         t_last_trigger_threshold = 0.5
#         floor_distance_reset = 0.2
#         left_foot, right_foot = self.body.joints[PyKinectV2.JointType_FootLeft], self.body.joints[PyKinectV2.JointType_FootRight]
#         left_foot_height, right_foot_height = self.get_floor_distance(left_foot), self.get_floor_distance(right_foot)
#         seconds_since_trigger = time.time() - self.t_last_jump

#         # print(f"left_foot_height=={left_foot_height:.3f}  left_foot_height=={left_foot_height:.3f}")

#         # # Check if feet are off the floor
#         if left_foot_height < floor_distance_reset or right_foot_height < floor_distance_reset:
#             self.t_last_jump = 0.0
#             return False
#         if left_foot_height < 0.3 or right_foot_height < 0.3:
#             return False
#         if left_foot_height < 0.4 or right_foot_height < 0.4:
#             return False
#         if abs(left_foot_height - right_foot_height) > 0.3:
#             return False
#         if seconds_since_trigger < t_last_trigger_threshold:
#             self.t_last_jump = time.time()
#             return False
#         self.t_last_jump = time.time()
#         return True
    

#     @staticmethod
#     def feet_are_neutral(left_foot_height, right_foot_height) -> bool:
#         if left_foot_height < 0.12 and right_foot_height < 0.12:
#             return True
#         else:
#             return False
        

#     @staticmethod
#     def left_foot_active(left_foot_height, right_foot_height) -> bool:
#         if left_foot_height > 0.20 and right_foot_height < 0.15:
#             return True
#         else:
#             return False
        

#     @staticmethod
#     def right_foot_active(left_foot_height, right_foot_height) -> bool:
#         if right_foot_height > 0.20 and left_foot_height < 0.15:
#             return True
#         else:
#             return False
        

#     def triggered_run(self):
#         if self.body is None:
#             return False
#         min_run_duration = 0.6  # Duration the button has to be presseed to run and not roll
#         delta_min = 0.7  # 0.7==slow marching, 0.6==Fast Marching, 0.5==Close to Jogging, 0.4 Jogging, 0.3 Running

#         t_now = time.time()
#         # Update the time the feet were last lifted/active
#         left_foot, right_foot = self.body.joints[PyKinectV2.JointType_FootLeft], self.body.joints[PyKinectV2.JointType_FootRight]
#         left_foot_height, right_foot_height = self.get_floor_distance(left_foot), self.get_floor_distance(right_foot)
#         if self.left_foot_active(left_foot_height, right_foot_height):
#             self.t_left_foot_active = t_now
#         if self.right_foot_active(left_foot_height, right_foot_height):
#             self.t_right_foot_active = t_now
#         # Check if running
#         feet_active_time_delta = abs(self.t_left_foot_active - self.t_right_foot_active)
#         seconds_since_last_foot_active = t_now - max(self.t_left_foot_active, self.t_right_foot_active) 
#         if feet_active_time_delta > delta_min:
#             if t_now - self.t_start_run < min_run_duration:  # Make sure the run button is pressed long enough to trigger runnign and not rolling
#                 return True
#             return False
#         if seconds_since_last_foot_active > delta_min:
#             if t_now - self.t_start_run < min_run_duration:  # Make sure the run button is pressed long enough to trigger runnign and not rolling
#                 return True
#             return False
#         self.t_start_run = t_now
#         return True
    
    
#     def triggered_roll(self):
#         """
#             Squating to trigger Gesture Detection for rolling. Returns True if squatting, False otherwise.
#             Due to problems with false-positive when only True if all conditions are met, it now is voted
#             on and only returns True if the portion of true detections is above the vote_true_threshold.
#         """
#         if self.body is None:
#             return False
#         vote_true_threshold = 0.66
#         vote_results = []
#         # req_hip_floor_distance = 0.38  # minimal required closeness to floor to allow squat detection
#         # req_feet_hip_distance = 0.48  # minimal required closeness from feet to hips to allow squat detection
#         req_hip_floor_distance = 0.40  # minimal required closeness to floor to allow squat detection
#         req_feet_hip_distance = 0.60  # minimal required closeness from feet to hips to allow squat detection
#         or_below_sum_distance = 2.00  # Also trigger roll if the sum of distances is below this value

#         # Get floor and joints
#         left_foot = self.body.joints[PyKinectV2.JointType_FootLeft]
#         right_foot = self.body.joints[PyKinectV2.JointType_FootRight]
#         left_hip = self.body.joints[PyKinectV2.JointType_HipLeft]
#         right_hip = self.body.joints[PyKinectV2.JointType_HipRight]

#         # Log values
#         left_hip_distance_to_floor = self.get_floor_distance(left_hip)
#         right_hip_distance_to_floor = self.get_floor_distance(right_hip)
#         left_foot_hip_distance = self.get_joint_distance(left_foot, left_hip)
#         right_foot_hip_distance = self.get_joint_distance(right_foot, right_hip)
#         # print()
#         # print(f"self.roll_gesture_ready : {self.roll_gesture_ready}")
#         # print(f"dist: left_hip_distance_to_floor({left_hip_distance_to_floor:.3f}) < req_hip_floor_distance({req_hip_floor_distance:.3f}) == {left_hip_distance_to_floor < req_hip_floor_distance}")
#         # print(f"dist: right_hip_distance_to_floor({right_hip_distance_to_floor:.3f}) < req_hip_floor_distance({req_hip_floor_distance:.3f}) == {right_hip_distance_to_floor < req_hip_floor_distance}")
#         # print(f"dist: left_foot_hip_distance({left_foot_hip_distance:.3f}) < req_feet_hip_distance({req_feet_hip_distance:.3f}) == {left_foot_hip_distance < req_feet_hip_distance}")
#         # print(f"dist: right_foot_hip_distance({right_foot_hip_distance:.3f}) < req_feet_hip_distance({req_feet_hip_distance:.3f}) == {right_foot_hip_distance < req_feet_hip_distance}")
#         # distances_sum = sum([left_hip_distance_to_floor, right_hip_distance_to_floor, left_foot_hip_distance, right_foot_hip_distance])
#         # print(f"distances_sum({distances_sum:.3f}) < or_below_sum_distance({or_below_sum_distance:.3f}) == {distances_sum < or_below_sum_distance}")
#         # print()

#         # Check the roll/squat parameter conditions
#         vote_results.append(left_hip_distance_to_floor > req_hip_floor_distance)
#         vote_results.append(right_hip_distance_to_floor > req_hip_floor_distance)
#         vote_results.append(left_foot_hip_distance > req_feet_hip_distance)
#         vote_results.append(right_foot_hip_distance > req_feet_hip_distance)

#         # Make decision
#         true_votes = vote_results.count(True)
#         # Gesture has not been reset/made ready yet. Return False.
#         if not self.roll_gesture_ready:
#             # Check if gesture can be reset to be triggered again
#             if true_votes == 0:
#                 self.roll_gesture_ready = True
#             return False
#         # Calculate and Evaluate number of conditions that are met
#         true_votes_portion = true_votes / len(vote_results)
#         if true_votes_portion > vote_true_threshold:
#             self.roll_gesture_ready = False
#             return True
#         # Trigger anyways if the sum of distances is low
#         elif sum([left_hip_distance_to_floor, right_hip_distance_to_floor, left_foot_hip_distance, right_foot_hip_distance]) < or_below_sum_distance:
#             self.roll_gesture_ready = False
#             return True
#         return False
    

#     def triggered_interaction(self):
#         """
#             Track the right hand for a grab gesture. Grab is detected as the right hand
#             held forwards changing from opened to closed.
#         """
#         if self.body is None:
#             return False
#         target_hand_shoulder_distance = 0.40

#         # Get joints and states
#         right_hand_position = self.body.joints[PyKinectV2.JointType_HandRight].Position
#         right_shoulder_position = self.body.joints[PyKinectV2.JointType_ShoulderRight].Position
#         right_hand_state = self.body.hand_right_state

#         # Check if hand is held towards kinect
#         hand_shoulder_z_distance = right_shoulder_position.z - right_hand_position.z
#         if hand_shoulder_z_distance > target_hand_shoulder_distance:
#             # Check if hand is closed
#             if right_hand_state == PyKinectV2.HandState_Closed:
#                 # Check if hand was previously open
#                 if self.t_last_grab_open_state > self.t_last_grab_closed_state:
#                     # Hand is closed and was previously open
#                     self.t_last_grab_closed_state = time.time()
#                     return True
#                 self.t_last_grab_closed_state = time.time()
#             # Check if hand is open
#             elif right_hand_state == PyKinectV2.HandState_Open:
#                 self.t_last_grab_open_state = time.time()
#         # Hand is not held towards kinect or is not open or closed
#         return False


#     def triggered_light_attack(self):
#         """
#             Track the right hand for a light attack gesture. Gesture motion starts with the right hand above the
#             right shoulder and ends with the right hand close to the left hip with little time inbetween.
#         """
#         if self.body is None:
#             return False
        
#         # Position parameters
#         max_motion_duration = 0.6
#         hand_shoulder_bounds = {
#             "min_x" : -0.05, "max_x" : 0.25,
#             "min_y" : 0.0, "max_y" : 0.35,
#             "min_z" :  -0.25, "max_z" : 0.1,
#             "min_dist" : 0.1, "max_dist" : 0.45,
#         }
#         hand_hip_bounds = {
#             "min_x" : -0.2, "max_x" : 0.05,
#             "min_y" : 0.0, "max_y" : 0.15,
#             "min_z" :  -0.3, "max_z" : 0.0,
#             "min_dist" : 0.1, "max_dist" : 0.3,
#         }

#         # Get joint and distances
#         right_hand_position = self.body.joints[PyKinectV2.JointType_HandRight].Position
#         right_shoulder_position = self.body.joints[PyKinectV2.JointType_ShoulderRight].Position
#         left_hip_position = self.body.joints[PyKinectV2.JointType_HipLeft].Position

#         # print()
#         # print(f"self.body.hand_right_state==PyKinectV2.HandState_Closed  ==  {self.body.hand_right_state == PyKinectV2.HandState_Closed}")
#         # print(f"t since shoulder state == {time.time() - self.t_last_light_attack_shoulder_state:.3f}")
#         # print(f"t since hip state == {time.time() - self.t_last_light_attack_hip_state:.3f}")
#         # self.check_distances_log(right_hand_position, right_shoulder_position, **hand_shoulder_bounds, log_name="right hand to right shoulder")
#         # self.check_distances_log(right_hand_position, left_hip_position, **hand_hip_bounds, log_name="right hand to left hip")
#         # print()

#         # Check if right-hand above right-shoulder
#         if self.body.hand_right_state == PyKinectV2.HandState_Closed:
#             if self.check_distances(right_hand_position, right_shoulder_position, **hand_shoulder_bounds):
#                 self.t_last_light_attack_shoulder_state = time.time()
#                 seconds_since_other_state = self.t_last_light_attack_shoulder_state - self.t_last_light_attack_hip_state
#                 if seconds_since_other_state < max_motion_duration:
#                     self.t_last_light_attack_hip_state = 0.0  # Require other state to be detected again before triggering again
#                     return True 
#                 else:
#                     return False

#         # Check if right-hand on left-hip
#         if self.check_distances(right_hand_position, left_hip_position, **hand_hip_bounds):
#             self.t_last_light_attack_hip_state = time.time()
#             seconds_since_other_state = self.t_last_light_attack_hip_state - self.t_last_light_attack_shoulder_state
#             if seconds_since_other_state < max_motion_duration:
#                 self.t_last_light_attack_shoulder_state = 0.0  # Require other state to be detected again before triggering again
#                 return True
#             else:
#                 return False

#         return False


#     def triggered_heavy_attack(self):
#         """ 
#             Look for stabbing movements.
#             Right hand fist on right waist -> to right hand fist forwards
#         """
#         if self.body is None:
#             return False
        
#         # Get joint and distances
#         right_hand_position = self.body.joints[PyKinectV2.JointType_HandRight].Position
#         right_shoulder_position = self.body.joints[PyKinectV2.JointType_ShoulderRight].Position
#         right_hip_position = self.body.joints[PyKinectV2.JointType_HipRight].Position
#         hand_shoulder_x, hand_shoulder_y, hand_shoulder_z = right_hand_position.x - right_shoulder_position.x, right_hand_position.y - right_shoulder_position.y, right_hand_position.z - right_shoulder_position.z
#         hand_shoulder_dist = math.sqrt(hand_shoulder_x**2 + hand_shoulder_y**2 + hand_shoulder_z**2)
#         hand_hip_x, hand_hip_y, hand_hip_z = right_hand_position.x - right_hip_position.x, right_hand_position.y - right_hip_position.y, right_hand_position.z - right_hip_position.z
#         hand_hip_dist = math.sqrt(hand_hip_x**2 + hand_hip_y**2 + hand_hip_z**2)
 
#         # print()
#         # print("Heavy attack")
#         # print(f"self.body.hand_right_state:{self.body.hand_right_state}")  # 3 = closed
#         # print(f"hand_shoulder_x:{hand_shoulder_x:.3f} | hand_shoulder_y:{hand_shoulder_y:.3f} | hand_shoulder_z:{hand_shoulder_z:.3f} | hand_shoulder_dist:{hand_shoulder_dist:.3f}")
#         # print(f"hand_hip_x:{hand_hip_x:.3f} | hand_hip_y:{hand_hip_y:.3f} | hand_hip_z:{hand_hip_z:.3f} | hand_hip_dist:{hand_hip_dist:.3f}")
#         # print()

#         return False


#     def triggered_block(self):
#         """
#             Left hand in closed state in front of torso
#         """
#         if self.body is None:
#             return False
        
#         # distance parameters
#         hand_spine_bounds = {
#             "min_x" : -0.1, "max_x" : 0.1,
#             "min_y" : -0.32, "max_y" : -0.03,
#             "min_z" :  -0.35, "max_z" : 0.0,
#             "min_dist" : 0.1, "max_dist" : 0.4,
#         }
        
#         # Get joint and distances
#         spine_shoulder_pos = self.body.joints[PyKinectV2.JointType_SpineShoulder].Position
#         left_hand_pos = self.body.joints[PyKinectV2.JointType_HandLeft].Position
#         # left_hand_is_closed = self.body.hand_left_state == PyKinectV2.HandState_Closed
#         # rel_x, rel_y, rel_z, dist = self.get_rel_joint_distances(left_hand_pos, spine_shoulder_pos)

#         # print(f"left-hand to spine-shoulder: x:{rel_x:.3f} | y:{rel_y:.3f} | z:{rel_z:.3f} | dist:{dist:.3f} | left_hand_is_closed:{left_hand_is_closed}")

#         distances_in_bounds = self.check_distances(left_hand_pos, spine_shoulder_pos, **hand_spine_bounds)

#         # if distances_in_bounds:
#         #     print("BLOCKING")
#         # else:
#         #     print("not blocking")

#         return distances_in_bounds


#     def triggered_parry(self):
#         """
#             Left Hand on right hip -> to left hand above left shoulder
#         """
#         if self.body is None:
#             return False
        
#         # Position parameters
#         target_start_end_duration = 0.6
#         hand_hip_bounds = {
#             "min_x" : 0.0, "max_x" : 0.15,
#             "min_y" : -0.01, "max_y" : 0.12,
#             "min_z" :  -0.2, "max_z" : -0.03,
#             "min_dist" : None, "max_dist" : 0.2,
#         }
#         hand_shoulder_bounds = {
#             "min_x" : -0.40, "max_x" : -0.05,
#             "min_y" : 0.18, "max_y" : 0.35,
#             "min_z" :  -0.21, "max_z" : 0.18,
#             "min_dist" : 0.10, "max_dist" : 0.50,
#         }

#         # Get joint and distances
#         left_hand_pos = self.body.joints[PyKinectV2.JointType_HandLeft].Position
#         left_shoulder_pos = self.body.joints[PyKinectV2.JointType_ShoulderLeft].Position
#         right_hip_pos = self.body.joints[PyKinectV2.JointType_HipRight].Position
        
#         hand_hip_x, hand_hip_y, hand_hip_z, hand_hip_dist = self.get_rel_joint_distances(left_hand_pos, right_hip_pos)
#         hand_shoulder_x, hand_shoulder_y, hand_shoulder_z, hand_shoulder_dist = self.get_rel_joint_distances(left_hand_pos, left_shoulder_pos)

#         # print()
#         # print(f"left hand state : {self.body.hand_left_state == PyKinectV2.HandState_Closed}")
#         # print(f"lHand-rHip:  x:{hand_hip_x:.3f} | y:{hand_hip_y:.3f} | z:{hand_hip_z:.3f} | dist:{hand_hip_dist:.3f}")
#         # print(f"lHand-lShoulder:  x:{hand_shoulder_x:.3f} | y:{hand_shoulder_y:.3f} | z:{hand_shoulder_z:.3f} | dist:{hand_shoulder_dist:.3f}")
#         # print()

#         # Parry start (hand on hip)
#         if self.check_distances(left_hand_pos, right_hip_pos, **hand_hip_bounds):
#             self.t_last_parry_start = time.time()
#             return False

#         # Parry end (hand over shoulder)
#         if self.body.hand_left_state == PyKinectV2.HandState_Closed:  # Hand closed
#             seconds_since_parry_start = time.time() - self.t_last_parry_start
#             if seconds_since_parry_start < target_start_end_duration:  # Parry start within time limit
#                 if not self.check_distances(left_hand_pos, left_shoulder_pos, **hand_shoulder_bounds):  # Hand over shoulder
#                     self.t_last_parry_start = 0.0
#                     return True

#         return False


#     def triggered_lock_on(self):
#         """
#             Both open hands stretched out forwards. Hands fairly close together.
#             check_distances(self, joint1, joint2, min_x=None, max_x=None, min_y=None, max_y=None, min_z=None, max_z=None, min_dist=None, max_dist=None)
#         """
#         if self.body is None:
#             return False
        
#         # Get joint positions and hand states
#         # Left side
#         left_hand_position = self.body.joints[PyKinectV2.JointType_HandLeft].Position
#         left_hand_open = self.body.hand_left_state == PyKinectV2.HandState_Open
#         left_shoulder_position = self.body.joints[PyKinectV2.JointType_ShoulderLeft].Position
#         # Right side
#         right_hand_position = self.body.joints[PyKinectV2.JointType_HandRight].Position
#         right_hand_open = self.body.hand_right_state == PyKinectV2.HandState_Open
#         right_shoulder_position = self.body.joints[PyKinectV2.JointType_ShoulderRight].Position

#         # print("\nLock-On Gesture:")
#         # print(f"self.floor_plane xyzw :  {self.floor_plane.x:.5f}  |  {self.floor_plane.y:.5f}  |  {self.floor_plane.z:.5f}  |  {self.floor_plane.w:.5f}")
#         # # distance parameters
#         # hand_shoulder_bounds = {
#         #     "min_x" : -0.15, "max_x" : 0.15,
#         #     "min_y" : -0.15, "max_y" : 0.15,
#         #     "min_z" :  None, "max_z" : -0.35,
#         # }
#         # hand_bounds = {
#         #     "min_x" : -0.3, "max_x" : 0.0,
#         #     "min_y" : -0.1, "max_y" : 0.1,
#         #     "min_z" :  -0.1, "max_z" : 0.1,
#         #     "min_dist" :  None, "max_dist" : 0.35,
#         # }
#         # self.check_distances_log(left_hand_position, left_shoulder_position, log_name="Left-Hand to Left-Shoulder", **hand_shoulder_bounds)
#         # self.check_distances_log(right_hand_position, right_shoulder_position, log_name="Right-Hand to Right-Shoulder", **hand_shoulder_bounds)
#         # self.check_distances_log(left_hand_position, right_hand_position, log_name="Left-Hand to Right-Hand", **hand_bounds)
#         # print()

#         # Handle gesture reset
#         if not self.lock_on_gesture_ready:
#             hands_x, hands_y, hands_z, hands_dist = self.get_rel_joint_distances(left_hand_position, right_hand_position)
#             if abs(hands_x) > 0.35:
#                 self.lock_on_gesture_ready = True
#                 return False
#             if abs(hands_y) > 0.35:
#                 self.lock_on_gesture_ready = True
#                 return False
#             if abs(hands_z) > 0.35:
#                 self.lock_on_gesture_ready = True
#                 return False
#             if abs(hands_dist) > 0.45:
#                 self.lock_on_gesture_ready = True
#                 return False
#             left_hand_shoulder_x, left_hand_shoulder_y, left_hand_shoulder_z, left_hand_shoulder_dist = self.get_rel_joint_distances(left_hand_position, left_shoulder_position)
#             right_hand_shoulder_x, right_hand_shoulder_y, right_hand_shoulder_z, right_hand_shoulder_dist = self.get_rel_joint_distances(right_hand_position, right_shoulder_position)
#             if abs(left_hand_shoulder_y) > 0.25 or abs(right_hand_shoulder_y) > 0.25:
#                 self.lock_on_gesture_ready = True
#                 return False
#             if left_hand_shoulder_z > -0.25 or right_hand_shoulder_z > -0.25:
#                 self.lock_on_gesture_ready = True
#                 return False
#             return False
            

#         if not left_hand_open or not right_hand_open:
#             return False
        
#         # # distance parameters
#         # hand_shoulder_bounds = {
#         #     "min_x" : -0.15, "max_x" : 0.15,
#         #     "min_y" : -0.15, "max_y" : 0.15,
#         #     "min_z" :  None, "max_z" : -0.35,
#         # }
#         # hand_bounds = {
#         #     "min_x" : -0.3, "max_x" : 0.0,
#         #     "min_y" : -0.1, "max_y" : 0.1,
#         #     "min_z" :  -0.1, "max_z" : 0.1,
#         #     "min_dist" :  None, "max_dist" : 0.35,
#         # }
#         # distance parameters
#         hand_shoulder_bounds = {
#             "min_x" : -0.15, "max_x" : 0.15,
#             "min_y" : -0.15, "max_y" : 0.15,
#             "min_z" :  None, "max_z" : -0.35,
#         }
#         hand_bounds = {
#             "min_x" : -0.45, "max_x" : 0.0,
#             "min_y" : -0.1, "max_y" : 0.1,
#             "min_z" :  -0.1, "max_z" : 0.1,
#             "min_dist" :  None, "max_dist" : 0.45,
#         }

#         # Check if all the criteria for the lock-on gesture are met
#         # Left-Hand to Left-Shoulder
#         if not self.check_distances(left_hand_position, left_shoulder_position, **hand_shoulder_bounds):
#             return False
#         # Right-Hand to Right-Shoulder
#         if not self.check_distances(right_hand_position, right_shoulder_position, **hand_shoulder_bounds):
#             return False
#         # Left-Hand to Right-Hand
#         if not self.check_distances(left_hand_position, right_hand_position, **hand_bounds):
#             return False

#         self.lock_on_gesture_ready = False
#         return True  # All the criteria for the lock-on gesture are met
    

#     def triggered_kick(self):
#         """
#             Right foot raised above floor and held towards kinect away from right
#             hip with the left foot on the floor.
#         """
#         if self.body is None:
#             return False
        
#         right_foot_pos = self.body.joints[PyKinectV2.JointType_FootRight].Position
#         left_foot_pos = self.body.joints[PyKinectV2.JointType_FootLeft].Position
#         right_hip_pos = self.body.joints[PyKinectV2.JointType_HipRight].Position

#         right_foot_floor_dist = self.get_floor_distance(right_foot_pos)
#         left_foot_floor_dist = self.get_floor_distance(left_foot_pos)
#         feet_x, feet_y, feet_z, feet_dist = self.get_rel_joint_distances(right_foot_pos, left_foot_pos)
#         # right_foot_hip_x, right_foot_hip_y, right_foot_hip_z, right_foot_hip_dist = self.get_rel_joint_distances(right_foot_pos, right_hip_pos)

#         # print()
#         # print(f"feet floor dist:  L:{left_foot_floor_dist:.3f}   R:{right_foot_floor_dist:.3f}")
#         # print(f"feet:   x:{feet_x:.3f}  y:{feet_y:.3f}  z:{feet_z:.3f}  dist:{feet_dist:.3f}")
#         # print(f"right_foot_hip:   x:{right_foot_hip_x:.3f}  y:{right_foot_hip_y:.3f}  z:{right_foot_hip_z:.3f}  dist:{right_foot_hip_dist:.3f}")
#         # print()

#         if not self.kick_gesture_ready:
#             # Check if the right foot is raised above the floor
#             if right_foot_floor_dist < 0.4:
#                 self.kick_gesture_ready = True
#                 return False
#             # Check if the left foot is on the floor
#             if feet_dist < 0.5:
#                 self.kick_gesture_ready = True
#                 return False
#             return False

#         # distance parameters
#         right_foot_floor_dist_min = 0.6
#         left_foot_floor_dist_max = 0.1
#         feet_bounds = {
#             # "min_z" : None, "max_z" : -0.4,
#             # "min_dist" : 0.7, "max_dist" : None,
#         }
#         foot_hip_bounds = {
#             "min_y" : -0.35, "max_y" : 0.35,
#             "min_dist" :  0.5, "max_dist" : None,
#         }

#         # Check if all the criteria for the lock-on gesture are met
#         if right_foot_floor_dist < right_foot_floor_dist_min:
#             return False  # Kick foot too low
#         if left_foot_floor_dist > left_foot_floor_dist_max:
#             return False  # Stand foot in air
        
#         if not self.check_distances(right_foot_pos, left_foot_pos, **feet_bounds):
#             return False
#         if not self.check_distances(right_foot_pos, right_hip_pos, **foot_hip_bounds):
#             return False
        
#         self.kick_gesture_ready = False
#         return True  # All the criteria for the lock-on gesture are met

    
#     def test_align_points_with_floor(self):
#         """Test if we can convert our points into a stable coordinate system where the xz-plane forms a plane parallel to the floor."""
#         if self.body is None:
#             return False
        
#         # Get original hand positions
#         original_left_hand_position = self.body.joints[PyKinectV2.JointType_HandLeft].Position
#         original_left_hand_position = original_left_hand_position.x, original_left_hand_position.y, original_left_hand_position.z
#         original_right_hand_position = self.body.joints[PyKinectV2.JointType_HandRight].Position
#         original_right_hand_position = original_right_hand_position.x, original_right_hand_position.y, original_right_hand_position.z
#         original_points = [original_left_hand_position, original_right_hand_position]
#         original_rel_x, original_rel_y, original_rel_z, original_rel_dist = self.get_rel_joint_distances(original_left_hand_position, original_right_hand_position)

#         # Get rotated hand positions (hopefully aligned with the floor plane)
#         rotated_points = self.align_points_with_floor(original_points)
#         rotated_left_hand_position, rotated_right_hand_position = rotated_points
#         rotated_rel_x, rotated_rel_y, rotated_rel_z, rotated_rel_dist = self.get_rel_joint_distances(rotated_left_hand_position, rotated_right_hand_position)

#         # Print results 
#         print()
#         print("Rotating points (hand joints) to hopefully align coordinate system with floor plane")
#         print()
#         print(f"self.floor_plane:                  x : {self.floor_plane.x:.3f}  |  y : {self.floor_plane.y:.3f}  |  z : {self.floor_plane.z:.3f}  |  w : {self.floor_plane.w:.3f}")
#         print()
#         print(f"original right hand:               x : {original_right_hand_position[0]:.3f}  |  y : {original_right_hand_position[1]:.3f}  |  z : {original_right_hand_position[2]:.3f}")
#         print(f"original left hand:                x : {original_left_hand_position[0]:.3f}  |  y : {original_left_hand_position[1]:.3f}  |  z : {original_left_hand_position[2]:.3f}")
#         print(f"original distances left to right:  x : {original_rel_x:.3f}  |  y : {original_rel_y:.3f}  |  z : {original_rel_z:.3f}  |  dist : {original_rel_dist:.3f}")
#         print()
#         print(f"rotated right hand:                x : {rotated_right_hand_position[0]:.3f}  |  y : {rotated_right_hand_position[1]:.3f}  |  z : {rotated_right_hand_position[2]:.3f}")
#         print(f"rotated left hand:                 x : {rotated_left_hand_position[0]:.3f}  |  y : {rotated_left_hand_position[1]:.3f}  |  z : {rotated_left_hand_position[2]:.3f}")
#         print(f"rotated distances left to right:   x : {rotated_rel_x:.3f}  |  y : {rotated_rel_y:.3f}  |  z : {rotated_rel_z:.3f}  |  dist : {rotated_rel_dist:.3f}")
#         print()  
#         print("-" * 100)


#     def test_align_points_with_upper_body(self):
#         """Test if we can convert our points into a stable coordinate system where the xz-plane forms a plane parallel to the floor."""
#         if self.body is None:
#             return False
        
#         # Get original hand positions
#         original_right_shoulder_position = self.body.joints[PyKinectV2.JointType_ShoulderRight].Position
#         original_right_shoulder_position = original_right_shoulder_position.x, original_right_shoulder_position.y, original_right_shoulder_position.z
#         original_right_hand_position = self.body.joints[PyKinectV2.JointType_HandRight].Position
#         original_right_hand_position = original_right_hand_position.x, original_right_hand_position.y, original_right_hand_position.z
#         original_points = [original_right_shoulder_position, original_right_hand_position]
#         original_rel_x, original_rel_y, original_rel_z, original_rel_dist = self.get_rel_joint_distances(original_right_hand_position, original_right_shoulder_position)

#         # Get rotated hand positions (hopefully aligned with the floor plane)
#         aligned_points = self.align_points_with_upper_body(original_points)
#         aligned_right_shoulder_position, aligned_right_hand_position = aligned_points
#         aligned_rel_x, aligned_rel_y, aligned_rel_z, aligned_rel_dist = self.get_rel_joint_distances(aligned_right_hand_position, aligned_right_shoulder_position)

#         # Print results
#         lean_x, lean_y = self.get_lean()
#         from_original_to_aligned = np.array([aligned_rel_x-original_rel_x, aligned_rel_y-original_rel_y, aligned_rel_z-original_rel_z, aligned_rel_dist-original_rel_dist])
#         print()
#         print("test_align_points_with_upper_body (align with spine first and then with shoulders)")
#         print()
#         print(f"self.lean:                            x : {lean_x:.3f}  |  y : {lean_y:.3f}")
#         print()
#         print(f"original right hand:                  x : {original_right_hand_position[0]:.3f}  |  y : {original_right_hand_position[1]:.3f}  |  z : {original_right_hand_position[2]:.3f}")
#         print(f"original right shoulder:              x : {original_right_shoulder_position[0]:.3f}  |  y : {original_right_shoulder_position[1]:.3f}  |  z : {original_right_shoulder_position[2]:.3f}")
#         print(f"original distances shoulder to hand:  x : {original_rel_x:.3f}  |  y : {original_rel_y:.3f}  |  z : {original_rel_z:.3f}  |  dist : {original_rel_dist:.3f}")
#         print()
#         print(f"aligned right hand:                   x : {aligned_right_hand_position[0]:.3f}  |  y : {aligned_right_hand_position[1]:.3f}  |  z : {aligned_right_hand_position[2]:.3f}")
#         print(f"aligned right shoulder:               x : {aligned_right_shoulder_position[0]:.3f}  |  y : {aligned_right_shoulder_position[1]:.3f}  |  z : {aligned_right_shoulder_position[2]:.3f}")
#         print(f"aligned distances shoulder to hand:   x : {aligned_rel_x:.3f}  |  y : {aligned_rel_y:.3f}  |  z : {aligned_rel_z:.3f}  |  dist : {aligned_rel_dist:.3f}")
#         print()
#         print(f"from_original_to_aligned:             x : {from_original_to_aligned[0]:.3f}  |  y : {from_original_to_aligned[1]:.3f}  |  z : {from_original_to_aligned[2]:.3f}  |  dist : {from_original_to_aligned[3]:.3f}")
#         print()
#         print("-" * 100)


#     def display_cam_cv2(self, resize_dim=(1280, 720), mirror=False):
#         """Display the Kinect's RGB video feed with cv2."""
#         sleep_duration = 1.0 / 200.0
#         last_color_frame_num = 0
#         while True:
#             if self.color_frame_num > last_color_frame_num:
#                 last_color_frame_num = self.color_frame_num
#                 frame = self.color_frame.reshape((1080, 1920, 4))[:, :, :3]
#                 if not mirror:
#                     frame = frame[:, ::-1, :]

#                 # Display the RGB video feed with cv2
#                 cv2.imshow('RGB', frame)
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     exit()
#             time.sleep(sleep_duration)


#     def display_cam(self, resize_dim=(1280, 720), mirror:bool=False, display_type:str="cv2"):
#         """Call methode to display the Kinect's RGB video feed with the specified display_type methode"""
#         display_type = display_type.strip().lower()
#         if display_type == "cv2":
#             self.display_cam_cv2(resize_dim, mirror)
#         elif display_type == "pygame":
#             self.display_cam_pygame(resize_dim, mirror)
        


# # ==================================================================================================



# def show_cam(resize_dim=(1280, 720), mirror=False):
#     while True:
#         # Get the current RGB video frame from the Kinect sensor
#         if kinect.has_new_color_frame():
#             frame = kinect.get_last_color_frame()
#             frame = frame.reshape((1080, 1920, 4))[:, ::, :3]
#             if not mirror:
#                 frame = frame[:, ::-1, :]  # Reverse rows of pixels to make in unmirrored
#             if resize_dim is not None:
#                 frame = cv2.resize(frame, resize_dim, interpolation=cv2.INTER_AREA)

#             # Display the RGB video feed with cv2
#             cv2.imshow('RGB', frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#         else:
#             time.sleep(0.003)


# def show_camera(*args, **kwargs):
#     import numpy as np
#     import pygame
#     from pykinect2 import PyKinectV2
    
#     BONE_CONNECTIONS = [(PyKinectV2.JointType_Head, PyKinectV2.JointType_Neck),
#                         (PyKinectV2.JointType_Neck, PyKinectV2.JointType_SpineShoulder),
#                         (PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_SpineMid),
#                         (PyKinectV2.JointType_SpineMid, PyKinectV2.JointType_SpineBase),
#                         (PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderRight),
#                         (PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderLeft),
#                         (PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipRight),
#                         (PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipLeft),
#                         (PyKinectV2.JointType_ShoulderRight, PyKinectV2.JointType_ElbowRight),
#                         (PyKinectV2.JointType_ElbowRight, PyKinectV2.JointType_WristRight),
#                         (PyKinectV2.JointType_WristRight, PyKinectV2.JointType_HandRight),
#                         (PyKinectV2.JointType_ShoulderLeft, PyKinectV2.JointType_ElbowLeft),
#                         (PyKinectV2.JointType_ElbowLeft, PyKinectV2.JointType_WristLeft),
#                         (PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_HandLeft),
#                         (PyKinectV2.JointType_HipRight, PyKinectV2.JointType_KneeRight),
#                         (PyKinectV2.JointType_KneeRight, PyKinectV2.JointType_AnkleRight),
#                         (PyKinectV2.JointType_AnkleRight, PyKinectV2.JointType_FootRight),
#                         (PyKinectV2.JointType_HipLeft, PyKinectV2.JointType_KneeLeft),
#                         (PyKinectV2.JointType_KneeLeft, PyKinectV2.JointType_AnkleLeft),
#                         (PyKinectV2.JointType_AnkleLeft, PyKinectV2.JointType_FootLeft),
#                         (PyKinectV2.JointType_HandTipRight, PyKinectV2.JointType_HandRight),
#                         (PyKinectV2.JointType_ThumbRight, PyKinectV2.JointType_WristRight),
#                         (PyKinectV2.JointType_HandTipLeft, PyKinectV2.JointType_HandLeft),
#                         (PyKinectV2.JointType_ThumbLeft, PyKinectV2.JointType_WristLeft)]

#     # Constants
#     WINDOW_TITLE = "Kinect Webcam with Skeleton Tracking"
    
#     # Initialize Pygame
#     pygame.init()
#     screen_width, screen_height = kinect.color_frame_desc.Width, kinect.color_frame_desc.Height
#     screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
#     pygame.display.set_caption(WINDOW_TITLE)

#     # Colors for each tracked skeleton
#     skeleton_colors = [pygame.color.THECOLORS["red"],
#                     pygame.color.THECOLORS["blue"],
#                     pygame.color.THECOLORS["green"],
#                     pygame.color.THECOLORS["orange"],
#                     pygame.color.THECOLORS["purple"],
#                     pygame.color.THECOLORS["yellow"],
#                     ]

#     # Function to get and process the Kinect color frame
#     def get_color_frame():
#         if kinect.has_new_color_frame():
#             frame = kinect.get_last_color_frame()
#             frame_reshaped = frame.reshape((kinect.color_frame_desc.Height, kinect.color_frame_desc.Width, 4))[:,:,:3]
#             frame_reshaped = np.transpose(frame_reshaped, (1, 0, 2))[:,:,::-1]  # Transpose the array for proper dimensions
#             frame_surface = pygame.surfarray.make_surface(frame_reshaped)
#             frame_surface = pygame.transform.scale(frame_surface,
#                                                 (screen_width, screen_height))
#             return frame_surface
#         else:
#             return None

#     # Function to draw the skeletons on the screen
#     def draw_skeletons(bodies, skeleton_colors=skeleton_colors, single_skeleton_color=False, show_handstate=True, *args, **kwargs):
#         # Define the bone connections (bones in a human skeleton)

#         for i in range(kinect.max_body_count):
#             body = bodies.bodies[i]
#             print(f"single_skeleton_color=={single_skeleton_color}")
#             if single_skeleton_color is False or single_skeleton_color is None:
#                 skel_color = skeleton_colors[i]
#             else:
#                 skel_color = single_skeleton_color

#             if not body.is_tracked:
#                 continue

#             joint_points = kinect.body_joints_to_color_space(body.joints)

#             for joint_index in range(25):
#                 x = joint_points[joint_index].x
#                 y = joint_points[joint_index].y

#                 if x < 0 or x > screen_width or y < 0 or y > screen_height:
#                     continue

#                 # Draw the joints as circles
#                 pygame.draw.circle(screen, skel_color, (int(x), int(y)), 8, width=0)

#             # Draw the hand states as filled partly transparent circles
#             if joint_index in (PyKinectV2.JointType_HandLeft, PyKinectV2.JointType_HandRight):
#                 hand_state = body.hand_left_state if joint_index == PyKinectV2.JointType_HandLeft else body.hand_right_state
#                 state_color = None

#                 if hand_state == PyKinectV2.HandState_Open:
#                     state_color = (0, 255, 0)  # Green for open hand state
#                 elif hand_state == PyKinectV2.HandState_Closed:
#                     state_color = (255, 0, 0)  # Red for closed hand state
#                 elif hand_state == PyKinectV2.HandState_Lasso:
#                     state_color = (0, 0, 255)  # Blue for lasso hand state

#                 if state_color is not None:
#                     pygame.draw.circle(screen, state_color, (int(x), int(y)), 12, width=0)


#             # Draw the bones as lines
#             for connection in BONE_CONNECTIONS:
#                 joint_1 = connection[0]
#                 joint_2 = connection[1]

#                 if body.joints[joint_1].TrackingState == PyKinectV2.TrackingState_NotTracked or body.joints[joint_2].TrackingState == PyKinectV2.TrackingState_NotTracked:
#                     continue
#                 x1 = joint_points[joint_1].x
#                 y1 = joint_points[joint_1].y
#                 if x1 < 0 or x1 > screen_width or y1 < 0 or y1 > screen_height:
#                     continue
#                 x2 = joint_points[joint_2].x
#                 y2 = joint_points[joint_2].y
#                 if x2 < 0 or x2 > screen_width or y2 < 0 or y2 > screen_height:
#                     continue

#                 pygame.draw.line(screen, skel_color, (int(x1), int(y1)), (int(x2), int(y2)), 4)


#     def draw_skeletons(bodies, skeleton_colors=skeleton_colors, single_skeleton_color=False, show_handstate=True, *args, **kwargs):
#         # Define the bone connections (bones in a human skeleton)

#         for i in range(kinect.max_body_count):
#             body = bodies.bodies[i]
#             if single_skeleton_color is False or single_skeleton_color is None:
#                 skel_color = skeleton_colors[i]
#             else:
#                 skel_color = single_skeleton_color

#             if not body.is_tracked:
#                 continue

#             joint_points = kinect.body_joints_to_color_space(body.joints)

#             for joint_index in range(25):
#                 x = joint_points[joint_index].x
#                 y = joint_points[joint_index].y

#                 if x < 0 or x > screen_width or y < 0 or y > screen_height:
#                     continue

#                 # Draw the hand states as filled partly transparent circles
#                 if show_handstate and joint_index in (PyKinectV2.JointType_HandLeft, PyKinectV2.JointType_HandRight):
#                     hand_state = body.hand_left_state if joint_index == PyKinectV2.JointType_HandLeft else body.hand_right_state
#                     state_color = None
#                     state_radius = 20
#                     state_opacity = 255

#                     if hand_state == PyKinectV2.HandState_Open:
#                         state_color = (70, 255, 70)  # Green for open hand state
#                     elif hand_state == PyKinectV2.HandState_Closed:
#                         state_color = (255, 70, 70)  # Red for closed hand state
#                     elif hand_state == PyKinectV2.HandState_Lasso:
#                         state_color = (70, 70, 255)  # Blue for lasso hand state

#                     if state_color is not None:
#                         # Create a surface with alpha channel
#                         state_surface = pygame.Surface((state_radius*2, state_radius*2), pygame.SRCALPHA)
#                         # Draw a filled circle on the surface
#                         pygame.draw.circle(state_surface, (*state_color, state_opacity), (state_radius, state_radius), state_radius)
#                         # Blit the surface onto the main surface
#                         screen.blit(state_surface, (int(x) - state_radius, int(y) - state_radius))
                
#                 # Draw the joints as circles
#                 pygame.draw.circle(screen, skel_color, (int(x), int(y)), 8, width=0)

#             # Draw the bones as lines
#             for connection in BONE_CONNECTIONS:
#                 joint_1 = connection[0]
#                 joint_2 = connection[1]

#                 if body.joints[joint_1].TrackingState == PyKinectV2.TrackingState_NotTracked or body.joints[joint_2].TrackingState == PyKinectV2.TrackingState_NotTracked:
#                     continue
#                 x1 = joint_points[joint_1].x
#                 y1 = joint_points[joint_1].y
#                 if x1 < 0 or x1 > screen_width or y1 < 0 or y1 > screen_height:
#                     continue
#                 x2 = joint_points[joint_2].x
#                 y2 = joint_points[joint_2].y

#                 if x2 < 0 or x2 > screen_width or y2 < 0 or y2 > screen_height:
#                     continue

#                 pygame.draw.line(screen, skel_color, (int(x1), int(y1)), (int(x2), int(y2)), 4)


#     def main_camera_loop(show_skeletons=True, toggle_skeletons_key=pygame.K_SPACE,  *args, **kwargs):
#         # Main game loop
#         sleep_duration = 1.0 / 100
#         while True:
#             # Handle events
#             time.sleep(sleep_duration)
#             for event in pygame.event.get():
#                 if event.type == pygame.QUIT:
#                     pygame.quit()
#                     exit()

#                 if event.type == pygame.KEYUP:
#                     if event.key == toggle_skeletons_key:
#                         show_skeletons = not show_skeletons

#             # Get and process the Kinect color frame
#             frame_surface = get_color_frame()

#             if frame_surface is None:
#                 continue
            
#             if frame_surface is not None:
#                 # Draw the color frame on the screen
#                 screen.blit(frame_surface, (0, 0))

#             # Get Kinect body frame
#             if not show_skeletons:
#                 pygame.display.update()
#                 continue
#             if not kinect.has_new_body_frame():
#                 continue
#             bodies = kinect.get_last_body_frame()
#             draw_skeletons(bodies, *args, **kwargs)

#             # Update the screen
#             pygame.display.update()

#     main_camera_loop(*args, **kwargs)


# def testy1():
#     import threading
#     cam_thread = threading.Thread(target=show_cam, args={}, kwargs={})
#     cam_thread.start()
#     kinect_detector = KinectDetector()
#     while True:
#         time.sleep(0.1)
#         kinect_detector.get_actions(update_body_frame=True)


# def testy2():
#     import threading
#     cam_thread = threading.Thread(target=show_camera, args={}, kwargs={"single_skeleton_color": (120, 20, 120)})
#     cam_thread.start()
#     kinect_detector = KinectDetector()
#     while True:
#         time.sleep(0.05)
#         if not cam_thread.is_alive():
#             exit()
#         kinect_detector.get_actions(update_body_frame=True)



# if __name__ == "__main__":
#     testy2()

# if __name__ == "__main__":
#     print(f"\n\n    Finished Script '{os.path.basename(__file__)}' at {time.strftime('%Y-%m-%d_%H-%M-%S')}    \n\n")