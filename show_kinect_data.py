import os
import time
import math

import numpy as np
import cv2
from pykinect2 import PyKinectV2, PyKinectRuntime


kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body)


def show_camera(window_size=(1280, 720), mirror:bool=False, display_type:str="pygame", *args, **kwargs):
    """Call methode to display the Kinect's RGB video feed with the specified display_type methode"""
    display_type = display_type.strip().lower()
    if display_type == "cv2":
        show_camera_cv2(window_size, mirror, *args, **kwargs)
    elif display_type == "pygame":
        show_camera_pygame(window_size, mirror, *args, **kwargs)
    else:
        raise ValueError(f"Invalid display_type: {display_type}")


def show_camera_cv2(window_size=(1280, 720), mirror:bool=False, show_fps=True):
    sleep_duration = 1.0 / 200
    fps_tracker = FpsTracker()
    while True:
        # Get the current RGB video frame from the Kinect sensor
        if kinect.has_new_color_frame():
            frame = kinect.get_last_color_frame()
            frame = frame.reshape((1080, 1920, 4))[:, ::, :3]
            if not mirror:
                frame = frame[:, ::-1, :]  # Reverse rows of pixels to make in unmirrored
            if window_size is not None:
                frame = cv2.resize(frame, window_size, interpolation=cv2.INTER_AREA)
            if show_fps:
                current_fps = fps_tracker.tick()
                print(f"{current_fps:.2f} FPS")

            # Display the RGB video feed with cv2
            cv2.imshow('RGB', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        time.sleep(sleep_duration)


def show_camera_pygame(window_size=(1280, 720), mirror:bool=False, show_fps=True, *args, **kwargs):
    # Mirror not implemented yet, cause it would require some bigger changes
    import numpy as np
    import pygame
    from pykinect2 import PyKinectV2
    

    BONE_CONNECTIONS = [(PyKinectV2.JointType_Head, PyKinectV2.JointType_Neck),
                        (PyKinectV2.JointType_Neck, PyKinectV2.JointType_SpineShoulder),
                        (PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_SpineMid),
                        (PyKinectV2.JointType_SpineMid, PyKinectV2.JointType_SpineBase),
                        (PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderRight),
                        (PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderLeft),
                        (PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipRight),
                        (PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipLeft),
                        (PyKinectV2.JointType_ShoulderRight, PyKinectV2.JointType_ElbowRight),
                        (PyKinectV2.JointType_ElbowRight, PyKinectV2.JointType_WristRight),
                        (PyKinectV2.JointType_WristRight, PyKinectV2.JointType_HandRight),
                        (PyKinectV2.JointType_ShoulderLeft, PyKinectV2.JointType_ElbowLeft),
                        (PyKinectV2.JointType_ElbowLeft, PyKinectV2.JointType_WristLeft),
                        (PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_HandLeft),
                        (PyKinectV2.JointType_HipRight, PyKinectV2.JointType_KneeRight),
                        (PyKinectV2.JointType_KneeRight, PyKinectV2.JointType_AnkleRight),
                        (PyKinectV2.JointType_AnkleRight, PyKinectV2.JointType_FootRight),
                        (PyKinectV2.JointType_HipLeft, PyKinectV2.JointType_KneeLeft),
                        (PyKinectV2.JointType_KneeLeft, PyKinectV2.JointType_AnkleLeft),
                        (PyKinectV2.JointType_AnkleLeft, PyKinectV2.JointType_FootLeft),
                        (PyKinectV2.JointType_HandTipRight, PyKinectV2.JointType_HandRight),
                        (PyKinectV2.JointType_ThumbRight, PyKinectV2.JointType_WristRight),
                        (PyKinectV2.JointType_HandTipLeft, PyKinectV2.JointType_HandLeft),
                        (PyKinectV2.JointType_ThumbLeft, PyKinectV2.JointType_WristLeft)]

    # Constants
    WINDOW_TITLE = "Kinect Webcam with Skeleton Tracking"
    
    # Initialize Pygame
    pygame.init()
    camera_width, camera_height = kinect.color_frame_desc.Width, kinect.color_frame_desc.Height
    intermediate_screen = pygame.Surface((camera_width, camera_height),  pygame.SRCALPHA, 32)
    displayed_screen = pygame.display.set_mode(window_size, pygame.RESIZABLE)
    pygame.display.set_caption(WINDOW_TITLE)

    # Colors for each tracked skeleton
    skeleton_colors = [pygame.color.THECOLORS["red"],
                    pygame.color.THECOLORS["blue"],
                    pygame.color.THECOLORS["green"],
                    pygame.color.THECOLORS["orange"],
                    pygame.color.THECOLORS["purple"],
                    pygame.color.THECOLORS["yellow"],
                    ]

    # Function to get and process the Kinect color frame
    def get_color_frame():
        if kinect.has_new_color_frame():
            frame = kinect.get_last_color_frame()
            frame_reshaped = frame.reshape((kinect.color_frame_desc.Height, kinect.color_frame_desc.Width, 4))[:,:,:3]
            frame_reshaped = np.transpose(frame_reshaped, (1, 0, 2))[:,:,::-1]  # Transpose the array for proper dimensions
            frame_surface = pygame.surfarray.make_surface(frame_reshaped)
            return frame_surface
        return None

    # Function to draw the skeletons on the screen
    def draw_skeletons(bodies, skeleton_colors=skeleton_colors, single_skeleton_color=False, show_handstate=True, *args, **kwargs):
        for i in range(kinect.max_body_count):
            body = bodies.bodies[i]
            if single_skeleton_color is False or single_skeleton_color is None:
                skel_color = skeleton_colors[i]
            else:
                skel_color = single_skeleton_color

            if not body.is_tracked:
                continue

            joint_points = kinect.body_joints_to_color_space(body.joints)

            for joint_index in range(25):
                x = joint_points[joint_index].x
                y = joint_points[joint_index].y

                if x < 0 or x > camera_width or y < 0 or y > camera_height:
                    continue

                # Draw the hand states as filled partly transparent circles
                if show_handstate and joint_index in (PyKinectV2.JointType_HandLeft, PyKinectV2.JointType_HandRight):
                    hand_state = body.hand_left_state if joint_index == PyKinectV2.JointType_HandLeft else body.hand_right_state
                    state_color = None

                    if hand_state == PyKinectV2.HandState_Open:
                        state_color = (50, 255, 50)  # Green for open hand state
                    elif hand_state == PyKinectV2.HandState_Closed:
                        state_color = (255, 50, 50)  # Red for closed hand state
                    elif hand_state == PyKinectV2.HandState_Lasso:
                        state_color = (50, 50, 255)  # Blue for lasso hand state

                    if state_color is not None:
                        pygame.draw.circle(intermediate_screen, state_color, (int(x), int(y)), 16, width=0)

                # Draw the joints as circles
                pygame.draw.circle(intermediate_screen, skel_color, (int(x), int(y)), 8, width=0)

            # Draw the bones as lines
            for connection in BONE_CONNECTIONS:
                joint_1 = connection[0]
                joint_2 = connection[1]

                if body.joints[joint_1].TrackingState == PyKinectV2.TrackingState_NotTracked or body.joints[joint_2].TrackingState == PyKinectV2.TrackingState_NotTracked:
                    continue
                x1 = joint_points[joint_1].x
                y1 = joint_points[joint_1].y
                if x1 < 0 or x1 > camera_width or y1 < 0 or y1 > camera_height:
                    continue
                x2 = joint_points[joint_2].x
                y2 = joint_points[joint_2].y
                if x2 < 0 or x2 > camera_width or y2 < 0 or y2 > camera_height:
                    continue

                pygame.draw.line(intermediate_screen, skel_color, (int(x1), int(y1)), (int(x2), int(y2)), 4)


    def main_camera_loop(show_skeletons=True, toggle_skeletons_key=pygame.K_SPACE,  *args, **kwargs):
        # Main game loop
        nonlocal intermediate_screen
        fps_tracker = FpsTracker(max_samples=20)
        sleep_duration = 1.0 / 100.0
        while True:
            time.sleep(sleep_duration)

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

                if event.type == pygame.KEYUP:
                    if event.key == toggle_skeletons_key:
                        show_skeletons = not show_skeletons

            if not (kinect.has_new_color_frame() and kinect.has_new_body_frame()):
                continue

            # Get and process the Kinect color frame
            frame_surface = get_color_frame()

            if frame_surface is not None:
                # Draw the color frame on the screen
                intermediate_screen.blit(frame_surface, (0, 0))

            # Get Kinect body frame
            if show_skeletons:
                if not kinect.has_new_body_frame():
                    continue
                bodies = kinect.get_last_body_frame()
                draw_skeletons(bodies, *args, **kwargs)

            # Undo default mirroring if user doesn't want mirroring
            if not mirror:
                intermediate_screen = pygame.transform.flip(intermediate_screen, True, False)
            # Apply changes and Update the displayed_screen 
            pygame.transform.scale(intermediate_screen, window_size, displayed_screen)
            # Add text for fps
            if show_fps:
                current_fps = fps_tracker.tick()
                font = pygame.font.Font(None, 24)
                text_string = f"{current_fps:.0f} FPS"
                text = font.render(text_string, True, (0, 0, 0))
                text_rect = text.get_rect()
                # displayed_screen.blit(text, text_rect)
                displayed_screen.blit(text, (5, 5))
            # Update display
            pygame.display.update()

    main_camera_loop(*args, **kwargs)


def display_kinect_color_frame(resolution):
    import pygame
    # Initialize Kinect runtime object
    kinect = PyKinectRuntime.PyKinectRuntime(PyKinectRuntime.FrameSourceTypes_Color)

    # Initialize Pygame
    pygame.init()

    # Set the resolution of the color frame
    color_frame_width, color_frame_height = kinect.color_frame_desc.Width, kinect.color_frame_desc.Height
    display_width, display_height = resolution

    # Create a Pygame display surface with the desired window size
    screen = pygame.display.set_mode((display_width, display_height))

    # Set up the main game loop
    while True:
        # Check for Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                kinect.close()
                pygame.quit()
                return

        # Get the latest color frame from the Kinect sensor
        if kinect.has_new_color_frame():
            frame = kinect.get_last_color_frame()
            frame = frame.reshape((color_frame_height, color_frame_width, 4))[:,:,:3]
            frame = np.transpose(frame, (1, 0, 2))[:,:,::-1]  # Transpose the array for proper dimensions

            # Convert the raw frame data to a format Pygame can handle
            frame_surface = pygame.surfarray.make_surface(frame)

            # Scale the color frame to the desired resolution and blit it onto the display surface
            pygame.transform.scale(frame_surface, (display_width, display_height), screen)

            # Update the screen
            pygame.display.update()


class FpsTracker:
    def __init__(self, max_samples=10, min_samples=3, default_fps=-1):
        self.max_samples = max_samples
        self.min_samples = min_samples
        self.frame_durations = []
        self.last_time = time.time()
        self.default_fps = default_fps
        self.current_fps = default_fps
    

    def tick(self) -> float:
        current_time = time.time()
        frame_duration = current_time - self.last_time
        self.last_time = current_time

        self.frame_durations.append(frame_duration)
        if len(self.frame_durations) > self.max_samples:
            self.frame_durations.pop(0)

        if len(self.frame_durations) >= self.min_samples:
            self.current_fps = 1.0 / np.mean(self.frame_durations)
        else:
            self.current_fps = self.default_fps

        return self.current_fps
    

    def get_fps(self) -> float:
        return self.current_fps


def test_pygame_camera():
    show_camera(display_type="pygame", window_size=(1280, 720), single_skeleton_color=(120, 20, 120), show_fps=True)


def test_cv2_camera():
    show_camera(display_type="cv2")


def test_chatgpt_scaled_pygame():
    # display_kinect_color_frame((640, 360))
    display_kinect_color_frame((1280, 720))


if __name__ == "__main__":
    test_pygame_camera()
    # test_cv2_camera()
    # test_chatgpt_scaled_pygame()

if __name__ == "__main__":
    print(f"\n\n    Finished Script '{os.path.basename(__file__)}' at {time.strftime('%Y-%m-%d_%H-%M-%S')}    \n\n")