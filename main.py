import os
import time
import math

import numpy as np
import cv2
import keyboard
import vgamepad as vg
from pykinect2 import PyKinectV2, PyKinectRuntime

from kinect_detector import KinectDetector
from show_kinect_data import show_camera


kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body)


class KinectGamepadController:
    def __init__(self, fps=20, pause_key="p"):
        self.kinect_detector = KinectDetector()
        self.gamepad = vg.VX360Gamepad()
        self.fps = fps
        self.secondary_actions_wait = 0.5 / self.fps
        self.pause_key = pause_key


    def run(self):
        """Runs the Kinect Gamepad Controller repeatedly"""
        target_frame_duration = 1.0 / self.fps
        while True:
            self.handle_pausing()
            t_frame_start = time.time()
            # Detect gestures using the Kinect
            actions = self.kinect_detector.get_actions(update_body_frame=True)
            # Send the corresponding action to the gamepad
            self.execute_actions(actions)
            # Wait a bit for next kinect frame
            t_frame_end = time.time()
            wait = target_frame_duration - (t_frame_end - t_frame_start)
            if wait > 0:
                time.sleep(wait)


    def execute_actions(self, actions):
        """
            Sends the action to the gamepad. Some actions involve two actions like pressing and releasing a button. Which is
            why we have the primary actions (pressing the button) and the optional secondary actions (releasing then button).
        """
        # PRIMARY ACTIONS
        # Send lean to gamepad
        move_x, move_y = actions["lean"]
        self.gamepad.left_joystick_float(x_value_float=move_x, y_value_float=move_y)
        # Moving right joystick
        cam_move_x, cam_move_y = actions["camera_move"]
        cam_move_y *= 0.5  # reduce up and down
        self.gamepad.right_joystick_float(x_value_float=cam_move_x, y_value_float=cam_move_y)  # divide y axis by 2 to reduce up and down and invert
        # Pressing X Button
        if actions["is_consuming_item"]:
            print("CONSUMING ITEM")
            self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
        # Pressing Left Thumbstick
        if actions["triggered_jump"]:
            print("triggered_jump")
            self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_THUMB)
        # Pressing B Button
        if actions["triggered_run"]:
            print("triggered_run")
            self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
        elif actions["triggered_roll"]:
            print("triggered_roll")
            self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
        else:
            self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
        # Pressing the A Button
        if actions["triggered_interaction"]:
            print("triggered_interaction")
            self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)

        # Pressing RB
        if actions["triggered_light_attack"]:
            print("triggered_light_attack")
            self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)
        # Pressing RT
        if actions["triggered_heavy_attack"]:
            print("triggered_heavy_attack")
            self.gamepad.right_trigger(value=255)
        else:
            self.gamepad.right_trigger(value=0)
        # Pressing LB
        if actions["triggered_block"]:
            print("triggered_block")
            self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)
        else:
            self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)
        # Pressing LT
        if actions["triggered_parry"]:
            print("triggered_parry")
            self.gamepad.left_trigger(value=255)
        # Pressing Right Thumbstick
        if actions["triggered_lock_on"]:
            print("triggered_lock_on")
            self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_THUMB)
        # Kick
        if actions["triggered_kick"]:
            print("triggered_kick")
            self.gamepad.left_joystick_float(x_value_float=0.0, y_value_float=1.0)
            self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)

        self.gamepad.update()

        # SECONDARY ACTIONS
        time.sleep(self.secondary_actions_wait)

        if actions["is_consuming_item"]:
            self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_X)

        if actions["triggered_jump"]:
            self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_THUMB)

        if actions["triggered_interaction"]:
            self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)

        if actions["triggered_light_attack"]:
            self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)

        if actions["triggered_parry"]:
            self.gamepad.left_trigger(value=0)

        if actions["triggered_lock_on"]:
            self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_THUMB)

        if actions["triggered_kick"]:
            self.gamepad.left_joystick_float(x_value_float=actions["lean"][0], y_value_float=actions["lean"][1])
            self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)

        self.gamepad.update()


    def handle_pausing(self, sleep_duration=0.01):
        """
            If pause_key is pressed wait until pause_key is released, pressed and released again.
        """
        # No pausing required
        if not keyboard.is_pressed(self.pause_key):
            return
        print(f"\nPAUSED! Press {self.pause_key} to continue...\n")
        self.gamepad.reset()
        self.gamepad.update()
        # Wait to release pause_key
        while keyboard.is_pressed(self.pause_key):
            time.sleep(sleep_duration)
        # Wait for unpause key press
        while not keyboard.is_pressed(self.pause_key):
            time.sleep(sleep_duration)
        # Wait for unpause key release
        while keyboard.is_pressed(self.pause_key):
            time.sleep(sleep_duration)
        print(f"\nUNPAUSED!\n")


def testy(fps=20):
    """Test the controller with the camera and optionally the skeleton being displayed with pygame"""
    import threading
    args = {}
    kwargs = {
        "display_type": "pygame",
        "single_skeleton_color": (120, 20, 120),
        "window_size": (1280, 720),
        "show_fps": False,
    }
    cam_thread = threading.Thread(target=show_camera, args=args, kwargs=kwargs)
    cam_thread.start()
    controller = KinectGamepadController(fps=fps)
    controller.run()


if __name__ == "__main__":
    testy(fps=25)

if __name__ == "__main__":
    print(f"\n\n    Finished Script '{os.path.basename(__file__)}' at {time.strftime('%Y-%m-%d_%H-%M-%S')}    \n\n")