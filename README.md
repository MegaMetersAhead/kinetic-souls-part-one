# Kinetic-Souls Part One
Playing Dark Souls 3 with the Kinect V2 like you're the character in the game.
<p>Watch the video about the journey here: https://youtu.be/AYUUGhctgag

## Quick Start Guide

### Disclaimer
Please note that this guide is not exhaustive. The project has primarily been tested on my machine and may not cover every aspect for every setup. It serves as guidance to assist someone with some technical knowledge. If you encounter specific issues that are not mentioned here, feel free to contact me, preferably here on Github.

### Requirements
- Kinect V2 for Windows (with USB connector).
- Windows 8 or later (Windows 10 recommended).
- Dark Souls 3 (Should mostly work for other SoulsBorne games too).

### Setup
1. [Install DARK SOULS III](https://store.steampowered.com/app/374320/DARK_SOULS_III/).
1. [Install Python 3.7](https://www.python.org/downloads/release/python-379/). With most other versions you'll run into some issues that are not listed here.
1. [Install Kinect for Windows SDK 2.0](https://www.microsoft.com/en-US/download/details.aspx?id=44561).
1. [Install Git](https://github.com/git-for-windows/git/releases/download/v2.42.0.windows.2/Git-2.42.0.2-64-bit.exe).
1. Open the command prompt (console/terminal) where you want to store the project folder by entering `cmd` into the Windows Explorer address bar.
1. Enter `git clone https://github.com/MegaMetersAhead/kinetic-souls-part-one.git` in the console.
1. Navigate to the project folder by entering `cd .\kinetic-souls-part-one` in the console.
1. Make sure you got the right python version installed by entering `py -3.7 --version` in the console. If it says anything with `Python 3.7` in it you're good. If it says `No suitable Python runtime found` it either means you haven't installed Python 3.7.X or your system can't find it.
1. Create a virtual environment by entering `py -3.7 -m venv venv` in the console.
1. Activate the virtual environment by entering `venv\Scripts\activate` in the console.
1. Install required modules by entering `pip install -r requirements.txt` in the console.
1. We now have to fix some issues in the PyKinect2 module. In the Windows Explorer, navigate from the 'kineti-souls-part-one' folder to the file 'venv\lib\site-packages\pykinect2\PyKinectV2.py'. Delete or comment out the code `assert sizeof(tagSTATSTG) == 72, sizeof(tagSTATSTG)` at line 2216. In the same file at line 2863 delete or comment out the line `from comtypes import _check_version; _check_version('')`.
1. At this point, the program should be ready to run.

### Run the Program
1. If you've closed the console, reopen it in the 'kinetic-souls-part-one' folder and activate the virtual environment with `venv\Scripts\activate`.
1. Enter `python main.py`.
1. The program should now be running. You should see a window displaying the RGB video footage from the Kinect.

### Usage
- By default, a skeleton should be drawn onto the video footage if one is detected. Ensure you're not too close to the camera.
- Toggle the skeleton display by having the video window in focus and pressing the spacebar.
- To terminate the program, close the video window, then press Ctrl+C in the console. A disconnected sound should play, and the Kinect's XBox logo should stop glowing (unless other applications are still using it).

## Bug Fixes and Tips
- You can check if your Kinect is working properly by using the Kinect SDK "SDK Browser v2.0 (Kinect for Windows)" and its example programs.
- If the image freezes periodically, adjust the 'Microphone Privacy Settings' to allow apps and desktop apps to access your microphone.
- Use [this online gamepad tester](https://hardwaretester.com/gamepad) to check if everything is working without starting a game.
