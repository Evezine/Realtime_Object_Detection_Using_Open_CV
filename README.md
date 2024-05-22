# Realtime_Object_Detection_Using_Open_CV

This project implements a live object detection system using the YOLOv8 model, which can trigger an alarm when the number of detected persons exceeds a threshold. It also includes user authentication and permissions to differentiate actions based on user roles.

#  Features
Live video capture using a webcam.
Object detection using the YOLOv8 model.
User authentication and role-based permissions.
Alarm system triggered based on the number of detected persons.
Real-time annotations on the video feed.


#  Requirements
Python 3.7 or later
OpenCV
Pygame
NumPy
Ultralytics YOLO
Supervision (sv) library


#  Usage
Run the script:
python main.py --webcam-resolution 1280 720
Enter your username and password when prompted. The credentials are hardcoded for demonstration purposes:
Admin credentials: admin / admin123
User credentials: user / user123
The live video feed will appear, showing detected objects with annotations.

If the number of detected persons exceeds the threshold, an alarm will be triggered.

#  Code Overview
main.py
This script contains the main logic for the live object detection system:

Argument Parsing: Parses webcam resolution from the command line arguments.
User Authentication: Prompts the user for credentials and verifies them against hardcoded values.
Video Capture: Captures live video from the webcam.
Object Detection: Uses the YOLOv8 model to detect objects in the video feed.
Annotations: Annotates detected objects and zones on the video feed.
Alarm System: Triggers an alarm if the number of detected persons exceeds a threshold and the user has appropriate permissions.
Customization

Changing the Detection Zone
The detection zone is defined by the ZONE_POLYGON array. Modify this array to change the zone's coordinates.

Updating User Credentials and Permissions
The user credentials and permissions are hardcoded in the USER_CREDENTIALS and USER_PERMISSIONS dictionaries. Update these dictionaries to change the user roles and permissions.

Adjusting the Alarm Threshold
The alarm_threshold variable defines the number of detected persons required to trigger the alarm. Modify this variable to change the threshold.

#  Dependencies
opencv-python
pygame
numpy
ultralytics
supervision
