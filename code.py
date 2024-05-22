import cv2
import argparse
import pygame
import numpy as np
from ultralytics import YOLO
import supervision as sv

ZONE_POLYGON = np.array([
    [0, 0],
    [0.5, 0],
    [0.5, 1],
    [0, 1]
])

# Define hardcoded user credentials and permissions
USER_CREDENTIALS = {
    "admin": "admin123",
    "user": "user123"
}

USER_PERMISSIONS = {
    "admin": ["admin", "user"],
    "user": ["user"]
}

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution",
        default=[1280, 720],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args

def authenticate_user():
    username = input("Enter your username: ")
    password = input("Enter your password: ")

    if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
        return username
    else:
        return None

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("yolov8l.pt")
    print(model.names)
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    zone_polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
    zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=tuple(args.webcam_resolution))
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone,
        color=sv.Color.red(),
        thickness=2,
        text_thickness=4,
        text_scale=2
    )

    alarm_triggered = False
    alarm_threshold = 1

    # Initialize pygame for playing sound
    pygame.mixer.init()
    alarm_sound = pygame.mixer.Sound("security-alarm-in-office.wav")  # Replace "alarm.wav" with your alarm sound file

    # Authenticate user
    authenticated_user = authenticate_user()
    if not authenticated_user:
        print("Authentication failed. Exiting.")
        return

    user_permissions = USER_PERMISSIONS.get(authenticated_user)

    while True:
        ret, frame = cap.read()

        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)
        person_detections = detections[detections.class_id == 0]

        # Count person detections
        num_persons = len(person_detections)

        # Annotate frame with boxes and labels
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in person_detections
        ]
        frame = box_annotator.annotate(
            scene=frame,
            detections=person_detections,
            labels=labels
        )

        # Trigger zone actions
        zone.trigger(detections=person_detections)
        frame = zone_annotator.annotate(scene=frame)

        # Check if the number of persons exceeds the threshold and trigger alarm
        if num_persons > alarm_threshold and not alarm_triggered:
            if "admin" in user_permissions:
                print("Admin triggered alarm: More than 3 persons detected!")
                alarm_sound.play()
            elif "user" in user_permissions:
                print("User triggered alarm: More than 3 persons detected!")

            alarm_triggered = True

        # Visualize the frame
        cv2.imshow("yolov8", frame)

        if cv2.waitKey(30) == 27:
            break

    # Release the video capture and stop pygame mixer
    cap.release()
    pygame.mixer.quit()


if __name__ == "__main__":
    main()
