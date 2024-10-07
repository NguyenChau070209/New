import argparse
import sys
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils import visualize
from picamera2 import Picamera2
from gpiozero import Servo, LED
from time import sleep

# Global variables to calculate FPS
COUNTER, FPS = 0, 0
START_TIME = time.time()

# Initialize camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Servo setup for X and Y axes using gpiozero
servo_x = Servo(18)  # GPIO pin for X-axis servo
servo_y = Servo(19)  # GPIO pin for Y-axis servo

# LED setup using gpiozero
led = LED(15)  # GPIO pin for LED
led.off()

# Variables to track servo angle direction for both servos
servo_angle_x = 90  # Start at middle position
servo_angle_y = 90
servo_direction_x = 1  # 1 for increasing, -1 for decreasing
servo_direction_y = 1

# Function to display FPS
def show_fps(image):
    global COUNTER, FPS, START_TIME
    COUNTER += 1
    if (time.time() - START_TIME) > 1:
        FPS = COUNTER / (time.time() - START_TIME)
        COUNTER = 0
        START_TIME = time.time()
    
    fps_text = f'FPS = {FPS:.1f}'
    text_location = (10, 30)
    font_size = 1
    font_color = (255, 255, 255)
    font_thickness = 2
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_DUPLEX,
                font_size, font_color, font_thickness, cv2.LINE_AA)

# Function to set servo angle for both X and Y axes
def set_angle(servo, angle):
    servo.value = (angle / 90) - 1  # Convert 0-180 degree range to -1 to 1 for gpiozero servos
    sleep(0.1)

# Move servo continuously between 0 and 180 degrees
def move_servo_continuous():
    global servo_angle_x, servo_direction_x, servo_angle_y, servo_direction_y
    set_angle(servo_x, servo_angle_x)
    set_angle(servo_y, servo_angle_y)
    
    servo_angle_x += 10 * servo_direction_x
    if servo_angle_x >= 180 or servo_angle_x <= 0:
        servo_direction_x *= -1

    servo_angle_y += 10 * servo_direction_y
    if servo_angle_y >= 180 or servo_angle_y <= 0:
        servo_direction_y *= -1

# Move servos based on detection box position
def move_servo_to_center(bbox_center_x, bbox_center_y, frame_center_x, frame_center_y):
    tolerance = 30  # Adjust this value if needed

    # X-axis control
    if abs(bbox_center_x - frame_center_x) > tolerance:
        if bbox_center_x < frame_center_x:
            new_angle_x = max(servo_angle_x - 5, 0)  # Prevent angle from going below 0
            set_angle(servo_x, new_angle_x)
            print("move to left")
        else:
            new_angle_x = min(servo_angle_x + 5, 180)  # Prevent angle from going above 180
            set_angle(servo_x, new_angle_x)
            print("move to right")
        global servo_angle_x
        servo_angle_x = new_angle_x
    
    # Y-axis control
    if abs(bbox_center_y - frame_center_y) > tolerance:
        if bbox_center_y < frame_center_y:
            new_angle_y = min(servo_angle_y + 5, 180)  # Prevent angle from going above 180
            set_angle(servo_y, new_angle_y)
            print("move up")
        else:
            new_angle_y = max(servo_angle_y - 5, 0)  # Prevent angle from going below 0
            set_angle(servo_y, new_angle_y)
            print("move down")
        global servo_angle_y
        servo_angle_y = new_angle_y

# Run function with two servo states
def run(model: str, max_results: int, score_threshold: float, 
        camera_id: int, width: int, height: int) -> None:
    
    frame_center_x = width // 2
    frame_center_y = height // 2
    detection_result_list = []
    detection_count = 0

    def save_result(result: vision.ObjectDetectorResult, unused_output_image: mp.Image, timestamp_ms: int):
        nonlocal detection_count
        detection_result_list.append(result)
        if len(result.detections) >= 2:
            detection_count += 1

    # Initialize the object detection model
    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.ObjectDetectorOptions(base_options=base_options,
                                           running_mode=vision.RunningMode.LIVE_STREAM,
                                           max_results=max_results, score_threshold=score_threshold,
                                           result_callback=save_result)
    detector = vision.ObjectDetector.create_from_options(options)

    while True:
        im = picam2.capture_array()
        image = cv2.resize(im, (width, height))  # Use width and height parameters
        image = cv2.flip(image, -1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        # State 1: Servo moves continuously until 2 objects are detected
        if detection_count < 2:
            move_servo_continuous()
            pass
        else:
            # State 2: Adjust servo based on object detection
            if detection_result_list:
                current_frame = visualize(image, detection_result_list[0])
                for detection in detection_result_list[0].detections:
                    bbox = detection.bounding_box
                    bbox_center_x = bbox.origin_x + bbox.width // 2
                    bbox_center_y = bbox.origin_y + bbox.height // 2

                    # Move servo to center the detection box
                    move_servo_to_center(bbox_center_x, bbox_center_y, frame_center_x, frame_center_y)

                led.on()  # Turn on LED when detection happens
                detection_result_list.clear()
                print("centered")
            else:
                led.off()  # Turn off LED when no detection

        # Show FPS on the frame
        show_fps(image)

        # Display camera frame
        cv2.imshow('object_detection', image)
        
        if cv2.waitKey(1) == 27:  # Press ESC to exit
            break

    detector.close()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', help='Path of the object detection model.', default='best.tflite')
    parser.add_argument('--maxResults', help='Max number of detection results.', default=5, type=int)
    parser.add_argument('--scoreThreshold', help='The score threshold of detection results.', type=float, default=0.6)
    parser.add_argument('--cameraId', help='Id of camera.', type=int, default=0)
    parser.add_argument('--frameWidth', help='Width of frame to capture from camera.', type=int, default=640)
    parser.add_argument('--frameHeight', help='Height of frame to capture from camera.', type=int, default=480)
    args = parser.parse_args()

    run(args.model, args.maxResults, args.scoreThreshold, args.cameraId, args.frameWidth, args.frameHeight)

if __name__ == '__main__':
    main()
