import sys
sys.path.append('/usr/lib/python3/dist-packages')

import cv2
import numpy as np
from picamera2 import Picamera2
import time
from gpiozero import Motor
from collections import deque
from ultralytics import YOLO

picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 640)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

model = YOLO("yolo11n_openvino_model/")

motor1 = Motor(forward=24, backward=23)
motor2 = Motor(forward=17, backward=22)

prev_error = 0
midpoint_buffer = deque(maxlen=3)
last_valid_midpoint = None
last_valid_time = time.time()
LANE_TIMEOUT = 0.4
turning_mode = False
turn_direction = 0

def check_object_between_lanes(results, frame_width, lane_boundaries):
    if not results or len(results) == 0:
        return False, None
    result = results[0]
    left_boundary, right_boundary = lane_boundaries
    if left_boundary is None or right_boundary is None:
        left_boundary = frame_width * 0.25
        right_boundary = frame_width * 0.75
    danger_zone_top = frame_width * 0.5
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        box_center_x = (x1 + x2) / 2
        if (left_boundary <= box_center_x <= right_boundary and y2 >= danger_zone_top):
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            box_height = y2 - y1
            distance_estimate = frame_width / box_height
            return True, {
                'class': class_name,
                'confidence': conf,
                'distance': distance_estimate,
                'position': (box_center_x, (y1 + y2) / 2)
            }
    return False, None

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

def get_best_fit_line(lines, height):
    if not lines:
        return None, None, None
    x_vals, y_vals = zip(*[(x1, y1) for x1, y1, x2, y2 in lines] + [(x2, y2) for x1, y1, x2, y2 in lines])
    poly = np.polyfit(x_vals, y_vals, 1) if len(x_vals) > 1 else None
    if poly is None:
        return None, None, None
    slope, intercept = poly
    x1, x2 = int((height - intercept) / slope), int((int(height * 0.6) - intercept) / slope)
    midpoint = ((x1 + x2) // 2, (height + int(height * 0.6)) // 2)
    last_point = (x2, int(height * 0.6))
    return (x1, height, x2, int(height * 0.6)), midpoint, last_point

def motor_control(error, should_stop=False):
    if should_stop:
        return 0, 0
    BASE_SPEED = 30
    MAX_ADJUSTMENT = 40
    error_ratio = abs(error) / 320.0
    adjustment = error_ratio * MAX_ADJUSTMENT
    if error > 0:
        left_speed = BASE_SPEED + adjustment
        right_speed = BASE_SPEED - adjustment
    else:
        left_speed = BASE_SPEED - adjustment
        right_speed = BASE_SPEED + adjustment
    left_speed = max(10, min(100, left_speed))
    right_speed = max(10, min(100, right_speed))
    return left_speed, right_speed

def detect_lanes(frame):
    global last_valid_midpoint, last_valid_time, turning_mode, turn_direction
    edges = preprocess_frame(frame)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 120, 100, minLineLength=50, maxLineGap=30)

    left_lines, right_lines = [], []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            if 0.3 < abs(slope) < 5:
                (left_lines if slope < 0 else right_lines).append((x1, y1, x2, y2))

    height = frame.shape[0]
    left_fit, left_mid, left_last = get_best_fit_line(left_lines, height)
    right_fit, right_mid, right_last = get_best_fit_line(right_lines, height)

    current_time = time.time()
    ROAD_WIDTH = 280
    THRESHOLD = 50

    lane_boundaries = (None, None)

    if left_fit:
        cv2.line(frame, (left_fit[0], left_fit[1]), (left_fit[2], left_fit[3]), (0, 255, 0), 4)
        lane_boundaries = (left_fit[2], lane_boundaries[1])
    if right_fit:
        cv2.line(frame, (right_fit[0], right_fit[1]), (right_fit[2], right_fit[3]), (0, 255, 0), 4)
        lane_boundaries = (lane_boundaries[0], right_fit[2])

    if left_mid and right_mid:
        avg_point = ((left_mid[0] + right_mid[0]) // 2, (left_mid[1] + right_mid[1]) // 2)
        midpoint_buffer.append(avg_point[0])
        smoothed_avg_x = int(sum(midpoint_buffer) / len(midpoint_buffer))
        avg_point = (smoothed_avg_x, avg_point[1])
        last_valid_midpoint = avg_point
        last_valid_time = current_time
        turning_mode = False
        turn_direction = 0
    else:
        if right_last and not left_mid:
            virtual_x = right_last[0] - ROAD_WIDTH
            virtual_y = right_last[1] + THRESHOLD
            avg_point = (virtual_x, virtual_y)
            turn_direction = -1
            turning_mode = True
        elif left_last and not right_mid:
            virtual_x = left_last[0] + ROAD_WIDTH
            virtual_y = left_last[1] - THRESHOLD
            avg_point = (virtual_x, virtual_y)
            turn_direction = 1
            turning_mode = True
        else:
            if current_time - last_valid_time > LANE_TIMEOUT:
                if not turning_mode:
                    if last_valid_midpoint:
                        frame_center = frame.shape[1] // 2
                        turn_direction = 1 if last_valid_midpoint[0] > frame_center else -1
                    turning_mode = True
                frame_width = frame.shape[1]
                virtual_x = frame_width * (0.85 if turn_direction > 0 else 0.15)
                avg_point = (int(virtual_x), height // 2)
            else:
                avg_point = last_valid_midpoint

    if avg_point:
        cv2.circle(frame, (avg_point[0], avg_point[1]), 5, (0, 0, 255), -1)
        status = "Turning" if turning_mode else "Normal"
        cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f"Mid: {avg_point}", (avg_point[0] + 10, avg_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2)

    return frame, avg_point, lane_boundaries

def set_motor_speed(motor, speed):
    speed = max(0, min(1, abs(speed) / 100))
    if speed < 0.1:
        motor.stop()
    else:
        motor.forward(speed)

prev_error, integral, previous_time = 0, 0, time.time()
try:
    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        results = model.predict(frame, imgsz=256)
        annotated_frame = results[0].plot()
        
        processed_frame, avg_point, lane_boundaries = detect_lanes(frame)
        
        should_stop, object_info = check_object_between_lanes(results, frame.shape[1], lane_boundaries)
        
        if should_stop:
            cv2.putText(
                annotated_frame,
                f"STOP! {object_info['class']} detected",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
        
        combined_frame = np.hstack((annotated_frame, processed_frame))
        cv2.imshow("Detection Results", combined_frame)

        if avg_point:
            frame_center = frame.shape[1] // 2
            error = avg_point[0] - frame_center
            left_speed, right_speed = motor_control(error, should_stop)
            
            if should_stop:
                print("Object detected! Stopping vehicle")
                motor1.stop()
                motor2.stop()
            else:
                set_motor_speed(motor1, left_speed)
                set_motor_speed(motor2, right_speed)
            
            prev_error = error

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Program stopped by user")
finally:
    cv2.destroyAllWindows()
    picam2.stop()
    motor1.stop()
    motor2.stop()
