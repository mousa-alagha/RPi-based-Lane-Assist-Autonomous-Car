# RPi-based Lane-Assist Autonomous Car

This project implements a lane-assist autonomous car using a **Raspberry Pi** and **YOLO** for object detection, **lane detection**, and **motor control**. The car autonomously navigates and avoids obstacles using computer vision.

## Features:
- **Lane Detection**: Detects and follows the road using edge detection and Hough Line Transform.
- **Object Detection**: Identifies objects in the lane and stops the car if an obstacle is detected.
- **Motor Control**: Adjusts the motor speeds to keep the car in the lane and avoid collisions.
- **Autonomous Navigation**: Real-time feedback is provided, and the car can turn and stop based on detected lanes and objects.

## Hardware Requirements:
- Raspberry Pi 5
- **PiCamera** for image capture
- Motors and motor controller
- **YOLO** object detection model for real-time object classification

## How to Use:
1. Clone or download this repository.
2. Install the required libraries:
   - `cv2` (OpenCV)
   - `ultralytics` (YOLO)
   - `gpiozero` (motor control)
3. Set up the **Raspberry Pi** and connect the motor controller and camera.
4. Run the main Python script (`lane_assist_car.py`).
5. The car will start autonomous navigation based on lane detection and obstacle avoidance.

## Model:
The **YOLO model** is loaded and used in real-time to detect objects and stop the car if necessary.

