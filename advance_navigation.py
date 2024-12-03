import cv2
import torch
import streamlit as st
import numpy as np
import math
import time
import geocoder
import os

# Load YOLOv5 model for general object detection
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load pothole detection model (YOLOv4-tiny)
pothole_net = cv2.dnn.readNet('project_files/yolov4_tiny.weights', 'project_files/yolov4_tiny.cfg')
try:
    pothole_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    pothole_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
except:
    pass  # Handle CUDA issues if not supported

pothole_model = cv2.dnn_DetectionModel(pothole_net)
pothole_model.setInputParams(size=(640, 480), scale=1/255, swapRB=True)

# Set Streamlit page configuration
st.set_page_config(layout="wide")
st.title("Object and Pothole Detection with Directional Feedback")
frame_placeholder = st.empty()

# Slider to adjust custom box dimensions
rect_width_percent = st.sidebar.slider("Bounding Box Width (%)", min_value=10, max_value=100, value=80, step=5)
rect_height_percent = st.sidebar.slider("Bounding Box Height (%)", min_value=10, max_value=100, value=30, step=5)

# Initialize camera
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Streamlit button to stop the video feed
stop_button = st.button("Stop Camera")

# Object real-world widths in cm
object_widths_cm = {
    'person': 45,
    'bicycle': 60,
    'car': 180,
    'motorbike': 80,
    'pothole': 50,  # Approximate width of a pothole in cm
}

# Assumed focal length (in pixels) for distance estimation
focal_length = 800  # Adjust based on camera calibration

# Create directory for pothole data
result_path = "pothole_coordinates"
os.makedirs(result_path, exist_ok=True)

# Variables for saving pothole images and locations
Conf_threshold = 0.5
NMS_threshold = 0.4
save_interval = 2  # seconds
last_save_time = 0
pothole_index = 0

# Function to calculate distance in cm using the pinhole camera model
def calculate_distance_cm(focal_length, real_object_width, object_width_in_image):
    distance = (real_object_width * focal_length) / object_width_in_image / 65
    return math.ceil(distance)

while cap.isOpened() and not stop_button:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally to fix the mirrored feed
    frame = cv2.flip(frame, 1)

    # Calculate custom box dimensions
    rect_width = int((rect_width_percent / 100) * frame_width)
    rect_height = int((rect_height_percent / 100) * frame_height)
    x_start = (frame_width - rect_width) // 2
    x_end = x_start + rect_width
    y_end = frame_height
    y_start = y_end - rect_height

    # Define section boundaries
    left_boundary = x_start + rect_width // 3
    right_boundary = x_start + 2 * (rect_width // 3)

    # Draw custom box and sections
    cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
    cv2.line(frame, (left_boundary, y_start), (left_boundary, y_end), (0, 255, 0), 2)
    cv2.line(frame, (right_boundary, y_start), (right_boundary, y_end), (0, 255, 0), 2)

    # Perform object detection with YOLOv5
    results = model(frame)

    # Perform pothole detection with YOLOv4-tiny
    classes, scores, boxes = pothole_model.detect(frame, Conf_threshold, NMS_threshold)
    
    # Track occupancy status for each section
    left_occupied = middle_occupied = right_occupied = pothole_detected = False

    # Process general object detections
    for result in results.xyxy[0]:
        x_min, y_min, x_max, y_max, confidence, cls = map(int, result.tolist()[:6])
        class_name = model.names[cls]
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2

        # Check if the object is within the custom box area
        if y_start <= center_y <= y_end and x_start <= center_x <= x_end:
            if center_x < left_boundary:
                left_occupied = True
            elif center_x < right_boundary:
                middle_occupied = True
            else:
                right_occupied = True

            # Draw bounding box and calculate distance
            x_min, x_max = max(x_min, x_start), min(x_max, x_end)
            y_min, y_max = max(y_min, y_start), min(y_max, y_end)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

            if class_name in object_widths_cm:
                object_width = x_max - x_min
                distance = calculate_distance_cm(focal_length, object_widths_cm[class_name], object_width)
                cv2.putText(frame, f"{class_name} ({distance} steps)", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            else:
                cv2.putText(frame, class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Process pothole detections with score > 50%
    for (classid, score, box) in zip(classes, scores, boxes):
        if score >= 0.5:  # Only process potholes with score > 50%
            label = "pothole"
            x, y, w, h = box
            pothole_width_in_image = w  # Width of the pothole in the image

            # Calculate distance for pothole
            distance = calculate_distance_cm(focal_length, object_widths_cm['pothole'], pothole_width_in_image)
            
            # Draw detection box and display distance
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(frame, f"{label} ({distance} steps)", (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)

            # Save detected pothole coordinates and image every save_interval seconds
            if time.time() - last_save_time >= save_interval:
                g = geocoder.ip('me')
                cv2.imwrite(os.path.join(result_path, f'pothole{pothole_index}.jpg'), frame)
                with open(os.path.join(result_path, f'pothole{pothole_index}.txt'), 'w') as f:
                    f.write(str(g.latlng))
                last_save_time = time.time()
                pothole_index += 1

    # Determine feedback message based on occupied sections
    feedback_text = "STOP" if left_occupied and middle_occupied and right_occupied else (
        "Move RIGHT" if not left_occupied else "Move LEFT" if not right_occupied else "Move Straight")
    if pothole_detected:
        feedback_text += " - Pothole ahead!"

    cv2.putText(frame, feedback_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

    # Display frame in Streamlit
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(rgb_frame, channels="RGB")

cap.release()
