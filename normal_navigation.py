import cv2
import torch
import streamlit as st
import numpy as np
import math

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Set Streamlit page configuration
st.set_page_config(layout="wide")

# Create Streamlit UI elements
st.title("Object Detection with Directional Feedback")
frame_placeholder = st.empty()

# Slider to adjust the custom box dimensions
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
    'person': 45,          # Approximate width of a person in cm
    'bicycle': 60,         # Standard bicycle width
    'car': 180,            # Width of a car
    'motorbike': 80,       # Width of a motorbike
}

# Assumed focal length (in pixels) for distance estimation
focal_length = 800  # Adjust based on camera calibration

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

    # Calculate custom box dimensions based on slider values
    rect_width = int((rect_width_percent / 100) * frame_width)
    rect_height = int((rect_height_percent / 100) * frame_height)
    x_start = (frame_width - rect_width) // 2  # Center horizontally
    x_end = x_start + rect_width
    y_end = frame_height  # Bottom edge
    y_start = y_end - rect_height  # Start of rectangle height

    # Define boundaries for sections within the custom box
    left_boundary = x_start + rect_width // 3
    right_boundary = x_start + 2 * (rect_width // 3)

    # Draw the custom box and sections on the frame
    cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
    cv2.line(frame, (left_boundary, y_start), (left_boundary, y_end), (0, 255, 0), 2)
    cv2.line(frame, (right_boundary, y_start), (right_boundary, y_end), (0, 255, 0), 2)
    cv2.putText(frame, "RIGHT", (x_start + 10, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "MIDDLE", (left_boundary + 10, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "LEFT", (right_boundary + 10, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Perform object detection
    results = model(frame)

    # Track whether each section has an object
    left_occupied = False
    middle_occupied = False
    right_occupied = False

    # Loop through detected objects and check if they are in the custom box area
    for result in results.xyxy[0]:  # `results.xyxy[0]` is a tensor of detections
        x_min, y_min, x_max, y_max, confidence, cls = map(int, result.tolist()[:6])
        class_name = model.names[cls]  # Object class name
        center_x = (x_min + x_max) // 2  # Center x-coordinate of the object
        center_y = (y_min + y_max) // 2  # Center y-coordinate of the object

        # Check if the detected object's center is within the custom box (green box) area
        if y_start <= center_y <= y_end and x_start <= center_x <= x_end:
            # Mark the section as occupied based on object's center x-coordinate
            if center_x < left_boundary:
                left_occupied = True
            elif center_x < right_boundary:
                middle_occupied = True
            else:
                right_occupied = True

            # Draw bounding box around the detected object (restrict it inside the green box)
            x_min = max(x_min, x_start)  # Ensure the bounding box doesn't go outside the green box
            x_max = min(x_max, x_end)
            y_min = max(y_min, y_start)
            y_max = min(y_max, y_end)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

            # If object is one of the specified types, calculate and display its distance
            if class_name in object_widths_cm:
                object_width = x_max - x_min  # Width of the object in the image
                distance = calculate_distance_cm(focal_length, object_widths_cm[class_name], object_width)
                cv2.putText(frame, f"{class_name} ({distance:.1f} steps)", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            else:
                # Just show class name if not in specified objects
                cv2.putText(frame, class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Determine feedback message based on occupied sections
    if left_occupied and middle_occupied and right_occupied:
        feedback_text = "STOP"
    elif not left_occupied and not right_occupied and not middle_occupied:
        feedback_text = "Move Straight"
    elif not left_occupied:
        feedback_text = "Move RIGHT"
    elif not right_occupied:
        feedback_text = "Move LEFT"
    elif left_occupied and right_occupied and not middle_occupied:
        feedback_text = "Move Straight"
    else:
        feedback_text = "STOP"

    # Display feedback text on the frame
    cv2.putText(frame, feedback_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

    # Convert the frame to RGB for Streamlit display
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(rgb_frame, channels="RGB")

cap.release()