import cv2
import mediapipe as mp
import math
import os

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# ADB Command Execution Function
def execute_adb_command(command):
    """Execute an ADB shell command."""
    os.system(command)

# Function to calculate distance between two landmarks
def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Variables for gesture detection
previous_y = None
previous_x = None

# Function to draw only the bounding box around the hand
def draw_hand_grid(frame, hand_landmarks):
    """Draw only the bounding box around the detected hand."""
    # Calculate bounding box of the hand
    min_x = min([landmark.x for landmark in hand_landmarks.landmark])
    min_y = min([landmark.y for landmark in hand_landmarks.landmark])
    max_x = max([landmark.x for landmark in hand_landmarks.landmark])
    max_y = max([landmark.y for landmark in hand_landmarks.landmark])
    
    # Convert the normalized coordinates to pixel values
    height, width, _ = frame.shape
    min_x_pixel = int(min_x * width)
    min_y_pixel = int(min_y * height)
    max_x_pixel = int(max_x * width)
    max_y_pixel = int(max_y * height)

    # Draw rectangle around the hand (bounding box)
    cv2.rectangle(frame, (min_x_pixel, min_y_pixel), (max_x_pixel, max_y_pixel), (0, 255, 0), 2)

# Instructions
print("Make gestures in front of the camera. Press 'ESC' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB for MediaPipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    gesture_text = ""  # Placeholder for gesture text

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Draw bounding box around the hand (without internal grid lines)
            draw_hand_grid(frame, hand_landmarks)

            # Get landmarks for gesture detection
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            # Swipe gestures based on wrist movement
            if previous_y is not None and previous_x is not None:
                if wrist.y < previous_y - 0.05:
                    gesture_text = "Swipe Up"
                    execute_adb_command("adb shell input swipe 500 1500 500 500")  # Swipe up
                elif wrist.y > previous_y + 0.05:
                    gesture_text = "Swipe Down"
                    execute_adb_command("adb shell input swipe 500 500 500 1500")  # Swipe down
                elif wrist.x < previous_x - 0.05:
                    gesture_text = "Swipe Left"
                    execute_adb_command("adb shell input swipe 800 500 200 500")  # Swipe left
                elif wrist.x > previous_x + 0.05:
                    gesture_text = "Swipe Right"
                    execute_adb_command("adb shell input swipe 200 500 800 500")  # Swipe right

            # Update previous coordinates
            previous_y = wrist.y
            previous_x = wrist.x

            # Zoom gestures based on distance between thumb and index finger
            distance = calculate_distance(thumb_tip, index_tip)
            zoom_in_threshold = 0.05  # Close proximity for zoom-in
            zoom_out_threshold = 0.2  # Far distance for zoom-out

            if distance < zoom_in_threshold:
                gesture_text = "Zoom In"
                execute_adb_command("adb shell input swipe 500 500 550 550")
                execute_adb_command("adb shell input swipe 600 600 550 550")
            elif distance > zoom_out_threshold:
                gesture_text = "Zoom Out"
                execute_adb_command("adb shell input swipe 550 550 500 500")
                execute_adb_command("adb shell input swipe 550 550 600 600")

    # Display gesture text on the frame
    if gesture_text:
        cv2.putText(frame, f"Gesture: {gesture_text}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame with bounding box around the hand
    cv2.imshow('Gesture Recognition with Bounding Box Around Hand', frame)

    # Exit on 'ESC' key press
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
