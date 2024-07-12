import cv2
import numpy as np
import matplotlib.pyplot as plt


# Function to process the user's drawing to fill in the circled areas
def process_user_drawing(user_drawing):
    user_drawing = np.array(user_drawing)  # Convert to numpy array
    user_drawing_gray = cv2.cvtColor(user_drawing, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    _, thresh = cv2.threshold(user_drawing_gray, 1, 255, cv2.THRESH_BINARY)  # Threshold the drawing
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
    cv2.drawContours(user_drawing_gray, contours, -1, (255), thickness=cv2.FILLED)  # Fill the contours
    return user_drawing_gray