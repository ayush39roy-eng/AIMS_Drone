import cv2
import os
import time
from datetime import datetime

# Folder to save images
folder_name = "dataset/idle"
os.makedirs(folder_name, exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("Capturing 2 photos per second...")
print("Press 'q' to stop.")

# 2 photos per second
capture_interval = 0.5   # 1 / 2 FPS

while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Show live webcam
    cv2.imshow("Webcam Live", frame)

    # Create unique filename using microseconds
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = os.path.join(folder_name, f"photo_{timestamp}.png")

    # Save image
    cv2.imwrite(filename, frame)
    print(f"Saved: {filename}")

    # Maintain 2 FPS timing
    elapsed_time = time.time() - start_time
    remaining_time = capture_interval - elapsed_time
    if remaining_time > 0:
        time.sleep(remaining_time)

    # Stop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Stopping capture...")
        break

cap.release()
cv2.destroyAllWindows()
