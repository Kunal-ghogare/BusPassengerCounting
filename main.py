import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np

# Tracker class definition
class Tracker:
    def __init__(self, max_distance=50):
        self.tracked_objects = {}  # Store object IDs and current positions
        self.max_distance = max_distance
        self.next_object_id = 1
        self.counted_objects = set()  # Track IDs of counted objects

    def update(self, new_rectangles):
        updated_objects = {}

        # Iterate over the new detected rectangles
        for new_rect in new_rectangles:
            matched = False

            for obj_id, obj_rect in self.tracked_objects.items():
                # Calculate the center of the new rectangle
                new_center = (
                    (new_rect[0] + new_rect[2]) / 2,
                    (new_rect[1] + new_rect[3]) / 2,
                )

                # Calculate the center of the existing tracked object's rectangle
                obj_center = (
                    (obj_rect[0] + obj_rect[2]) / 2,
                    (obj_rect[1] + obj_rect[3]) / 2,
                )

                # Calculate the Euclidean distance between the centers
                distance = ((new_center[0] - obj_center[0]) ** 2 +
                            (new_center[1] - obj_center[1]) ** 2) ** 0.5

                # If the distance is within the threshold, update the tracked object
                if distance <= self.max_distance:
                    updated_objects[obj_id] = new_rect
                    matched = True
                    break

            # If no match is found, create a new tracked object
            if not matched:
                updated_objects[self.next_object_id] = new_rect
                self.next_object_id += 1

        # Update the tracked_objects dictionary with the updated objects
        self.tracked_objects = updated_objects

        return self.tracked_objects

# YOLO model initialization
model = YOLO('yolov8s.pt')

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', lambda event, x, y, flags, param: None)
cap = cv2.VideoCapture('busfinal.mp4')

# Reading class names from file
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

count = 0
total_passengers = 0  # Variable to store the total number of passengers

# Define the entry zone (interior of the bus where passengers should be counted)
entry_zone = [(500, 100), (900, 500)]  # Adjust these coordinates to match the interior area of the bus

# Tracker initialization
tracker = Tracker()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    current_rects = []
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'person' in c:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
            current_rects.append([x1, y1, x2, y2])

    tracked_objects = tracker.update(current_rects)

    # Check if any tracked objects enter the bus interior area (entry zone)
    for obj_id, bbox in tracked_objects.items():
        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2

        # Visualize the center of each person with a blue dot
        cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

        # Check if the person is fully inside the entry zone and not yet counted
        if (entry_zone[0][0] < center_x < entry_zone[1][0] and
            entry_zone[0][1] < center_y < entry_zone[1][1] and
            obj_id not in tracker.counted_objects):
            
            total_passengers += 1
            tracker.counted_objects.add(obj_id)  # Mark the object as counted
            print(f"Person entered the bus, Total Passengers: {total_passengers}")

    # Display the total passenger count on the top-right corner
    cv2.putText(frame, f'Total Passengers: {total_passengers}', 
                (50, 50),  # Position for text
                cv2.FONT_HERSHEY_SIMPLEX, 1.5,  # Font size
                (0, 255, 0), 3)  # Thickness

    # Draw the entry zone for visualization (optional)
    cv2.rectangle(frame, entry_zone[0], entry_zone[1], (0, 255, 0), 2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
