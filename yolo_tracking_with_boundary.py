from ultralytics import YOLO
import cv2
import numpy as np
import time
from collections import deque
from typing import List, Tuple

class DetectedObject:
    def __init__(self, obj_id: int, initial_position: Tuple[int, int], max_history: int = 20):
        self.id = obj_id
        self.history = deque([], maxlen=max_history)
        self.initial_position = initial_position
        self.counted = False
        self.last_detected = time.time()

    def update_position(self, position: Tuple[int, int]):
        self.history.append(position)
        self.last_detected = time.time()

    def get_last_position(self) -> Tuple[int, int]:
        return self.history[-1] if self.history else None

    def mark_counted(self):
        self.counted = True

    def is_counted(self) -> bool:
        return self.counted

class ObjectTracker:
    def __init__(self, max_history=20, max_inactive_frames=30, distance_threshold=20):
        self.max_history = max_history
        self.max_inactive_frames = max_inactive_frames  # Max frames an object can go undetected
        self.distance_threshold = distance_threshold  # Max distance to consider duplicate detections
        self.tracked_objects = {}


model = YOLO('yolo11n.pt')  # or use a custom-trained model
video_path = 'busfinal.mp4'
cap = cv2.VideoCapture(video_path)
roi = [(300, 150), (850, 500)]  # Top-left and bottom-right corners
counting_line = [(780, 150), (700, 500)]  # Adjust these coordinates to match the interior area of the bus
entry_edge = 'top'   # Define entry edge (e.g., 'left', 'top', 'right', 'bottom')
exit_edge = 'right'   # Define exit edge (opposite side)
passenger_count = 0
tracker = ObjectTracker()

def is_inside_roi(point, roi):
    x, y = point
    top_left = roi[0]
    bot_right = roi[1]
    return top_left[0] < x < bot_right[0] and top_left[1] < y < bot_right[1]

# Intersection function
def orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    return 0 if val == 0 else (1 if val > 0 else -1)

def on_segment(p, q, r):
    return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])

def do_intersect(p1, p2, q1, q2):
    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and on_segment(p1, q1, p2): return True
    if o2 == 0 and on_segment(p1, q2, p2): return True
    if o3 == 0 and on_segment(q1, p1, q2): return True
    if o4 == 0 and on_segment(q1, p2, q2): return True
    return False

def determine_direction(history):
    if len(history) < 2:
        return None
    first_x = history[0][0]
    last_x = history[-1][0]
    if last_x - first_x > 30:  # Threshold to determine significant movement
        return "right"
    elif first_x - last_x > 30:
        return "left"
    return None

def check_entry_edge(position, edge, top_left, bottom_right):
    x, y = position
    if edge == 'left':
        return top_left[0] - x <=10
    elif edge == 'right':
        return x == bottom_right[0]
    elif edge == 'top':
        return top_left[1] - y <=10
    elif edge == 'bottom':
        return y == bottom_right[1]
    return False

# Check if the object exits through a specific edge
def check_exit_edge(position, edge, top_left, bottom_right):
    return check_entry_edge(position, edge, top_left, bottom_right)


# tracker_type: bytetrack
# track_high_thresh: 0.5
# track_low_thresh: 0.1
# new_track_thresh: 0.6
# track_buffer: 30
# match_thresh: 0.9

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))

    results = model.track(frame, persist=True, classes=[0], conf=0.4, tracker='bytetrack.yaml', iou=0.5)  # 0 is the class ID for person

    for result in results:
        try:
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            ids = result.boxes.id.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy().astype(float)
        except:
            continue

        for box, id, conf in zip(boxes, ids, confs):
            x1, y1, x2, y2 = box
            center_point = ((x1 + x2) // 2, (y1 + y2) // 2)

            if not is_inside_roi(center_point, roi):
                continue

            if id not in tracker.tracked_objects:
                tracker.tracked_objects[id] = DetectedObject(id, center_point)

                # {
                #     'entry_edge': check_entry_edge(center_point, entry_edge, roi[0], roi[1]),
                #     'history': deque(maxlen=tracker.max_history),
                #     'counted': False
                # }

            prev_point = tracker.tracked_objects[id].history[-1] if tracker.tracked_objects[id].history else center_point
            tracker.tracked_objects[id].history.append(center_point)

            if len(tracker.tracked_objects[id].history) < 15:
                continue

            direction = determine_direction(tracker.tracked_objects[id].history)

            if not tracker.tracked_objects[id].counted and do_intersect(prev_point, center_point, counting_line[0], counting_line[1]):
                if direction == "right":  # Count only passengers entering (moving right)
                    passenger_count += 1
                    tracker.tracked_objects[id].counted = True

                    print({
                        "direction": direction,
                        "center_point": center_point,
                        "prev_point": prev_point,
                        "object": tracker.tracked_objects[id],
                        "other_objects": tracker.tracked_objects.keys(),
                    })
                    print('-'*20)

    
            # Draw bounding box and ID
            # frame = results[0].plot()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"ID: {id}, Confidence: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            points = np.hstack(tracker.tracked_objects[id].history).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=(0, 230, 230), thickness=2)
            cv2.circle(frame, (center_point[0], center_point[1]), 5, (255, 0, 0), -1)
            
            # Draw bounding box, ID, and center point
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.putText(frame, f"ID: {id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # cv2.circle(frame, center_point, 5, (255, 0, 0), -1)

            

    # Draw counting line
    cv2.line(frame, counting_line[0], counting_line[1], (0, 0, 255), 2)
    cv2.rectangle(frame, roi[0], roi[1], (100, 255, 100), 2)

    # Display passenger count
    cv2.putText(frame, f"Passenger Count: {passenger_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Passenger Counting', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("total passenger count:", passenger_count)
print("passenger info-")

for id, obj in tracker.tracked_objects.items():
    print('Id', id)
    print('vals-', obj.counted, obj.initial_position, obj.last_detected)
  