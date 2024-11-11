from ultralytics import YOLO
import cv2
from collections import deque

import time

class ObjectTracker:
    def __init__(self, max_history=20, max_inactive_frames=30, distance_threshold=20):
        self.max_history = max_history
        self.max_inactive_frames = max_inactive_frames  # Max frames an object can go undetected
        self.distance_threshold = distance_threshold  # Max distance to consider duplicate detections
        self.tracked_objects = {}

    def add_or_update_object(self, obj_id, center_point, entry_edge, roi):
        # If the object ID exists, update its position and timestamp
        if obj_id in self.tracked_objects:
            self.tracked_objects[obj_id]['history'].append(center_point)
            self.tracked_objects[obj_id]['last_detected'] = time.time()
        else:
            # Check for duplicate objects within the distance threshold
            for existing_id, data in self.tracked_objects.items():
                if self.is_duplicate(center_point, data['history'][-1]):
                    obj_id = existing_id  # Treat it as the same object ID
                    break
            # Add new object if it’s unique
            if obj_id not in self.tracked_objects:
                self.tracked_objects[obj_id] = {
                    'entry_edge': self.check_entry_edge(center_point, entry_edge, roi[0], roi[1]),
                    'history': deque([center_point], maxlen=self.max_history),
                    'counted': False,
                    'last_detected': time.time()
                }

    def is_duplicate(self, new_point, existing_point):
        # Calculate Euclidean distance and check if within threshold
        dist = ((new_point[0] - existing_point[0]) ** 2 + (new_point[1] - existing_point[1]) ** 2) ** 0.5
        return dist < self.distance_threshold

    def remove_stale_objects(self):
        # Remove objects not detected within the max_inactive_frames threshold
        current_time = time.time()
        stale_ids = [
            obj_id for obj_id, data in self.tracked_objects.items()
            if (current_time - data['last_detected']) > self.max_inactive_frames
        ]
        for obj_id in stale_ids:
            del self.tracked_objects[obj_id]

    def get_object_history(self, obj_id):
        return self.tracked_objects.get(obj_id, {}).get('history', deque())

    def mark_counted(self, obj_id):
        if obj_id in self.tracked_objects:
            self.tracked_objects[obj_id]['counted'] = True

    def is_counted(self, obj_id):
        return self.tracked_objects.get(obj_id, {}).get('counted', False)

    @staticmethod
    def check_entry_edge(position, edge, top_left, bottom_right):
        x, y = position
        if edge == 'left':
            return top_left[0] - x <= 10
        elif edge == 'right':
            return x == bottom_right[0]
        elif edge == 'top':
            return top_left[1] - y <= 10
        elif edge == 'bottom':
            return y == bottom_right[1]
        return False

class PassengerCounter:
    def __init__(self, counting_line, entry_edge, exit_edge):
        self.counting_line = counting_line
        self.entry_edge = entry_edge
        self.exit_edge = exit_edge
        self.passenger_count = 0

    @staticmethod
    def is_inside_roi(point, roi):
        x, y = point
        top_left = roi[0]
        bot_right = roi[1]
        return top_left[0] < x < bot_right[0] and top_left[1] < y < bot_right[1]

    @staticmethod
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        return 0 if val == 0 else (1 if val > 0 else -1)

    @staticmethod
    def on_segment(p, q, r):
        return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])

    def do_intersect(self, p1, p2, q1, q2):
        o1 = self.orientation(p1, p2, q1)
        o2 = self.orientation(p1, p2, q2)
        o3 = self.orientation(q1, q2, p1)
        o4 = self.orientation(q1, q2, p2)
        return (o1 != o2 and o3 != o4) or (
            (o1 == 0 and self.on_segment(p1, q1, p2)) or
            (o2 == 0 and self.on_segment(p1, q2, p2)) or
            (o3 == 0 and self.on_segment(q1, p1, q2)) or
            (o4 == 0 and self.on_segment(q1, p2, q2))
        )

    def count_if_crossed(self, obj_id, history, prev_point, current_point, tracker):
        direction = self.determine_direction(history)
        if direction == "right" and not tracker.is_counted(obj_id) and self.do_intersect(prev_point, current_point, *self.counting_line):
            self.passenger_count += 1
            tracker.mark_counted(obj_id)

    @staticmethod
    def determine_direction(history):
        if len(history) < 2:
            return None
        first_x = history[0][0]
        last_x = history[-1][0]
        if last_x - first_x > 30:
            return "right"
        elif first_x - last_x > 30:
            return "left"
        return None

class VideoProcessor:
    def __init__(self, model_path, video_path, roi, counting_line, entry_edge, exit_edge):
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(video_path)
        self.roi = roi
        self.tracker = ObjectTracker()
        self.counter = PassengerCounter(counting_line, entry_edge, exit_edge)

    def process_video(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (1020, 500))
            results = self.model.track(frame, persist=True, classes=[0], conf=0.2)

            for result in results:
                if not result.boxes:
                    continue

                boxes = result.boxes.xyxy.cpu().numpy().astype(int)
                ids = result.boxes.id.cpu().numpy().astype(int)

                for box, obj_id in zip(boxes, ids):
                    x1, y1, x2, y2 = box
                    center_point = ((x1 + x2) // 2, (y1 + y2) // 2)

                    if self.counter.is_inside_roi(center_point, self.roi):
                        self.tracker.add_or_update_object(obj_id, center_point, self.counter.entry_edge, self.roi)

                        history = self.tracker.get_object_history(obj_id)
                        if history:
                            prev_point = history[-1]
                            if len(history) >= 15:
                                self.counter.count_if_crossed(obj_id, history, prev_point, center_point, self.tracker)

                        # Draw bounding box, ID, and center point
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.circle(frame, center_point, 5, (255, 0, 0), -1)

            # Remove stale objects after each frame
            self.tracker.remove_stale_objects()

            # Draw counting line and ROI
            cv2.line(frame, *self.counter.counting_line, (0, 0, 255), 2)
            cv2.rectangle(frame, self.roi[0], self.roi[1], (255, 0, 0), 2)

            # Display passenger count
            cv2.putText(frame, f"Passenger Count: {self.counter.passenger_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('Passenger Counting', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        print("Total passenger count:", self.counter.passenger_count)

# Usage
processor = VideoProcessor('yolo11n.pt', 'busfinal.mp4', [(300, 150), (850, 500)], [(780, 150), (700, 500)], 'top', 'right')
processor.process_video()
print()
