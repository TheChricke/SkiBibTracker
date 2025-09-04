import cv2
import torch
import re
from ultralytics import YOLO
from pathlib import Path

# Load the YOLO model (Choose the right model: yolov8n.pt, yolov8s.pt, etc.)
MERGE_THRESHOLD = 15  # Pixels
MERGE_Y_THRESHOLD = 30  # Pixels

from collections import defaultdict

class TrackedBox:
    def __init__(self, label, bbox):
        self.label = label            # integer label detected by YOLO
        self.bbox = bbox              # (x1, y1, x2, y2)
        self.frames_found = 1         # how many frames this box has been detected consistently
        self.pending = False          # waiting for merge with another number
        self.found = False            # considered stable enough to be "found"
    
    def update_bbox(self, bbox):
        # You could do averaging or just replace for now
        self.bbox = bbox

def process_frame(merged_boxes, tracked_boxes, found_boxes, threshold_found=8):
    """
    merged_boxes: list of tuples (x1, y1, x2, y2, label) for current frame
    tracked_boxes: dict label -> TrackedBox for boxes currently tracked but not found yet
    found_boxes: dict label -> TrackedBox for confirmed boxes (found)
    threshold: frames_found threshold to confirm detection
    
    Returns updated tracked_boxes and found_boxes.
    """
    # Count labels occurrences in this frame
    label_counts = defaultdict(list)
    for bbox in merged_boxes:
        x1, y1, x2, y2, label = bbox
        label_counts[label].append((x1, y1, x2, y2))

    # Mark duplicates as pending
    for label, boxes in label_counts.items():
        if len(boxes) > 1:
            # Multiple detections of same label -> all pending
            for box in boxes:
                if label in tracked_boxes:
                    tb = tracked_boxes[label]
                    tb.pending = True
                    # You might want to update bbox as well? Let's keep first box for now
                else:
                    tb = TrackedBox(label, box)
                    tb.pending = True
                    tracked_boxes[label] = tb
        else:
            # Single detection
            box = boxes[0]
            if label in tracked_boxes:
                tb = tracked_boxes[label]
                tb.pending = False
                tb.frames_found = min(tb.frames_found + 1, threshold_found)  # increment capped at threshold
                tb.update_bbox(box)
            elif label in found_boxes:
                # Already found, keep it stable in found_boxes, update bbox?
                found_boxes[label].update_bbox(box)
            else:
                # New box
                tb = TrackedBox(label, box)
                tracked_boxes[label] = tb

    # Decrement frames_found for labels not found in this frame
    current_labels = set(label_counts.keys())
    for label in list(tracked_boxes.keys()):
        if label not in current_labels:
            tb = tracked_boxes[label]
            tb.frames_found -= 1
            if tb.frames_found <= 0:
                # Remove box if lost for too long
                del tracked_boxes[label]

    # Move boxes that reached threshold frames and are not pending to found_boxes
    for label, tb in list(tracked_boxes.items()):
        if tb.frames_found >= threshold_found and not tb.pending:
            tb.found = True
            found_boxes[label] = tb
            del tracked_boxes[label]

    return tracked_boxes, found_boxes

tracked_boxes = {}
found_boxes = {}
threshold_tracking = 4
threshold_found = 15

def merge_boxes(bboxes, classes, x_threshold, y_threshold):
    """Merges bounding boxes that are close together and concatenates their labels."""
    if not bboxes:
        return []

    # Sort by x1 coordinate (left to right)
    sorted_data = sorted(zip(bboxes, classes), key=lambda b: b[0][0])
    bboxes, classes = zip(*sorted_data)

    merged = []
    current_x1, current_y1, current_x2, current_y2 = bboxes[0]
    current_label = str(int(classes[0]))

    for (x1, y1, x2, y2), cls in zip(bboxes[1:], classes[1:]):
        if x1 - current_x2 <= x_threshold and (abs(y1 - current_y1) <= y_threshold):
            # Expand current bounding box
            current_x1 = min(current_x1, x1)
            current_y1 = min(current_y1, y1)
            current_x2 = max(current_x2, x2)
            current_y2 = max(current_y2, y2)
            # Concatenate class labels
            current_label += f"{int(cls)}"
        else:
            # Save current merged box and start a new one
            merged.append((current_x1, current_y1, current_x2, current_y2, current_label))
            current_x1, current_y1, current_x2, current_y2 = x1, y1, x2, y2
            current_label = f"{int(cls)}"

    # Add the last merged box
    merged.append((current_x1, current_y1, current_x2, current_y2, current_label))
    return merged
    
def get_latest_train_folder(base_path):
    base_path = Path(base_path)
    train_dirs = [d for d in base_path.iterdir() if d.is_dir() and re.match(r"train\d+", d.name)]

    if not train_dirs:
        raise FileNotFoundError("No 'trainXX' folders found in the specified path.")

    # Sort by the number after 'train'
    latest = max(train_dirs, key=lambda d: int(re.search(r"train(\d+)", d.name).group(1)))
    print(f"[INFO] Loaded model from folder: {latest}")
    return latest

# Usage:
runs_path = "/home/chricke/Projects/runs/detect"
latest_train_folder = get_latest_train_folder(runs_path)
best_model_path = latest_train_folder / "weights" / "best.pt"

# Load the model
model = YOLO(str(best_model_path))

# Open the video file
#/home/chricke/Projects/VideoCollector/output_test_mod.mp4
video_path = "/home/chricke/Projects/test1.webm"
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps > 120 or fps <= 0:  # unrealistic, so override
    fps = 60  # or the actual FPS of your video

# Create a VideoWriter to save the output
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Create a resizable window
cv2.namedWindow("YOLOv11 Video", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv11 Video", 1000, 1000)

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop if the video ends

    # Run YOLOv11 model on the frame
    results = model(frame, conf=0.7, iou=0.5)

    # Extract detections
    for result in results:
        xyxy = result.boxes.xyxy  # Bounding box in (x_min, y_min, x_max, y_max) format
        classes = result.boxes.cls.tolist()  # Class indices (1D tensor)
        conf = result.boxes.conf  # Confidence scores (1D tensor)

        bboxes = []
        for box in xyxy:  # Loop over detections
            x1, y1, x2, y2 = map(int, box[:4].tolist())  # Convert tensor to integers
            bboxes.append((x1, y1, x2, y2))

            # Merge overlapping/close bounding boxes and concatenate labels
        merged_bboxes = merge_boxes(bboxes, classes, MERGE_THRESHOLD, MERGE_Y_THRESHOLD)
        tracked_boxes, found_boxes = process_frame(merged_bboxes, tracked_boxes, found_boxes, threshold_found)
        
        # Draw merged bounding boxes with concatenated labels
        for label, tb in tracked_boxes.items():
            x1, y1, x2, y2 = tb.bbox
            if tb.frames_found >= threshold_tracking:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 6)
    print("Found labels:", list(found_boxes.keys()))
    # Show the frame
    cv2.imshow("YOLOv11 Video", frame)
    
    # Save the frame to the output video
    out.write(frame)

    delay = int(1000 / fps)
    # Press 'q' to exit
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

