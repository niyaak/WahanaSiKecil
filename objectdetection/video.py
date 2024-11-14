import cv2
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Function to mask out the region of interest
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# Function to draw the filled polygon between the lane lines
def draw_lane_lines(img, left_line, right_line, color=[0, 255, 0], thickness=10):
    line_img = np.zeros_like(img)
    poly_pts = np.array([[left_line[:2], left_line[2:], right_line[2:], right_line[:2]]], dtype=np.int32)
    cv2.fillPoly(line_img, poly_pts, color)
    img = cv2.addWeighted(img, 0.8, line_img, 0.5, 0.0)
    return img

# Lane detection pipeline
def pipeline(image):
    height, width = image.shape[0], image.shape[1]
    region_of_interest_vertices = [(0, height), (width / 2, height / 2), (width, height)]
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cannyed_image = cv2.Canny(gray_image, 100, 200)
    cropped_image = region_of_interest(cannyed_image, np.array([region_of_interest_vertices], np.int32))

    lines = cv2.HoughLinesP(cropped_image, rho=6, theta=np.pi / 60, threshold=160,
                            lines=np.array([]), minLineLength=40, maxLineGap=25)

    left_line_x, left_line_y, right_line_x, right_line_y = [], [], [], []
    if lines is None:
        return image

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
            if abs(slope) < 0.5:
                continue
            if slope <= 0:
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
            else:
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])

    min_y, max_y = int(image.shape[0] * 3 / 5), image.shape[0]
    if left_line_x and left_line_y:
        poly_left = np.poly1d(np.polyfit(left_line_y, left_line_x, 1))
        left_x_start, left_x_end = int(poly_left(max_y)), int(poly_left(min_y))
    else:
        left_x_start, left_x_end = 0, 0

    if right_line_x and right_line_y:
        poly_right = np.poly1d(np.polyfit(right_line_y, right_line_x, 1))
        right_x_start, right_x_end = int(poly_right(max_y)), int(poly_right(min_y))
    else:
        right_x_start, right_x_end = 0, 0

    lane_image = draw_lane_lines(image, [left_x_start, max_y, left_x_end, min_y],
                                 [right_x_start, max_y, right_x_end, min_y])
    return lane_image

# Function to estimate distance based on bounding box size
def estimate_distance(bbox_width, bbox_height):
    focal_length, known_width = 1000, 2.0
    distance = (known_width * focal_length) / bbox_width
    return distance

# Main function to read and process video with YOLOv8
def process_video():
    # Load the YOLOv8 model
    model = YOLO('weights/yolov8n.pt')

    # Open the video file
    cap = cv2.VideoCapture('C:/Users/HP/Downloads/Tugas Akhir/YOLO/Latihan/latihan2.mp4')
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    # Set desired frame rate
    target_fps = 30  # Define the target frames per second
    frame_time = 1.0 / target_fps  # Time per frame to maintain target FPS

    # Initialize VideoWriter for saving output
    output_filename = 'output3.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_filename, fourcc, target_fps, (1280, 720))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # keluar dari loop jika video sudah habis dibaca

        resized_frame = cv2.resize(frame, (1280, 720))

        # Run the lane detection pipeline
        lane_frame = pipeline(resized_frame)
        
        # If the pipeline function fails to return a frame, use the original frame
        if lane_frame is None:
            lane_frame = resized_frame.copy()

        # Run YOLOv8 to detect cars in the current frame
        results = model(resized_frame)

        # Process the detections from YOLOv8
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])

                # Ambil nama kelas asli dari YOLO
                original_class_name = model.names[cls]

                # Ganti label tampilan bus menjadi odong
                if original_class_name == 'bus':
                    display_class_name = 'odong'
                else:
                    display_class_name = original_class_name

                # Pastikan pengecekan sesuai untuk kelas yang Anda inginkan
                if original_class_name in ['car', 'motorcycle', 'bus', 'person'] and conf >= 0.5:
                    label = f'{display_class_name} {conf:.2f}'
                    cv2.rectangle(lane_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(lane_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                    bbox_width = x2 - x1
                    distance = estimate_distance(bbox_width, y2 - y1)
                    distance_label = f'Distance: {distance:.2f}m'
                    cv2.putText(lane_frame, distance_label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Save the frame to output video file
        out.write(lane_frame)

        # Limit the frame rate to 30fps
        time.sleep(frame_time)

    cap.release()
    out.release()
    # cv2.destroyAllWindows() 

# Function to evaluate model directly
def evaluate_model():
    # Load model and perform evaluation
    model = YOLO('weights/yolov8n.pt')
    results = model.val(data='coco8.yaml')  

    # Retrieve metrics dictionary
    metrics = results.results_dict
    
    # Print the entire metrics dictionary to inspect its structure
    print("Metrics Dictionary:", metrics)

    # Extract metrics using exact keys from metrics dictionary
    precision = metrics.get('metrics/precision(B)')
    recall = metrics.get('metrics/recall(B)')
    map50 = metrics.get('metrics/mAP50(B)')
    map50_95 = metrics.get('metrics/mAP50-95(B)')

    # Calculate F1-score manually if precision and recall are available
    if precision is not None and recall is not None:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = None

    # Print final metrics
    print(f"Final Precision: {precision:.4f}" if precision is not None else "Precision not available")
    print(f"Final Recall: {recall:.4f}" if recall is not None else "Recall not available")
    print(f"Final F1-score: {f1_score:.4f}" if f1_score is not None else "F1-score not available")
    print(f"Final mAP@0.5: {map50:.4f}" if map50 is not None else "mAP@0.5 not available")
    print(f"Final mAP@0.5:0.95: {map50_95:.4f}" if map50_95 is not None else "mAP@0.5:0.95 not available")

# Run the video processing function
process_video()

# Run the model evaluation
#evaluate_model()