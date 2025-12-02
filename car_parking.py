"""
Parking Violation Detection System using YOLO and OpenCV with ROI Manager
Author: Prince
Email: princekumar26600@gmail.com

This system monitors a designated no-parking zone in a video feed and detects 
vehicles that park illegally. It uses YOLOv8 for vehicle detection and provides
visual and audio alerts when violations are detected.

Features:
- Interactive polygon zone selection using ROI Manager
- Real-time vehicle detection within polygon zones
- Violation tracking with time-based alerts
- Visual overlay alerts and audio notifications
- Draggable polygon zones for easy adjustment
- Support for complex, irregular no-parking areas
"""

import cv2
import numpy as np
import time
import argparse
import os
from ultralytics import YOLO
import winsound
from roi_manager import ROIManager

def parse_arguments():
    """Parse command line arguments for the parking violation detection system."""
    parser = argparse.ArgumentParser(
        description="Parking Violation Detection System using YOLO and OpenCV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python car_parking.py --video CarParking.mp4
  python car_parking.py --video input.mp4 --output violations.mp4 --threshold 2.0
  python car_parking.py --video input.mp4 --model yolov8s.pt --confidence 0.6
  python car_parking.py --video input.mp4 --no-audio --alert-duration 3
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--video", "-v",
        type=str,
        required=True,
        help="Path to the input video file"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="parking_violation_output.mp4",
        help="Path for the output video file (default: parking_violation_output.mp4)"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="yolov8n.pt",
        help="Path to YOLO model file (default: yolov8n.pt)"
    )
    
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=1.0,
        help="Time threshold in seconds before triggering violation (default: 1.0)"
    )
    
    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=0.5,
        help="Minimum confidence score for vehicle detection (default: 0.5)"
    )
    
    parser.add_argument(
        "--alert-duration", "-d",
        type=float,
        default=2.0,
        help="Duration of visual alert in seconds (default: 2.0)"
    )
    
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable audio alerts"
    )
    
    parser.add_argument(
        "--window-size",
        type=str,
        default="1280x720",
        help="Window size for display (format: WIDTHxHEIGHT, default: 1280x720)"
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Override video FPS for output (default: use input video FPS)"
    )
    
    return parser.parse_args()

# Parse command line arguments
args = parse_arguments()

# Validate input video file
if not os.path.exists(args.video):
    print(f"Error: Video file '{args.video}' not found!")
    exit(1)

# Parse window size
try:
    window_width, window_height = map(int, args.window_size.split('x'))
except ValueError:
    print("Error: Invalid window size format. Use WIDTHxHEIGHT (e.g., 1280x720)")
    exit(1)

# Load YOLOv8 model for object detection
if not os.path.exists(args.model):
    print(f"Error: Model file '{args.model}' not found!")
    exit(1)

model = YOLO(args.model)

# Video file path
video_path = args.video
cap = cv2.VideoCapture(video_path)

# Read the first frame to get video dimensions
ret, first_frame = cap.read()
if not ret:
    print("Error: Could not read video file")
    exit()

frame_height, frame_width = first_frame.shape[:2]

# Initialize ROI Manager
roi_manager = ROIManager((frame_height, frame_width))

# Vehicle tracking variables
vehicles_in_zone = {}  # Track vehicles and their entry time
alert_triggered = {}   # Track which vehicles have triggered alerts

# Violation detection parameters
violation_threshold = args.threshold  # Time in seconds before triggering violation
alert_active = False
alert_start_time = 0
alert_duration = args.alert_duration  # Duration of visual alert in seconds

# Video output setup
fps = args.fps if args.fps is not None else int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_path = args.output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Set up OpenCV window and mouse callback
cv2.namedWindow("Parking Monitor", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Parking Monitor", window_width, window_height)
roi_manager.setup_move_mode("Parking Monitor")

# Reset video to beginning
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

print(f"Output video will be saved as: {output_path}")

print("=== Parking Violation Detection System ===")
print(f"Input video: {args.video}")
print(f"Output video: {args.output}")
print(f"Model: {args.model}")
print(f"Violation threshold: {args.threshold} seconds")
print(f"Confidence threshold: {args.confidence}")
print(f"Alert duration: {args.alert_duration} seconds")
print(f"Audio alerts: {'Disabled' if args.no_audio else 'Enabled'}")
print(f"Window size: {window_width}x{window_height}")
print("\nInstructions:")
print("- The video will pause until you select a no-parking zone")
print("- Use the ROI selection window to create your polygon")
print("- Press 'r' to reset to default zone")
print("- Press 'n' to create new polygon")
print("- Press 'q' to quit")

# Main processing loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    current_time = time.time()
    current_vehicles = set()  # Vehicles detected in current frame
    new_violation_detected = False
    
    # If ROI is not selected, pause video and show selection interface
    if not roi_manager.roi_selected:
        # Use ROI Manager's interactive selection
        success = roi_manager.select_interactive_roi(first_frame)
        if success:
            print("ROI selected successfully!")
            roi_info = roi_manager.get_roi_info()
            print(f"Polygon area: {roi_info['area']:.0f} pixels")
        else:
            print("Using default full frame ROI")
            roi_manager.create_predefined_roi("full_frame")
    
    # Process frame only if ROI is selected
    if roi_manager.roi_selected:
        # Get ROI bounding box for detection
        roi_info = roi_manager.get_roi_info()
        bbox = roi_info['bbox']
        x1, y1, x2, y2 = bbox
        
        # Ensure coordinates are within frame bounds
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(frame.shape[1], int(x2))
        y2 = min(frame.shape[0], int(y2))
        
        # Extract region of interest (ROI) from the frame
        if x2 > x1 and y2 > y1:
            roi = frame[y1:y2, x1:x2]
            
            # Run YOLO detection on the ROI
            if roi.shape[0] > 0 and roi.shape[1] > 0:
                results = model(roi)
                
                # Process each detection
                for result in results:
                    for box in result.boxes:
                        # Extract bounding box coordinates
                        rx1, ry1, rx2, ry2 = map(int, box.xyxy[0])
                        clss = int(box.cls[0].item())  # Class ID
                        confidence = float(box.conf[0].item())  # Confidence score
                        
                        # Convert ROI coordinates back to original frame coordinates
                        ox1 = rx1 + x1
                        oy1 = ry1 + y1
                        ox2 = rx2 + x1
                        oy2 = ry2 + y1
                        
                        # Check if detected object is a vehicle (car, motorcycle, bus, truck)
                        # YOLO class IDs: 2=car, 3=motorcycle, 5=bus, 7=truck
                        if clss in [2, 3, 5, 7] and confidence > args.confidence:
                            # Check if vehicle center is inside the polygon
                            center_x = (ox1 + ox2) // 2
                            center_y = (oy1 + oy2) // 2
                            
                            if roi_manager.is_point_in_roi(center_x, center_y):
                                # Create unique vehicle ID based on class and position
                                vehicle_id = f"{clss}_{ox1}_{oy1}_{ox2}_{oy2}"
                                current_vehicles.add(vehicle_id)
                                
                                # Track new vehicles entering the zone
                                if vehicle_id not in vehicles_in_zone:
                                    vehicles_in_zone[vehicle_id] = current_time
                                    alert_triggered[vehicle_id] = False
                                
                                # Calculate how long the vehicle has been in the zone
                                time_in_zone = current_time - vehicles_in_zone[vehicle_id]
                                
                                # Check for parking violation
                                if time_in_zone >= violation_threshold:
                                    # Vehicle has been in zone too long - violation detected
                                    box_color = (0, 0, 255)  # Red color for violation
                                    cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), box_color, 2)
                                    cv2.putText(frame, "WRONG PARKING", (ox1, oy1 - 5),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                                    
                                    # Trigger alert if not already triggered for this vehicle
                                    if not alert_triggered[vehicle_id]:
                                        new_violation_detected = True
                                        alert_triggered[vehicle_id] = True
                                        alert_active = True
                                        alert_start_time = current_time
                                        
                                        # Play audio alert (Windows only) - only if not disabled
                                        if not args.no_audio:
                                            try:
                                                winsound.Beep(1000, 500)  # 1000Hz for 500ms
                                            except:
                                                pass  # Ignore if audio fails
                                else:
                                    # Vehicle is in zone but within allowed time
                                    box_color = (0, 255, 0)  # Green color for acceptable
                                    cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), box_color, 2)
    
    # Remove vehicles that are no longer detected
    vehicles_to_remove = []
    for v_id in vehicles_in_zone:
        if v_id not in current_vehicles:
            vehicles_to_remove.append(v_id)
    
    # Clean up tracking dictionaries
    for v_id in vehicles_to_remove:
        vehicles_in_zone.pop(v_id)
        if v_id in alert_triggered:
            alert_triggered.pop(v_id)
    
    # Draw the no-parking zone using ROI Manager
    roi_manager.draw_roi(frame, color=(0, 0, 255), thickness=3)
    
    # Add custom label for parking zone
    if roi_manager.roi_coords:
        cv2.putText(frame, "No Parking Zone", (roi_manager.roi_coords[0][0], roi_manager.roi_coords[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Display instructions
    cv2.putText(frame, "Drag inside ROI to move polygon", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Press 'r' to reset zone, Press 'n' for new polygon", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Press 'q' to quit", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display violation alert overlay
    if alert_active and (current_time - alert_start_time < alert_duration):
        # Create red overlay for alert
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Add alert text
        alert_text = "PARKING VIOLATION DETECTED!"
        text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_TRIPLEX, 1.5, 3)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = (frame.shape[0] + text_size[1]) // 2
        cv2.putText(frame, alert_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255), 3)
    elif current_time - alert_start_time >= alert_duration:
        alert_active = False
    
    # Write frame to output video
    video_writer.write(frame)
    
    # Display the frame
    cv2.imshow('Parking Monitor', frame)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        # Quit the application
        break
    elif key == ord("r"):
        # Reset the ROI to default position
        roi_manager.create_predefined_roi("center")
        vehicles_in_zone = {}
        alert_triggered = {}
        print("ROI reset to center zone")
    elif key == ord("n"):
        # Start new ROI selection
        roi_manager.roi_selected = False
        roi_manager.roi_coords = []
        vehicles_in_zone = {}
        alert_triggered = {}
        print("Starting new ROI selection...")

# Clean up
cap.release()
video_writer.release()
cv2.destroyAllWindows()

print(f"\n=== Processing Complete ===")
print(f"Output video saved as: {output_path}")
print("✅ Parking violation detection completed successfully!")