"""
ROI Manager Module
Handles all Region of Interest (ROI) operations including:
- Interactive ROI selection (polygon and rectangle)
- Predefined ROI shapes
- ROI movement during runtime
- ROI visualization and cropping
"""

import cv2
import numpy as np

class ROIManager:
    """
    Comprehensive ROI management class for computer vision applications
    
    Features:
    - Interactive polygon and rectangle ROI selection
    - Predefined ROI shapes (left/right half, center, etc.)
    - Real-time ROI movement during video playback
    - ROI visualization and cropping functionality
    """
    
    def __init__(self, frame_shape):
        """
        Initialize ROI Manager
        
        Args:
            frame_shape: Tuple (height, width) of the video frame
        """
        self.frame_height, self.frame_width = frame_shape[:2]
        self.roi_points = []
        self.roi_selected = False
        self.roi_bbox = (0, 0, self.frame_width, self.frame_height)
        self.roi_mask = None
        self.roi_coords = []
        
        # For moving ROI functionality
        self.moving_roi = False
        self.drag_start = None
        self.original_roi_points = []
        self.move_threshold = 15  # Pixel threshold for detecting if click is near ROI
    
    def mouse_callback_selection(self, event, x, y, flags, param):
        """
        Mouse callback function for ROI selection mode
        
        Args:
            event: OpenCV mouse event
            x, y: Mouse coordinates
            flags: OpenCV mouse event flags
            param: Additional parameters (unused)
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.roi_points.append((x, y))
            print(f"Point {len(self.roi_points)}: ({x}, {y})")
    
    def mouse_callback_move(self, event, x, y, flags, param):
        """
        Mouse callback function for ROI movement mode
        
        Args:
            event: OpenCV mouse event
            x, y: Mouse coordinates
            flags: OpenCV mouse event flags
            param: Additional parameters (unused)
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if click is near any ROI point or inside ROI
            if self._is_point_near_roi(x, y):
                self.moving_roi = True
                self.drag_start = (x, y)
                self.original_roi_points = self.roi_coords.copy()
                print("Started moving ROI...")
        
        elif event == cv2.EVENT_MOUSEMOVE and self.moving_roi:
            if self.drag_start:
                # Calculate offset from drag start
                dx = x - self.drag_start[0]
                dy = y - self.drag_start[1]
                
                # Move all ROI points by the same offset
                self.roi_coords = [(px + dx, py + dy) for px, py in self.original_roi_points]
                
                # Update ROI properties (mask and bbox)
                self._update_roi_properties()
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.moving_roi:
                self.moving_roi = False
                self.drag_start = None
                print("ROI moved successfully!")
    
    def _is_point_near_roi(self, x, y):
        """
        Check if a point is near ROI boundary or inside ROI polygon
        
        Args:
            x, y: Point coordinates to check
            
        Returns:
            bool: True if point is inside or near ROI
        """
        if not self.roi_coords:
            return False
        
        # Check if point is inside ROI polygon
        pts = np.array(self.roi_coords, np.int32)
        result = cv2.pointPolygonTest(pts, (x, y), False)
        
        return result >= 0  # Inside or on the boundary
    
    def _update_roi_properties(self):
        """
        Update ROI mask and bounding box after ROI modification
        Internal method called after ROI coordinates change
        """
        if not self.roi_coords:
            return
        
        # Ensure ROI points are within frame bounds
        self.roi_coords = [(max(0, min(self.frame_width-1, int(x))), 
                           max(0, min(self.frame_height-1, int(y)))) 
                          for x, y in self.roi_coords]
        
        # Create new mask
        self.roi_mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        if len(self.roi_coords) >= 3:
            pts = np.array(self.roi_coords, np.int32)
            cv2.fillPoly(self.roi_mask, [pts], 1)
            
            # Update bounding box
            x_coords = [pt[0] for pt in self.roi_coords]
            y_coords = [pt[1] for pt in self.roi_coords]
            self.roi_bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
    
    def select_interactive_roi(self, first_frame):
        """
        Interactive ROI selection by clicking points to create polygon
        
        Args:
            first_frame: First frame of video for ROI selection
            
        Returns:
            bool: True if ROI was successfully selected, False otherwise
        """
        self.roi_points = []
        self.roi_selected = False
        clone = first_frame.copy()
        
        print("\n=== Interactive ROI Selection ===")
        print("Instructions:")
        print("- Left click to add points for ROI polygon")
        print("- Press 'r' to reset all points")
        print("- Press 'c' to confirm ROI (minimum 3 points required)")
        print("- Press 'q' to cancel and use full frame")
        print("- Press 's' to switch to rectangle selection mode")
        
        cv2.namedWindow("ROI Selection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ROI Selection", 1280, 720)
        cv2.setMouseCallback("ROI Selection", self.mouse_callback_selection)
        
        while True:
            display_frame = clone.copy()
            
            # Draw existing points
            for i, point in enumerate(self.roi_points):
                cv2.circle(display_frame, point, 8, (0, 255, 0), -1)
                cv2.putText(display_frame, str(i+1), (point[0]+15, point[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Draw lines connecting points
            if len(self.roi_points) > 1:
                for i in range(len(self.roi_points)):
                    start_point = self.roi_points[i]
                    end_point = self.roi_points[(i + 1) % len(self.roi_points)]
                    cv2.line(display_frame, start_point, end_point, (255, 0, 0), 3)
            
            # Fill polygon if we have enough points
            if len(self.roi_points) > 2:
                pts = np.array(self.roi_points, np.int32)
                overlay = display_frame.copy()
                cv2.fillPoly(overlay, [pts], (0, 255, 0))
                display_frame = cv2.addWeighted(display_frame, 0.7, overlay, 0.3, 0)
            
            # Add instruction text on frame
            instruction_text = f"Points: {len(self.roi_points)} | 'c' confirm | 'r' reset | 'q' cancel | 's' rectangle"
            cv2.putText(display_frame, instruction_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("ROI Selection", display_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):
                if len(self.roi_points) >= 3:
                    self.roi_selected = True
                    self.roi_coords = self.roi_points.copy()
                    print(f"ROI confirmed with {len(self.roi_points)} points")
                    break
                else:
                    print("Need at least 3 points to create ROI")
            
            elif key == ord('r'):
                self.roi_points = []
                print("Points reset")
            
            elif key == ord('q'):
                print("ROI selection cancelled - using full frame")
                self.roi_points = []
                break
                
            elif key == ord('s'):
                # Switch to rectangle selection mode
                print("Switching to rectangle selection mode...")
                cv2.destroyWindow("ROI Selection")
                return self.select_rectangle_roi(first_frame)
        
        cv2.destroyWindow("ROI Selection")
        
        if self.roi_selected:
            self._update_roi_properties()
            return True
        else:
            # Set full frame as ROI
            self.roi_coords = [(0, 0), (self.frame_width, 0), 
                             (self.frame_width, self.frame_height), (0, self.frame_height)]
            self._update_roi_properties()
            return False
    
    def select_rectangle_roi(self, first_frame):
        """
        Rectangle ROI selection using OpenCV's built-in selectROI
        
        Args:
            first_frame: First frame of video for ROI selection
            
        Returns:
            bool: True if ROI was successfully selected, False otherwise
        """
        print("\n=== Rectangle ROI Selection ===")
        print("Instructions:")
        print("- Click and drag to select rectangular ROI")
        print("- Press SPACE or ENTER to confirm")
        print("- Press 'c' to cancel")
        
        roi = cv2.selectROI("Select ROI Rectangle", first_frame, False, False)
        cv2.destroyWindow("Select ROI Rectangle")
        
        if roi[2] > 0 and roi[3] > 0:  # Valid selection
            x, y, w, h = roi
            self.roi_coords = [
                (x, y),           # top-left
                (x + w, y),       # top-right
                (x + w, y + h),   # bottom-right
                (x, y + h)        # bottom-left
            ]
            self.roi_selected = True
            self._update_roi_properties()
            print(f"Rectangle ROI selected: {self.roi_coords}")
            return True
        else:
            print("Rectangle selection cancelled")
            self.roi_coords = [(0, 0), (self.frame_width, 0), 
                             (self.frame_width, self.frame_height), (0, self.frame_height)]
            self._update_roi_properties()
            return False
    
    def create_predefined_roi(self, roi_type, custom_coords=None):
        """
        Create predefined ROI shapes
        
        Args:
            roi_type: Type of ROI ('right_half', 'left_half', 'center', 'bottom_half', 
                     'top_half', 'custom', 'full_frame')
            custom_coords: List of (x,y) coordinates for custom ROI
        """
        width, height = self.frame_width, self.frame_height
        
        if roi_type == "right_half":
            self.roi_coords = [(width//2, 0), (width, 0), (width, height), (width//2, height)]
        elif roi_type == "left_half":
            self.roi_coords = [(0, 0), (width//2, 0), (width//2, height), (0, height)]
        elif roi_type == "center":
            x1, y1 = width//4, height//4
            x2, y2 = 3*width//4, 3*height//4
            self.roi_coords = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        elif roi_type == "bottom_half":
            self.roi_coords = [(0, height//2), (width, height//2), (width, height), (0, height)]
        elif roi_type == "top_half":
            self.roi_coords = [(0, 0), (width, 0), (width, height//2), (0, height//2)]
        elif roi_type == "custom" and custom_coords:
            self.roi_coords = custom_coords
        else:
            # Full frame (default)
            self.roi_coords = [(0, 0), (width, 0), (width, height), (0, height)]
        
        self.roi_selected = True
        self._update_roi_properties()
        print(f"Created {roi_type} ROI: {self.roi_coords}")
    
    def setup_move_mode(self, window_name):
        """
        Setup mouse callback for moving ROI during video playback
        
        Args:
            window_name: Name of the OpenCV window to attach mouse callback
        """
        cv2.setMouseCallback(window_name, self.mouse_callback_move)
        print("\n=== ROI Move Mode Active ===")
        print("- Click and drag inside ROI to move it")
        print("- ROI will move in real-time during video playback")
    
    def crop_roi_from_frame(self, frame):
        """
        Crop the ROI area from the full frame
        
        Args:
            frame: Full video frame
            
        Returns:
            numpy.ndarray: Cropped frame containing only ROI area
        """
        x1, y1, x2, y2 = self.roi_bbox
        
        # Ensure coordinates are within frame bounds
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(frame.shape[1], int(x2))
        y2 = min(frame.shape[0], int(y2))
        
        # Crop the frame
        cropped_roi = frame[y1:y2, x1:x2]
        return cropped_roi
    
    def draw_roi(self, frame, color=(0, 255, 0), thickness=3):
        """
        Draw ROI boundary and information on the frame
        
        Args:
            frame: Frame to draw ROI on
            color: BGR color tuple for ROI boundary
            thickness: Line thickness for ROI boundary
        """
        if len(self.roi_coords) > 2:
            # Draw ROI polygon
            pts = np.array(self.roi_coords, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, color, thickness)
            
            # Add ROI label
            label_x = self.roi_coords[0][0] + 10
            label_y = self.roi_coords[0][1] + 30
            cv2.putText(frame, "ROI", (label_x, label_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
            
            # Add move instruction if not currently moving
            if not self.moving_roi:
                instruction_y = frame.shape[0] - 20
                cv2.putText(frame, "Click & drag inside ROI to move", (10, instruction_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                # Show moving status
                cv2.putText(frame, "Moving ROI...", (10, frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    
    def get_roi_info(self):
        """
        Get comprehensive ROI information
        
        Returns:
            dict: Dictionary containing ROI coordinates, bbox, mask, selection status, and area
        """
        roi_area = 0
        if len(self.roi_coords) > 2:
            roi_area = cv2.contourArea(np.array(self.roi_coords, np.int32))
        
        return {
            'coords': self.roi_coords,
            'bbox': self.roi_bbox,
            'mask': self.roi_mask,
            'selected': self.roi_selected,
            'area': roi_area,
            'width': self.roi_bbox[2] - self.roi_bbox[0] if self.roi_bbox else 0,
            'height': self.roi_bbox[3] - self.roi_bbox[1] if self.roi_bbox else 0,
            'moving': self.moving_roi
        }
    
    def reset_roi(self):
        """Reset ROI to full frame"""
        self.roi_coords = [(0, 0), (self.frame_width, 0), 
                          (self.frame_width, self.frame_height), (0, self.frame_height)]
        self.roi_selected = True
        self._update_roi_properties()
        print("ROI reset to full frame")
    
    def is_point_in_roi(self, x, y):
        """
        Check if a point is inside the current ROI
        
        Args:
            x, y: Point coordinates
            
        Returns:
            bool: True if point is inside ROI
        """
        if not self.roi_coords or len(self.roi_coords) < 3:
            return True  # If no ROI defined, consider all points valid
        
        pts = np.array(self.roi_coords, np.int32)
        result = cv2.pointPolygonTest(pts, (x, y), False)
        return result >= 0

# Utility functions for ROI operations
def get_predefined_roi_types():
    """
    Get list of available predefined ROI types
    
    Returns:
        dict: Dictionary mapping ROI type names to descriptions
    """
    return {
        'interactive': 'Click points to draw custom ROI polygon',
        'rectangle': 'Select rectangular ROI by dragging',
        'right_half': 'Right half of the frame',
        'left_half': 'Left half of the frame',
        'center': 'Center quarter of the frame',
        'bottom_half': 'Bottom half of the frame',
        'top_half': 'Top half of the frame',
        'custom': 'Use predefined coordinates',
        'full_frame': 'Entire frame (no ROI restriction)'
    }

def print_roi_help():
    """Print help information for ROI usage"""
    print("\n=== ROI Manager Help ===")
    print("Available ROI Types:")
    for roi_type, description in get_predefined_roi_types().items():
        print(f"  {roi_type}: {description}")
    
    print("\nInteractive Selection Controls:")
    print("  - Left click: Add point to polygon")
    print("  - 'c': Confirm ROI selection")
    print("  - 'r': Reset all points")
    print("  - 's': Switch to rectangle mode")
    print("  - 'q': Cancel and use full frame")
    
    print("\nRuntime Controls:")
    print("  - Click & drag inside ROI: Move ROI position")
    print("  - ROI moves in real-time during video playback")
    print("  - All detection continues to work on moved ROI")
