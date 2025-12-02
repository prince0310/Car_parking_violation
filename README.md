# 🚗 Parking Violation Detection System

A real-time parking violation detection system using YOLO object detection and OpenCV. This system monitors designated no-parking zones in video feeds and detects vehicles that park illegally, providing visual and audio alerts.

## ✨ Features

- **Interactive ROI Selection**: Create custom polygon zones for no-parking areas
- **Real-time Vehicle Detection**: Uses YOLOv8 for accurate vehicle detection
- **Violation Tracking**: Time-based tracking with configurable thresholds
- **Visual & Audio Alerts**: Red overlay alerts and optional audio notifications
- **Draggable Zones**: Easy adjustment of monitoring areas
- **Command Line Interface**: Flexible configuration via command line arguments
- **Multiple Vehicle Types**: Detects cars, motorcycles, buses, and trucks
- **Video Output**: Saves processed video with violation overlays

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Windows OS (for audio alerts)
- Webcam or video file

### Installation

1. **Clone or download the project files**
   ```bash
   # Ensure you have these files:
   # - car_parking.py
   # - roi_manager.py
   # - requirements.txt
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download model** (if not already present)
   ```bash
   https://drive.google.com/file/d/1CMUodxfbMnkOicKXvYpH83rG_1MEr7pI/view?usp=drive_link
   ```

### Basic Usage

```bash
# Run with a video file
python car_parking.py --video CarParking.mp4

# Run with webcam (if available)
python car_parking.py --video 0
```

## 📖 Detailed Usage

### Command Line Arguments

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--video` | `-v` | str | **Required** | Path to input video file or camera index |
| `--output` | `-o` | str | `parking_violation_output.mp4` | Output video path |
| `--model` | `-m` | str | `yolov8n.pt` | YOLO model file path |
| `--threshold` | `-t` | float | `1.0` | Time threshold (seconds) before violation |
| `--confidence` | `-c` | float | `0.5` | Minimum confidence for detection |
| `--alert-duration` | `-d` | float | `2.0` | Visual alert duration (seconds) |
| `--no-audio` | - | flag | False | Disable audio alerts |
| `--window-size` | - | str | `1280x720` | Display window size (WIDTHxHEIGHT) |
| `--fps` | - | int | None | Override output video FPS |

### Usage Examples

```bash
# Basic usage with default settings
python car_parking.py --video input.mp4

# Custom output and threshold
python car_parking.py --video input.mp4 --output violations.mp4 --threshold 2.0

# Higher confidence detection with different model
python car_parking.py --video input.mp4 --model Parking_voilation.pt --confidence 0.7

# Disable audio and custom window size
python car_parking.py --video input.mp4 --no-audio --window-size 1920x1080

# Custom FPS and alert duration
python car_parking.py --video input.mp4 --fps 30 --alert-duration 3.0

# Use webcam with custom settings
python car_parking.py --video 0 --threshold 0.5 --confidence 0.6
```

## 🎮 Interactive Controls

Once the application starts:

1. **ROI Selection**: The video will pause for you to select a no-parking zone
   - Click to create polygon vertices
   - Press `Enter` to confirm the polygon
   - Press `Escape` to cancel and use default zone

2. **Runtime Controls**:
   - `r` - Reset ROI to center zone
   - `n` - Create new polygon zone
   - `q` - Quit application

3. **Zone Management**:
   - Drag inside the ROI polygon to move it
   - The polygon can be adjusted in real-time

## 🔧 Configuration

### Model Selection

The system supports different YOLO model sizes:

- `yolov8n.pt` - Nano (fastest, least accurate)
- `yolov8s.pt` - Small (balanced)
- `yolov8m.pt` - Medium (more accurate)
- `yolov8l.pt` - Large (very accurate)
- `yolov8x.pt` - Extra Large (most accurate, slowest)

### Detection Classes

The system detects these vehicle types:
- **Car** (class ID: 2)
- **Motorcycle** (class ID: 3)
- **Bus** (class ID: 5)
- **Truck** (class ID: 7)

### Violation Logic

1. Vehicle enters the no-parking zone
2. System tracks time spent in zone
3. If time exceeds threshold → Violation detected
4. Visual and audio alerts triggered
5. Red bounding box and "WRONG PARKING" label displayed

## 📁 Project Structure

```
Parking_voilation/
├── car_parking.py          # Main application script
├── roi_manager.py          # ROI selection and management
├── requirements.txt        # Python dependencies
├── README.md              # This documentation
├── yolov8n.pt            # YOLO model file (auto-downloaded)
├── CarParking.mp4        # Sample input video
└── parking_violation_output.mp4  # Output video
```

## 🛠️ Dependencies

- **OpenCV** - Computer vision and video processing
- **Ultralytics** - YOLO model implementation
- **NumPy** - Numerical computations
- **PyTorch** - Deep learning framework
- **Winsound** - Audio alerts (Windows only)

## 🎯 Performance Tips

1. **Model Selection**: Use `yolov8n.pt` for real-time performance, `yolov8s.pt` for better accuracy
2. **Confidence Threshold**: Higher values (0.6-0.8) reduce false positives
3. **Violation Threshold**: Adjust based on your use case (0.5s for strict, 2-3s for lenient)
4. **ROI Size**: Smaller ROIs improve performance
5. **Video Resolution**: Lower resolution videos process faster

## 🐛 Troubleshooting

### Common Issues

**Video won't load:**
```bash
# Check if file exists and is readable
python -c "import cv2; cap = cv2.VideoCapture('your_video.mp4'); print(cap.isOpened())"
```

**Model not found:**
```bash
# Download model manually
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

**Audio not working:**
- Use `--no-audio` flag to disable audio alerts
- Audio only works on Windows systems

**Poor detection accuracy:**
- Increase confidence threshold: `--confidence 0.7`
- Use a larger model: `--model yolov8s.pt`
- Ensure good lighting in video

### Error Messages

- `Error: Video file 'filename' not found!` - Check video path
- `Error: Model file 'model.pt' not found!` - Check model path
- `Error: Invalid window size format` - Use format like `1280x720`

## 📊 Output

The system generates:
- **Real-time display**: Live video with detection overlays
- **Output video**: Processed video saved to specified path
- **Console logs**: Status updates and violation alerts

### Visual Indicators

- **Green boxes**: Vehicles within allowed time
- **Red boxes**: Vehicles exceeding time threshold
- **Red polygon**: No-parking zone boundary
- **Red overlay**: Violation alert (temporary)
- **Text labels**: "WRONG PARKING" for violations

## 🤝 Contributing

Feel free to contribute by:
- Reporting bugs
- Suggesting new features
- Improving documentation
- Optimizing performance

## 📄 License

This project is open source. Please ensure you have proper licenses for any video content you process.

## 👨‍💻 Author

**Prince Kumar**
- Email: princekumar26600@gmail.com
- GitHub: [Your GitHub Profile]

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO implementation
- [OpenCV](https://opencv.org/) for computer vision tools
- [PyTorch](https://pytorch.org/) for deep learning framework

---

**Happy Parking Monitoring! 🚗🚫**

