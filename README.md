# Enhanced Drowsiness Detection System

A real-time drowsiness detection system that monitors driver/user alertness using computer vision and facial landmark detection. The system detects eye closure and yawning patterns to alert users and prevent accidents caused by fatigue.

## Features

- **Real-time Eye Tracking**: Monitors Eye Aspect Ratio (EAR) to detect drowsiness
- **Yawn Detection**: Identifies excessive yawning as a fatigue indicator
- **Audio Alerts**: Plays alarm sounds when drowsiness is detected
- **Emergency Contact System**: Automatically notifies emergency contacts after prolonged drowsiness
- **Session Statistics**: Tracks and exports detailed monitoring statistics
- **Customizable Sensitivity**: Adjustable thresholds for different use cases
- **User-Friendly GUI**: Modern interface with tabbed controls and real-time video feed

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/XeHunter/Drowsiness-Detector.git
cd Drowsiness-Detector
```

### 2. Create Virtual Environment
```bash
python -m venv env
```

### 3. Activate Virtual Environment

**For Windows:**
```bash
env\Scripts\activate
```

**For Linux/Mac:**
```bash
source env/bin/activate
```

### 4. Install Requirements
```bash
pip install -r requirements.txt
```

### 5. Running the Application
```bash
python main.py
```

### Basic Workflow

1. Launch the application
2. Configure settings in the "Settings" tab (optional)
3. Set emergency contact in the "Emergency" tab (optional)
4. Click "Start Monitoring" button
5. Ensure your face is visible to the camera
6. View statistics and export reports from the "Statistics" tab

## Configuration

**Eye Aspect Ratio (EAR) Threshold:**
- Default: 0.25
- Lower values: More sensitive
- Higher values: Less sensitive

**Consecutive Frames:**
- Default: 20 frames
- Adjust based on desired detection speed

**Yawn Threshold:**
- Default: 30 pixels
- Adjust based on facial structure

## Project Structure
```
Drowsiness-Detector/
│
├── main.py                          # Main application file
├── requirements.txt                 # Python dependencies
├── shape_predictor_68_face_landmarks.dat  # Facial landmark model
├── alarm.wav                         # Alert sound file
├── reports/                          # Exported statistics folder
```

## How It Works

**Eye Aspect Ratio (EAR):** Calculates the ratio of eye landmark distances to detect eye closure.

**Yawn Detection:** Measures vertical distance between lip landmarks to identify yawning.

**Alert System:** Triggers audio/visual alerts and notifies emergency contacts after 15 seconds of continuous drowsiness.
