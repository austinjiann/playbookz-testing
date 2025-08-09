# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a football analysis system that uses computer vision and machine learning to detect, track, and analyze players, referees, and balls in football videos. The system provides match analytics including goals, passes, and team ball possession statistics exported as JSON.

## Core Architecture

The codebase follows a modular architecture with the following key components:

### Processing Pipeline (main.py)
The main processing flow follows this sequence:
1. Video frame extraction using OpenCV
2. Object detection and tracking with YOLO and ByteTrack
3. Team assignment using color clustering (K-means)
4. Camera movement compensation using optical flow
5. Perspective transformation for real-world measurements
6. Ball possession assignment
7. Event detection (goals, passes) and possession analysis
8. JSON output generation

### Key Modules

- **trackers/**: YOLO-based object detection and tracking using ultralytics and supervision
- **team_assigner/**: K-means clustering for team identification based on shirt colors
- **player_ball_assigner/**: Logic for determining ball possession
- **camera_movement_estimator/**: Optical flow-based camera movement detection
- **view_transformer/**: Perspective transformation for converting pixel measurements to real-world units
- **events/**: Event detection modules for goals, passes, and possession analysis
- **utils/**: Video I/O utilities and bounding box operations
- **tests/**: Unit tests for event detection modules

## Development Commands

### Running the Main Analysis
```bash
python main.py
```

### Running YOLO Inference Only
```bash
python yolo_inference.py
```

### Running Unit Tests
```bash
python -m pytest tests/
# or
python tests/test_events.py
```

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install ultralytics supervision opencv-python numpy matplotlib pandas scikit-learn
```

## Required Dependencies
The project requires these Python packages (install via pip):
- ultralytics (YOLO model)
- supervision (tracking framework)
- opencv-python (video processing)
- numpy (numerical operations)
- pandas (data manipulation)
- scikit-learn (K-means clustering)

## Model Requirements

- Pre-trained YOLO model must be placed at `models/best.pt`
- Download link available in README.md
- Input videos should be placed in `input_videos/` directory

## Data Flow and Caching

The system uses pickle-based caching for expensive operations:
- Object tracking results cached in `stubs/track_stubs.pkl`
- Camera movement data cached in `stubs/camera_movement_stub.pkl`
- Set `read_from_stub=False` in main.py to regenerate cached data

## Output Structure

- Analysis results saved to `output_videos/summary.json`
- No video output generated - JSON only
- JSON contains goals, passes, possession statistics, and video metadata

## Key Implementation Details

### Object Detection Classes
The YOLO model detects:
- **Players**: Tracked with unique IDs and team assignments
- **Referees**: Tracked separately with yellow ellipse annotations
- **Ball**: Single object tracked with interpolation for missing frames
- **Goalkeeper**: Converted to player class during processing

### Team Assignment Logic
- Uses K-means clustering on player shirt colors from bounding box top half
- Extracts player cluster by excluding corner background clusters
- Assigns team colors (team 1 and team 2) based on clustering results
- Special case: Player ID 91 is hardcoded to team 1 (line 68-69 in team_assigner.py)

### Camera Movement Compensation
- Uses optical flow to detect camera movement between frames
- Adjusts player positions to account for camera motion
- Cached in `stubs/camera_movement_stub.pkl` for performance

### Perspective Transformation
- Converts pixel coordinates to real-world measurements
- Court dimensions: 68m width, 23.32m length
- Hardcoded pixel vertices for transformation matrix (view_transformer.py:9-12)
- Only transforms points inside the defined court polygon

### Ball Possession Logic
- Assigns ball to nearest player within proximity threshold
- Maintains possession history for team statistics
- Calculates possession percentages for JSON output

## Event Detection

### Goal Detection (`events/goal_detector.py`)
- Defines left and right goal zones based on configurable width percentage
- Requires ball to be in goal zone for consecutive frames
- Applies cooldown period between goals to avoid duplicates
- Uses Y-band filtering to avoid corner flag false positives
- Determines scoring team from recent possession history

### Pass Detection (`events/pass_detector.py`)
- Detects possession changes between same-team players
- Validates timing constraints (max 1.5 seconds)
- Validates distance constraints (minimum ball travel)
- Counts completed passes per team

### Possession Calculation (`events/possession.py`)
- Tracks nearest player to ball each frame within distance threshold
- Calculates possession percentages for each team
- Handles unknown possession periods
- Returns percentages rounded to 1 decimal place

## Configuration

### Environment Variables
Thresholds can be customized via environment variables:

**Goal Detection:**
- `GOAL_ZONE_WIDTH_PCT`: Goal zone width as percentage of field width (default: 0.15)
- `GOAL_COOLDOWN_SECONDS`: Minimum seconds between goals (default: 3.0)
- `CONSECUTIVE_FRAMES_REQ`: Frames ball must be in goal (default: 3)
- `GOAL_Y_BAND_TOP`: Top Y boundary for goal zone (default: 0.3)
- `GOAL_Y_BAND_BOTTOM`: Bottom Y boundary for goal zone (default: 0.8)

**Pass/Possession Detection:**
- `MIN_PASS_DISTANCE`: Minimum ball travel distance for pass (default: 5.0)
- `MAX_PASS_TIME`: Maximum time for pass transition (default: 1.5)
- `POSSESSION_DISTANCE`: Distance threshold for possession (default: 70)

## JSON Output Format

```json
{
  "version": "1.0",
  "goals": {
    "team1": 0,
    "team2": 1,
    "events": [
      {"frame": 3702, "timestamp": 123.4, "team": 2}
    ]
  },
  "passes": {"team1": 15, "team2": 12},
  "possession": {
    "team1_pct": 54.3,
    "team2_pct": 45.7,
    "unknown_pct": 0.0
  },
  "video": {
    "frames": 2976,
    "fps": 24.0,
    "duration_sec": 124.0
  }
}
```