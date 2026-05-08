# Traffic Video Vehicle Tracking

This project performs vehicle (and related object) detection, tracking, and trajectory extraction from traffic videos using state-of-the-art deep learning models.

## Model Used
- **YOLOv8 (You Only Look Once, version 8)**: A real-time object detection model pretrained on the COCO dataset. Two variants are used:
  - `yolov8n.pt`: Fast, lightweight model (default in `vehicle_tracking.py`).
  - `yolov8x.pt`: Larger, more accurate model (used in `vehicle_tracking.py` for enhanced detection and more classes).

## Tracking Method
- **ByteTrack** (via the Supervision library): A robust multi-object tracking algorithm that assigns consistent IDs to detected objects across frames, enabling trajectory analysis.

## Output Format
- **Annotated Video**:
  - `outputs/annotated_video.mp4` (or `output/annotated_video.mp4`): The input video with bounding boxes, labels, and track IDs drawn for each detected object.
- **Trajectory CSV**:
  - `outputs/trajectory.csv` (or `output/trajectory.csv`): A CSV file with the following columns:
    - `frame`: Frame number
    - `track_id`: Unique ID for each tracked object
    - `vehicle_class`: Detected class (e.g., car, bus, truck, person, bicycle, train)
    - `x1`, `y1`, `x2`, `y2`: Bounding box coordinates
    - `centroid_x`, `centroid_y`: Center of the bounding box

## How to Run
1. **Install Requirements**
   ```bash
   pip install ultralytics opencv-python pandas numpy supervision
   ```
2. **Place your input video** in the project directory (default: `traffic_video.mp4`).
3. **Run the script**:
   - Basic tracking (default classes):
     ```bash
     python vehicle_tracking.py --video traffic_video.mp4 --output_dir outputs
     ```
   - Enhanced tracking (more classes, better accuracy):
     ```bash
     python vehicle_tracking2.py --video traffic_video.mp4 --output_dir output2
     ```
4. **Check the outputs** in the specified output directory.

## Customization
- You can adjust detection confidence, frame range, and output directory using command-line arguments:
  ```bash
  python vehicle_tracking2.py --video your_video.mp4 --conf 0.3 --start_frame 0 --end_frame 500 --output_dir output2
  ```

## Notes
- The first run will automatically download the required YOLOv8 model weights.
- For best results, use `vehicle_tracking.py` with `yolov8x.pt` for more accurate and diverse object detection.

---
**Author:** SVNIT Internship Project, enhanced by Copilot
