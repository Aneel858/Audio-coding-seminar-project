# Step 1: Update locale packages and generate en_US.UTF-8
!sudo apt-get update -y
!sudo apt-get install locales -y
!sudo locale-gen en_US.UTF-8

# Step 2: Export locale environment variables
%env LANG=en_US.UTF-8
%env LANGUAGE=en_US:en
%env LC_ALL=en_US.UTF-8

# Just to confirm
!echo "Locale is now: $LANG"


from google.colab import drive
drive.mount('/content/drive')

!pip install deep_sort_realtime opencv-python-headless==4.6.0.66


!pip install ultralytics

from ultralytics import YOLO

# Load YOLO model (using the small model for quick training)
model = YOLO("/content/drive/MyDrive/highway_project-20250209T201859Z-001/highway_project/Vehicle_Detection_Image_Dataset/yolo11n.pt")

# Train the model using your dataset
model.train(data="/content/drive/MyDrive/highway_project-20250209T201859Z-001/highway_project/Vehicle_Detection_Image_Dataset/data.yaml", epochs=50, imgsz=640)


# Evaluate the model on validation set
metrics = model.val()
print(metrics)

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from google.colab.patches import cv2_imshow  # For Colab


# Load YOLO model (replace with your own .pt path)
model = YOLO("/content/runs/detect/train/weights/best.pt")  # e.g. 'best.pt' for your custom weights

# Initialize DeepSort tracker
tracker = DeepSort(
    max_age=5,       # max frames to keep track of lost objects
    n_init=2,        # how many frames an object must appear in before registering
    max_cosine_distance=0.2,
    nn_budget=None,
    embedder='mobilenet',
    embedder_gpu=True
)


# Open input video
cap = cv2.VideoCapture("/content/drive/MyDrive/highway_project-20250209T201859Z-001/highway_project/Vehicle_Detection_Image_Dataset/sample_video.mp4")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # should be 1280
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # should be 720
fps = cap.get(cv2.CAP_PROP_FPS)                        # should be 30

# Prepare video writer for output
output_path = "/content/drive/MyDrive/highway_project-20250209T201859Z-001/highway_project/Vehicle_Detection_Image_Dataset"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


# Import required libraries
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from google.colab.patches import cv2_imshow  # For displaying frames in Colab

# Load YOLO model
model = YOLO("runs/detect/train/weights/best.pt")  # Replace with your YOLO weights file

# Initialize DeepSort tracker
tracker = DeepSort(
    max_age=5,
    n_init=2,
    max_cosine_distance=0.2,
    nn_budget=None,
    embedder="mobilenet",
    embedder_gpu=True
)

# Open input video
input_path = "/content/drive/MyDrive/highway_project-20250209T201859Z-001/highway_project/Vehicle_Detection_Image_Dataset/sample_video.mp4"  # Replace with your input video
cap = cv2.VideoCapture(input_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Prepare output video writer
output_path = "/content/drive/MyDrive/highway_project-20250209T201859Z-001/highway_project/Vehicle_Detection_Image_Dataset/outputvideo.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Center line and focal length for angle calculation
frame_center_x = frame_width // 2
focal_length = 70

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Draw the center line
    cv2.line(
        frame,
        (frame_center_x, 0),
        (frame_center_x, frame_height),
        (0, 0, 255), 2  # Red color
    )

    # YOLO detection
    results = model(frame)

    # Prepare detections for DeepSort
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = x2 - x1
            h = y2 - y1
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            detections.append(([x1, y1, w, h], conf, "vehicle"))

    # Update the DeepSort tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw tracking results and calculate angles
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        # Calculate angle
        x_center = (x1 + x2) // 2
        angle = np.arctan((x_center - frame_center_x) / focal_length) * (180 / np.pi)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw ID and angle
        cv2.putText(
            frame,
            f"ID: {track_id} | Angle: {angle:.2f} deg",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255), 2
        )

    # Write the frame to output video
    out.write(frame)

cap.release()
out.release()

# Display success message
print("✅ Processing complete! Video saved to:", output_path)

# View the output video
from IPython.display import Video
Video(output_path, embed=True)
