# inference.py
from ultralytics import YOLO

# This line loads your model. We will rename the file to 'best.pt'.
model = YOLO('best.pt') 

# This line sets the video file to use.
video_file_name = '30sec.mp4' 

# This command runs the model on your video.
results = model.track(
    source=video_file_name, 
    show=True,
    save=True
)

print("\nDone! Your processed video is in the new 'runs' folder.")