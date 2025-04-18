from enum import Enum
import torch
import os
import cv2

class Models(Enum):
    YOLO = 1
    DETECTRON2 = 2
    GROUNDED_SAM2 = 3  # grounding-dino-tiny


######## Options
SLICE_VIDEO = True
QUALITY_CHECK = True
DEPTH_CALC = True
MODEL = Models.GROUNDED_SAM2

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)



# --- Model-Specific Detection Functions ---

def detect_objects(image_path, model_type: Models):
    if model_type == Models.YOLO:
        detect_with_yolo(image_path)
    elif model_type == Models.DETECTRON2:
        detect_with_detectron2(image_path)
    elif model_type == Models.GROUNDED_SAM2:
        detect_with_grounded_sam2(image_path)
    else:
        raise ValueError(f"Unsupported model: {model_type}")



# --- Run Pipeline ---

VIDEO_PATH = "sample_video.mp4"
FRAME_DIR = "frames"

if SLICE_VIDEO:
    frame_files = slice_video_to_frames(VIDEO_PATH, FRAME_DIR)
else:
    frame_files = sorted(os.listdir(FRAME_DIR))

for frame_file in frame_files:
    frame_path = os.path.join(FRAME_DIR, frame_file)

    # Quality Check
    if QUALITY_CHECK:
        if not check_image_quality(frame_path):
            print(f"Skipping {frame_file} due to low quality.")
            continue

    # Depth Calculation
    if DEPTH_CALC:
        calculate_depth(frame_path)

    # Object Detection
    detect_objects(frame_path, MODEL)
