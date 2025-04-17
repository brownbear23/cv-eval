from enum import Enum
import torch

class Models(Enum):
    YOLO = 1
    DETECTRON2 = 2
    GROUNDED_SAM2 = 3  # grounding-dino-tiny


######## Options
SLICE_VIDEO = False
QUALITY_CHECK = False
DEPTH_CALC = False
MODEL = Models.YOLO


print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
