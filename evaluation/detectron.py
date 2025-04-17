import os
import cv2
import pandas as pd
import numpy as np
import torch
from media_processing.drawer.img_drawer import draw_on_image

# For setup
from detectron2.config import get_cfg
from detectron2 import model_zoo

# For inference
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog # Holds class names and color mappings for each dataset

# -------------------------------------------------------------------------------------
# Detectron2 Installation Guide:
#
# 🖥️ If you are using a **GPU**:
# Use the official prebuilt wheel from Facebook AI's CDN.
# Example for CUDA 11.8 and PyTorch 2.1:
#   pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.1/index.html
#
# ⚠️ IMPORTANT:
# - Replace `cu118` with your actual CUDA version (e.g., cu117, cu121)
# - Replace `torch2.1` with your actual PyTorch version (e.g., torch1.13, torch2.0)
# - You can find the latest links here:
#   https://detectron2.readthedocs.io/en/latest/tutorials/install.html
#
# 🧱 To check your PyTorch and CUDA version:
#   >>> import torch
#   >>> print(torch.__version__)
#   >>> print(torch.version.cuda)
#   >>> print(torch.cuda.is_available())
#
# 🧮 If you are using **CPU only**:
# Prebuilt wheels are not available — you must build from source:
#   pip install 'git+https://github.com/facebookresearch/detectron2.git'
#
# This will compile Detectron2 using your current environment and allow CPU-only execution.
#
# 🛠️ Certificate Fix (Required for some macOS or Linux environments):
# macOS:
#   /Applications/Python\ 3.x/Install\ Certificates.command
#
# Ubuntu/Debian:
#   sudo apt-get update && sudo apt-get install -y ca-certificates
# -------------------------------------------------------------------------------------



def setup_detectron2(device="cuda", score_threshold=0.0):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
    cfg.MODEL.DEVICE = device
    return cfg


def evaluate_detectron2(cfg_in, image_dir, output_img_dir, excel_data, device_used="cuda"):
    predictor = DefaultPredictor(cfg_in)

    # Load image
    image = cv2.imread(image_dir)
    file_name = os.path.basename(image_dir)
    eval_img_path = str(os.path.join(output_img_dir, file_name))

    # Load depth array
    depth_array = np.load(os.path.splitext(image_dir)[0] + "_depth.npy")
    depth_tensor = torch.tensor(depth_array, device=device_used)  # Move to GPU

    metadata = MetadataCatalog.get(cfg_in.DATASETS.TRAIN[0])

    # Run inference
    outputs = predictor(image)
    instances = outputs["instances"]

    cls_lst = instances.pred_classes.to(device_used)  # Keep classes on GPU
    box_lst = instances.pred_boxes.tensor.to(device_used)  # Keep boxes on GPU
    scores = instances.scores.to(device_used)  # Keep scores on GPU

    zipped_results = list(zip(cls_lst, box_lst, scores))

    new_row = {"File": file_name, "Classes(score/distance)": ""}

    if len(zipped_results) == 0:
        height, width, _ = image.shape
        center_distance = round(float(depth_tensor[height // 2, width // 2].item()), 2)
        draw_on_image(image, center_distance)
        cv2.imwrite(eval_img_path, image)
        new_row["Classes(score/distance)"] += f"None({center_distance}m)"
        excel_data.append(new_row)
        return

    for cls, bound, score in zipped_results:
        cls_name = metadata.thing_classes[int(cls.item())]
        score = round(float(score.item()), 2)
        x1, y1, x2, y2 = bound.tolist()  # Convert tensor to list for drawing
        center_y = int((y1 + y2) // 2)
        center_x = int((x1 + x2) // 2)

        # Safeguard against out-of-bounds access
        try:
            depth_value = round(float(depth_tensor[center_y, center_x].item()), 2)
        except IndexError:
            depth_value = None

        new_row["Classes(score/distance)"] += f"{cls_name}({score:.2f}/{depth_value}m),  "
        draw_on_image(image, depth_value, cls_name, score, [x1, y1, x2, y2])

    new_row["Classes(score/distance)"] = new_row["Classes(score/distance)"].rstrip(",  ")
    excel_data.append(new_row)
    cv2.imwrite(eval_img_path, image)


frame_dir = "../media/frames"
output_dir = "../outputs/detectron2"
os.makedirs(output_dir, exist_ok=True)
excel_rows = []

device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = setup_detectron2(device=device)

for directory in os.listdir(frame_dir):
    frame_folder = os.path.join(frame_dir, directory)
    if os.path.isdir(frame_folder):
        print("Processing folder:", directory)

        output_img_folder = os.path.join(output_dir, directory)
        os.makedirs(output_img_folder, exist_ok=True)

        for filename in os.listdir(frame_folder):
            if os.path.exists(os.path.join(frame_folder, filename)) and os.path.join(frame_folder, filename).endswith("jpg"):
                image_path = os.path.join(frame_folder, filename)
                evaluate_detectron2(cfg, image_path, output_img_folder, excel_rows, device)

df = pd.DataFrame(excel_rows)
df.to_excel(os.path.join(output_dir, "detectron2_eval.xlsx"), index=False)
