# MOVE TO /workspace/cv-eval/library/Grounded-SAM-2/
# Grounded SAM 2 Image Demo (with Grounding DINO)
# TODO: CUDA GPU allocation update

import os
import cv2
import json
import torch
import numpy as np
from PIL import Image
import argparse
import pandas as pd
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from img_drawer_pillow import draw_on_image 
from dds_cloudapi_sdk import Config
from dds_cloudapi_sdk import Client
from dds_cloudapi_sdk import DetectionTask
from dds_cloudapi_sdk import TextPrompt
from dds_cloudapi_sdk import DetectionModel
from dds_cloudapi_sdk import DetectionTarget
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from torch.nn import DataParallel
from media_processing.drawer.img_drawer import draw_on_image



'''
cd /Grounded-SAM-2
cd checkpoints
bash download_ckpts.sh

cd /Grounded-SAM-2
cd gdino_checkpoints
bash download_ckpts.sh

cd /Grounded-SAM-2
pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip3 install supervision pandas transformers addict yapf pycocotools timm pillow argparse openpyxl
pip3 install -e .
pip3 install --no-build-isolation -e grounding_dino


'''

def evaluate_SAM(image_dir, output_img_dir, excel_data, device_used="cuda"):

    """
    Process an image using Grounded SAM2 and a depth map.

    Args:
        image_dir (str): Path to the input image.
        output_img_dir (str): Path where the annotated image files get saved.
        excel_data (list): List that stores data that goes into the csv file.
        device_used (str): String identifying CPU or GPU usage status.

    Returns:
        None
    """

    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.8)

    # Load image
    image = Image.open(image_dir)
    file_name = os.path.basename(image_dir)
    eval_img_path = str(os.path.join(output_img_dir, file_name))

    # Load depth array
    depth_array = np.load(os.path.splitext(image_dir)[0] + "_depth.npy")
    depth_tensor = torch.tensor(depth_array, device=device_used)  # Move to GPU









    parser = argparse.ArgumentParser()
    parser.add_argument('--grounding-model', default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--text-prompt", default="car. tire.")
    parser.add_argument("--img-path", default="notebooks/images/truck.jpg")
    parser.add_argument("--sam2-checkpoint", default="./checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2-model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--output-dir", default="outputs/test_sam2.1")
    parser.add_argument("--no-dump-json", action="store_true")
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()

    GROUNDING_MODEL = args.grounding_model
    TEXT_PROMPT = "chair . dining table . potted plant . couch . backpack . door . carpet . trash bin . shoes . ladder ."
    IMG_PATH = image_dir
    SAM2_CHECKPOINT = args.sam2_checkpoint
    SAM2_MODEL_CONFIG = args.sam2_model_config


    # environment settings
    # use bfloat16
    torch.autocast(device_type=device_used, dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # build SAM2 image predictor
    sam2_checkpoint = SAM2_CHECKPOINT
    model_cfg = SAM2_MODEL_CONFIG
    model_id = GROUNDING_MODEL

    

    processor = AutoProcessor.from_pretrained(model_id)
    

    # # Initialize SAM2
    # sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device_used)
    # sam2_predictor = SAM2ImagePredictor(sam2_model)
    # # Initialize Grounding DINO
    # grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device_used),

    sam2_device = "cuda:0"
    grounding_device = "cuda:1"

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=sam2_device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(grounding_device)





    # setup the input image and text prompt for SAM 2 and Grounding DINO
    # VERY important: text queries need to be lowercased + end with a dot
    text = TEXT_PROMPT
    img_path = IMG_PATH


    sam2_predictor.set_image(np.array(image.convert("RGB")))

    # inputs = processor(images=image, text=text, return_tensors="pt").to(device_used)
    inputs = processor(images=image, text=text, return_tensors="pt").to(grounding_device)

    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
)


    # Extract the results
    cls_lst = results[0]["labels"]
    box_lst = results[0]["boxes"].cpu().numpy()
    scores = results[0]["scores"].cpu().numpy().tolist()

    zipped_results = list(zip(cls_lst, box_lst, scores))










    new_row = {"File": file_name, "Classes(score/distance)": ""}

    if len(zipped_results) == 0:
        height, width, _ = image.shape
        center_distance = round(float(depth_tensor[height // 2, width // 2].item()), 2)
        draw_on_image(image, center_distance)
        image.save(eval_img_path)
        new_row["Classes(score/distance)"] += f"None({center_distance}m)"
        excel_data.append(new_row)
        return

    for cls_name, bound, score in zipped_results:
        score = round(float(score), 2)
        x1, y1, x2, y2 = bound
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
    image.save(eval_img_path)



# Directories and initialization
frame_dir = "../../media/frames2"
output_dir = "../../outputs/GroundedSAM"
os.makedirs(output_dir, exist_ok=True)
excel_rows = []

device = "cuda" if torch.cuda.is_available() else "cpu"

# Process frames
for directory in os.listdir(frame_dir):
    frame_folder = os.path.join(frame_dir, directory)
    if os.path.isdir(frame_folder):
        print("Processing folder:", directory)

        output_img_folder = os.path.join(output_dir, directory)
        os.makedirs(output_img_folder, exist_ok=True)

        for filename in os.listdir(frame_folder):
            if os.path.exists(os.path.join(frame_folder, filename)) and os.path.join(frame_folder, filename).endswith("jpg"):
                image_path = os.path.join(frame_folder, filename)
                evaluate_SAM(image_path, output_img_folder, excel_rows, device)

# Save results to Excel
df = pd.DataFrame(excel_rows)
df.to_excel(os.path.join(output_dir, "SAM2_eval2.xlsx"), index=False)