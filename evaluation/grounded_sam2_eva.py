import os
import cv2
import json
import torch
import numpy as np
import pandas as pd
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from img_drawer import draw_on_image 

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

    # Load image
    image = cv2.imread(image_dir)
    file_name = os.path.basename(image_dir)
    eval_img_path = str(os.path.join(output_img_dir, file_name))

    # Load depth array
    depth_array = np.load(os.path.splitext(image_dir)[0] + "_depth.npy")
    depth_tensor = torch.tensor(depth_array, device=device_used)  # Move to GPU

    # Initialize SAM2 predictor
    sam2_predictor = SAM2ImagePredictor(device=device_used)

    # Define the prompt based on the COCO and custom keywords
    prompt = (
        "Detect objects in the image including chair, dining table, potted plant, "
        "couch, backpack, door, carpet, trash bin, shoes, and ladder."
    )

    # Run SAM2 inference
    sam2_results = sam2_predictor.predict(image, depth_tensor, prompt)

    # Extract the results
    cls_lst = sam2_results["classes"]  # List of class indices
    box_lst = sam2_results["boxes"]    # List of bounding boxes in [x1, y1, x2, y2] format
    scores = sam2_results["scores"]    # List of confidence scores

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
        # Determine the class name based on COCO or custom keywords
        cls_name = cls  # Assuming `cls` contains the class name; update as per SAM2 output
        if cls_name in ["chair", "dining table", "potted plant", "couch", "backpack"]:
            keyword = cls_name
        else:
            keyword_mapping = {
                "door": "door",
                "carpet": "carpet",
                "trash bin": "trash bin",
                "shoes": "shoes",
                "ladder": "ladder"
            }
            keyword = keyword_mapping.get(cls_name, "unknown")

        # Extract bounding box coordinates
        x1, y1, x2, y2 = bound.tolist()
        center_y = int((y1 + y2) // 2)
        center_x = int((x1 + x2) // 2)

        # Safeguard against out-of-bounds access
        try:
            depth_value = round(float(depth_tensor[center_y, center_x].item()), 2)
        except IndexError:
            depth_value = None

        # Add the result to the row
        new_row["Classes(score/distance)"] += f"{keyword}({score:.2f}/{depth_value}m),  "
        draw_on_image(image, depth_value, keyword, score, [x1, y1, x2, y2])

    new_row["Classes(score/distance)"] = new_row["Classes(score/distance)"].rstrip(",  ")
    excel_data.append(new_row)
    cv2.imwrite(eval_img_path, image)


# Directories and initialization
frame_dir = "frames"
output_dir = "output"
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
df.to_excel(os.path.join(output_dir, "SAM2_eval.xlsx"), index=False)
