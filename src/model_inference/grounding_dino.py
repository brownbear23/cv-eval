import os
import torch
import numpy as np
from PIL import Image
import pandas as pd
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from src.frame_processing.drawer.img_drawer import draw_on_image
from src.util.eval_util import has_quality


# -------------------------------------------------------------------------------------
# Grounded SAM2 Submodule Requirement
#
# This project optionally supports SAM2-based segmentation in combination with
# Grounding DINO (Grounded SAM2). By default, this script only uses Grounding DINO
# for zero-shot object detection with bounding boxes (no segmentation).
#
# 🧩 When do you need the `Grounded-SAM-2` submodule?
# - Only required if you want to use SAM2 for segmentation (mask prediction)
# - Not required if you're only using Grounding DINO for bounding box detection
#
# ✅ To enable Grounded SAM2:
# 1. Clone the submodule into your project:
#    git submodule add https://github.com/IDEA-Research/Grounded-Segment-Anything.git library/Grounded-SAM-2
#
# 2. Download SAM2 checkpoints:
#    cd library/Grounded-SAM-2/checkpoints
#    bash download_ckpts.sh
#
# 3. Update your Python path or working directory to include `sam2`:
#    from sam2.sam2_image_predictor import SAM2ImagePredictor
#    from sam2.build_sam import build_sam2
#
# 4. Modify the pipeline to use `sam2_predictor.set_image(...)` and call
#    `predictor.predict(...)` for segmentation masks.
#
# 🚫 If you do not need segmentation (only bounding boxes), this submodule is not necessary.
# -------------------------------------------------------------------------------------

def get_prompt(target_obj):
    prompt_map = {
        "chair": "chair .",
        "dining": "dining table .",
        "potted": "potted plant .",
        "couch": "couch .",
        "backpack": "backpack .",
        "door": "door .",
        "rolled": "rolled carpet .",
        "trash": "trash bin .",
        "shoes": "shoes .",
        "ladder": "ladder ."
    }
    target_obj_lower = target_obj.lower()
    for keyword, class_idx_lst in prompt_map.items():
        if keyword in target_obj_lower:
            return class_idx_lst
    return None

def run_grounding_dino(image_dir, output_img_dir, excel_data):
    device_used = "cuda" if torch.cuda.is_available() else "cpu"

    if device_used == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.8)

    image_pil = Image.open(image_dir).convert("RGB")
    image_np = np.array(image_pil)
    file_name = os.path.basename(image_dir)
    eval_img_path = os.path.join(output_img_dir, file_name)

    depth_array = np.load(os.path.splitext(image_dir)[0] + "_depth.npy")
    depth_tensor = torch.tensor(depth_array, device=device_used)

    # Grounding DINO setup
    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device_used)

    text_prompt = get_prompt(file_name.split('-')[0])
    # text_prompt = "chair . dining table . potted plant . couch . backpack . door . carpet . trash bin . shoes . ladder ."
    inputs = processor(images=image_pil, text=text_prompt, return_tensors="pt").to(device_used)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        # Confidence for bounding boxes
        # This threshold filters predictions based on the model's confidence scores for detecting the presence of an object within the bounding boxes. scores directly corresponds to this threshold.
        text_threshold=0.3,
        # Confidence for text labels
        # Controls the match between the text prompt and the detected object's label. It does not directly relate to the numerical confidence score stored in scores.
        target_sizes=[image_pil.size[::-1]]  # (height, width)
    )

    labels = results[0]["labels"]
    boxes = results[0]["boxes"].cpu().numpy()
    scores = results[0]["scores"].cpu().numpy().tolist() # Confidence for bounding boxes

    new_row = {"File": file_name, "Classes(score/distance)": ""}
    if len(labels) == 0:
        height, width, _ = image_np.shape
        center_distance = round(float(depth_tensor[height // 2, width // 2].item()), 2)
        draw_on_image(image_np, center_distance)
        Image.fromarray(image_np).save(eval_img_path)
        new_row["Classes(score/distance)"] += f"None({center_distance}m)"
        excel_data.append(new_row)
        return

    for cls, box, score in zip(labels, boxes, scores):
        x1, y1, x2, y2 = box
        center_x = int((x1 + x2) // 2)
        center_y = int((y1 + y2) // 2)

        try:
            depth_value = round(float(depth_tensor[center_y, center_x].item()), 2)
        except IndexError:
            depth_value = None

        score = round(float(score), 2)
        new_row["Classes(score/distance)"] += f"{cls}({score:.2f}/{depth_value}m),  "
        draw_on_image(image_np, depth_value, str(cls), score, [x1, y1, x2, y2])

    new_row["Classes(score/distance)"] = new_row["Classes(score/distance)"].rstrip(",  ")
    excel_data.append(new_row)
    Image.fromarray(image_np).save(eval_img_path)


# === Run over all images ===
def evaluate_grounding_dino_tiny(frame_dir="../../media/frames_scores", output_dir="../../outputs/grounding_dino_tiny"):
    if os.path.isdir(output_dir):
        print(f"WARNING: \"{output_dir}\" directory exists\nDelete the directory to run\nExiting...")
        return False


    os.makedirs(output_dir, exist_ok=True)
    excel_rows = []


    for folder in os.listdir(frame_dir):
        folder_path = os.path.join(frame_dir, folder)
        if os.path.isdir(folder_path):
            print("Processing folder:", folder)
            out_folder = os.path.join(output_dir, folder)
            os.makedirs(out_folder, exist_ok=True)

            for fname in os.listdir(folder_path):
                if fname.endswith("jpg") and has_quality(fname, 0.0):
                    img_path = os.path.join(folder_path, fname)
                    run_grounding_dino(img_path, out_folder, excel_rows)

    df = pd.DataFrame(excel_rows)
    df.to_excel(os.path.join(output_dir, "grounding_dino_tiny_eval.xlsx"), index=False)

    return True
