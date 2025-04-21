import os
import cv2
import pandas as pd
import numpy as np
import torch
from src.frame_processing.drawer.img_drawer import draw_on_image

# For setup
from detectron2.config import get_cfg
from detectron2 import model_zoo

# For inference
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog # Holds class names and color mappings for each dataset

from src.util.eval_util import has_quality, get_coco_lst

'''
Detectron2 installation guide: https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md

torch `2.2.0+cu118` is the latest version Detectron2 supports.

pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -e /workspace/cv-eval/library/detectron2
'''


def setup_detectron2(device="cuda", score_threshold=0.0):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
    cfg.MODEL.DEVICE = device
    return cfg


def run_detectron2(cfg_in, image_dir, output_img_dir, excel_data, device_used="cuda"):
    predictor = DefaultPredictor(cfg_in)

    # Load image
    image = cv2.imread(image_dir)
    file_name = os.path.basename(image_dir)
    eval_img_path = str(os.path.join(output_img_dir, file_name))

    # Load frame_analysis array
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

    acceptable_class_idx_lst = get_coco_lst(file_name.split('-')[0])
    annotated_cnt = 0

    for cls, bound, score in zipped_results:
        cls = int(cls.item())
        if cls in acceptable_class_idx_lst:
            annotated_cnt += 1
            cls_name = metadata.thing_classes[cls]
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

    if annotated_cnt > 0:
        new_row["Classes(score/distance)"] = new_row["Classes(score/distance)"].rstrip(",  ")
    else:
        height, width, _ = image.shape
        center_distance = round(float(depth_array[height // 2, width // 2]))
        draw_on_image(image, center_distance)
        cv2.imwrite(eval_img_path, image)
        new_row["Classes(score/distance)"] += f"None({center_distance}m)"

    excel_data.append(new_row)
    cv2.imwrite(eval_img_path, image)


def evaluate_detectron2(frame_dir="../../media/frames_scores", output_dir="../../outputs/detectron2"):
    if os.path.isdir(output_dir):
        print(f"WARNING: \"{output_dir}\" directory exists\nDelete the directory to run\nExiting...")
        return False



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
                if (os.path.exists(os.path.join(frame_folder, filename)) and os.path.join(frame_folder, filename).endswith("jpg")
                        and has_quality(filename, 0.4)):
                    image_path = os.path.join(frame_folder, filename)
                    run_detectron2(cfg, image_path, output_img_folder, excel_rows, device)

    df = pd.DataFrame(excel_rows)
    df.to_excel(os.path.join(output_dir, "detectron2_eval.xlsx"), index=False)

    return True
