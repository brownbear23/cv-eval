import os
import torch
import numpy as np
from PIL import Image
import pandas as pd
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from media_processing.drawer.img_drawer import draw_on_image

from transformers import AutoModelForCausalLM, LlamaTokenizer

def evaluate_cogvlm(image_dir, output_img_dir, excel_data):
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

    # CogVLM setup
    model_id = "THUDM/cogvlm-grounding-generalist-hf"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, trust_remote_code=True).to(device_used)

    text_prompt = "chair . dining table . potted plant . couch . backpack . door . carpet . trash bin . shoes . ladder ."
    inputs = processor(images=image_pil, text=text_prompt, return_tensors="pt").to(device_used)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.0,
        text_threshold=0.0,
        target_sizes=[image_pil.size[::-1]]  # (height, width)
    )

    labels = results[0]["labels"]
    boxes = results[0]["boxes"].cpu().numpy()
    scores = results[0]["scores"].cpu().numpy().tolist()

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
frame_dir = "../media/frames"
output_dir = "../outputs/GroundedSAM"
os.makedirs(output_dir, exist_ok=True)
excel_rows = []


for folder in os.listdir(frame_dir):
    folder_path = os.path.join(frame_dir, folder)
    if os.path.isdir(folder_path):
        print("Processing:", folder)
        out_folder = os.path.join(output_dir, folder)
        os.makedirs(out_folder, exist_ok=True)

        for fname in os.listdir(folder_path):
            if fname.endswith("jpg"):
                img_path = os.path.join(folder_path, fname)
                evaluate_cogvlm(img_path, out_folder, excel_rows)

df = pd.DataFrame(excel_rows)
df.to_excel(os.path.join(output_dir, "cogvlm_eval2.xlsx"), index=False)