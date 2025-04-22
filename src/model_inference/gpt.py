import os
import base64

import cv2
import requests
import numpy as np
from PIL import Image
import pandas as pd
import torch
import time

from src.frame_processing.drawer.img_drawer import draw_on_image, draw_on_image_gpt
from src.util.eval_util import has_quality


# Helper to encode an image to base64
def encode_image_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# Send image + prompt to GPT-4o
def query_gpt4o(api_key, image_path, prompt):
    base64_img = encode_image_base64(image_path)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_img}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1000
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# Parse GPT-4o response for boxes, labels, scores
def parse_gpt4o_response_label_only(response_text):
    import re
    pattern = r"([a-zA-Z\s]+)\s*\(\s*([0-1]\.\d{1,2})\s*\)"
    matches = re.findall(pattern, response_text)
    return [(label.strip(), float(score)) for label, score in matches] if matches else None




# Run on one image
def run_gpt4o(image_dir, output_img_dir, excel_data, api_key, prompt):
    device_used = "cuda" if torch.cuda.is_available() else "cpu"

    image = Image.open(image_dir).convert("RGB")
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    file_name = os.path.basename(image_dir)
    eval_img_path = str(os.path.join(output_img_dir, file_name))

    depth_array = np.load(os.path.splitext(image_dir)[0] + "_depth.npy")
    depth_tensor = torch.tensor(depth_array, device=device_used)

    try:
        gpt_response = query_gpt4o(api_key, image_dir, prompt)
        # print("-----")
        # print(image_dir)
        # print(gpt_response)
        # print("-----")
        objects = parse_gpt4o_response_label_only(gpt_response)
        time.sleep(2)
    except Exception as e:
        print(f"[ERROR] Failed on {file_name}: {e}")
        return

    new_row = {"File": file_name, "Classes(score)": ""}

    height, width, _ = image.shape
    center_distance = round(float(depth_tensor[height // 2, width // 2].item()), 2)

    if not objects:
        draw_on_image(image, center_distance)
        cv2.imwrite(eval_img_path, image)
        new_row["Classes(score)"] += f"None | {center_distance}m"
        excel_data.append(new_row)
        return

    cls_print_str = ""
    for cls_name, score in objects:
        cls_print_str += f"{cls_name}({score:.2f}), "

    cls_print_str = cls_print_str.rstrip(",  ")
    draw_on_image_gpt(image, center_distance, cls_print_str)
    new_row["Classes(score)"] += f"{cls_print_str} | {center_distance}m"

    excel_data.append(new_row)
    cv2.imwrite(eval_img_path, image)


# Run over all images
def evaluate_gpt4o(api_key, prompt, frame_dir="../../media/frames_scores", output_dir="../../outputs/gpt4o"):
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
                    run_gpt4o(img_path, out_folder, excel_rows, api_key, prompt)

    df = pd.DataFrame(excel_rows)
    df.to_excel(os.path.join(output_dir, "gpt4o_eval.xlsx"), index=False)

    return True