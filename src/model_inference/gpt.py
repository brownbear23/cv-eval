import os
import base64
import requests
import numpy as np
from PIL import Image
import pandas as pd
from io import BytesIO

from src.frame_processing.drawer.img_drawer import draw_on_image
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
def parse_gpt4o_response(response_text):
    import re
    pattern = r"([a-zA-Z\s]+)\s*\(\s*([0-1]\.\d{1,2})\s*/\s*\[\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*\]\s*\)"
    matches = re.findall(pattern, response_text)
    return [(label.strip(), float(score), [int(x0), int(y0), int(x1), int(y1)]) for label, score, x0, y0, x1, y1 in matches]


# Run on one image
def run_gpt4o(image_dir, output_img_dir, excel_data, api_key):
    image_pil = Image.open(image_dir).convert("RGB")
    image_np = np.array(image_pil)
    file_name = os.path.basename(image_dir)
    eval_img_path = os.path.join(output_img_dir, file_name)

    prompt = (
        "Please detect and label the following objects in the image: "
        "chair, dining table, potted plant, couch, backpack, door, carpet, trash bin, shoes, ladder. "
        "Provide bounding box coordinates in the format [[x0, y0, x1, y1]] and a confidence score between 0 and 1 for each object. "
        "List the format like this: label(score/[x0, y0, x1, y1])"
    )

    try:
        gpt_response = query_gpt4o(api_key, image_dir, prompt)
        objects = parse_gpt4o_response(gpt_response)
    except Exception as e:
        print(f"[ERROR] Failed on {file_name}: {e}")
        return

    new_row = {"File": file_name, "Classes(score/bbox)": ""}
    for label, score, bbox in objects:
        x0, y0, x1, y1 = bbox
        new_row["Classes(score/bbox)"] += f"{label}({score:.2f}/[{x0},{y0},{x1},{y1}]),  "
        draw_on_image(image_np, None, label, score, bbox)

    new_row["Classes(score/bbox)"] = new_row["Classes(score/bbox)"].rstrip(",  ")
    excel_data.append(new_row)
    Image.fromarray(image_np).save(eval_img_path)


# Run over all images
def evaluate_gpt4o(api_key, frame_dir="../../media/frames", output_dir="../../outputs/gpt4o_eval"):
    os.makedirs(output_dir, exist_ok=True)
    excel_rows = []

    for folder in os.listdir(frame_dir):
        folder_path = os.path.join(frame_dir, folder)
        if os.path.isdir(folder_path):
            print("Processing folder:", folder)
            out_folder = os.path.join(output_dir, folder)
            os.makedirs(out_folder, exist_ok=True)

            for fname in os.listdir(folder_path):
                if fname.endswith("jpg") and has_quality(fname, 0.4):
                    img_path = os.path.join(folder_path, fname)
                    run_gpt4o(img_path, out_folder, excel_rows, api_key)

    df = pd.DataFrame(excel_rows)
    df.to_excel(os.path.join(output_dir, "gpt4o_eval.xlsx"), index=False)

evaluate_gpt4o()