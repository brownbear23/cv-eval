from torchvision import transforms
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
import importlib.util
import numpy as np
import random
import os
from matplotlib import font_manager

from PIL import Image, ImageDraw, ImageFont

def IQA_init():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the model class definition dynamically
    class_path = hf_hub_download(repo_id="PerceptCLIP/PerceptCLIP_IQA", filename="modeling.py")
    spec = importlib.util.spec_from_file_location("modeling", class_path)
    modeling = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modeling)

    # initialize a model
    ModelClass = modeling.clip_lora_model
    model = ModelClass().to(device)

    # Load pretrained model
    model_path = hf_hub_download(repo_id="PerceptCLIP/PerceptCLIP_IQA", filename="perceptCLIP_IQA.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


# Preprocess and predict
def IQA_preprocess():
    random.seed(3407)
    transform = transforms.Compose([
        transforms.Resize((512, 384)),
        transforms.RandomCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711))
    ])
    return transform



def evaluate_quality(frame_dir="../../../media/frames", output_dir="../../../media/frames_scores"):
    if os.path.isdir(output_dir):
        print(f"WARNING: \"{output_dir}\" directory exists\nDelete the directory to run\nExiting...")
        return False

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    model = IQA_init()

    for directory in os.listdir(frame_dir):
        frame_folder = os.path.join(frame_dir, directory)
        if os.path.isdir(frame_folder):
            print("Processing folder:", directory)
            output_img_folder = os.path.join(output_dir, directory)
            os.makedirs(output_img_folder, exist_ok=True)
            for filename in os.listdir(frame_folder):
                if os.path.exists(os.path.join(frame_folder, filename)) and os.path.join(frame_folder, filename).endswith("jpg"):
                    image_path = os.path.join(frame_folder, filename)
                    image = Image.open(image_path).convert("RGB")
                    batch = torch.stack([IQA_preprocess()(image) for _ in range(15)]).to(device)  # Shape: (15, 3, 224, 224)
                    with torch.no_grad():
                        scores = model(batch).cpu().numpy()

                    iqa_score = np.mean(scores)

                    # maps the predicted score to [0,1] range
                    min_pred = -6.52
                    max_pred = 3.11
                    normalized_score = ((iqa_score - min_pred) / (max_pred - min_pred))
                    rounded_score = round(normalized_score, 2)
                    score_str = f"{rounded_score:.2f}".replace(".", "-")
                    name, ext = os.path.splitext(filename)

                    # Create new filename with score
                    new_filename = f"{name}_{score_str}{ext}"
                    output_path = os.path.join(output_img_folder, new_filename)

                    image.save(output_path)
                    print(f"    Processed & Saved image: {output_path}")

    return True

# evaluate_quality()


