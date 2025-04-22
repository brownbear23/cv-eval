import depth_pro
from depth_pro.depth_pro import DepthProConfig
import os
import numpy as np
import torch
from torch import amp

from src.util.eval_util import has_quality

'''
cd /workspace/cv-eval/library/ml-depth-pro
pip install -e .
source get_pretrained_models.sh
'''


def save_depth_map(depth_np, output_dir, image_filename):
    base_filename = os.path.splitext(os.path.basename(image_filename))[0]
    npy_filename = os.path.join(output_dir, f"{base_filename}_depth.npy")

    # Save as .npy file
    np.save(npy_filename, depth_np)


def extract_depth(image_name, image_folder, custom_config):
    has_gpu = torch.cuda.is_available()
    depth_model, transform = depth_pro.create_model_and_transforms(config=custom_config)
    if has_gpu:
        depth_model = depth_model.to("cuda")  # Move frame_analysis model to GPU
    depth_model.eval()

    image, _, f_px = depth_pro.load_rgb(str(os.path.join(image_folder, image_name)))


    depth_input = transform(image).to("cuda" if has_gpu else "cpu")


    with amp.autocast("cuda" if has_gpu else "cpu"):
        prediction = depth_model.infer(depth_input, f_px=f_px)

    depth = prediction["depth"]
    depth_np = depth.squeeze().cpu().numpy()  # Convert frame_analysis tensor to NumPy array

    save_depth_map(depth_np, image_folder, image_name)


def eval_depth(in_frame_dir="../../../media/frames_scores", checkpoint_dir="../../../library/ml-depth-pro/checkpoints/depth_pro.pt"):
    custom_config = DepthProConfig(
        patch_encoder_preset="dinov2l16_384",
        image_encoder_preset="dinov2l16_384",
        checkpoint_uri=checkpoint_dir,
        decoder_features=256,
        use_fov_head=True,
        fov_encoder_preset="dinov2l16_384",
    )

    for directory in os.listdir(in_frame_dir):
        frame_folder = os.path.join(in_frame_dir, directory)
        if os.path.isdir(frame_folder):
            print("Processing folder:", directory)
            for filename in os.listdir(frame_folder):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):  # Only process images
                    if has_quality(filename, 0.0):
                        print(f"    Processing: {filename}")
                        extract_depth(filename, frame_folder, custom_config)
                    else:
                        print(f"    Rejected: {filename}")