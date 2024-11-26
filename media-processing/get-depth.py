import time
import depth_pro
from depth_pro.depth_pro import DepthProConfig
import cv2
import os
import numpy as np


def save_depth_map(depth_np, output_dir, image_filename):
    base_filename = os.path.splitext(os.path.basename(image_filename))[0]
    npy_filename = os.path.join(output_dir, f"{base_filename}_depth.npy")

    # Save as .npy file
    np.save(npy_filename, depth_np)


def extract_depth(image_name, image_folder):
    depth_model, transform = depth_pro.create_model_and_transforms(config=custom_config)
    if has_gpu:
        depth_model = depth_model.to("cuda")  # Move depth model to GPU
    else:
        depth_model = depth_model
    depth_model.eval()

    image, _, f_px = depth_pro.load_rgb(str(os.path.join(image_folder, image_name)))
    if has_gpu:
        depth_input = transform(image).to("cuda")  # Move input tensor to GPU
    else:
        depth_input = transform(image)
    prediction = depth_model.infer(depth_input, f_px=f_px)
    depth = prediction["depth"]
    depth_np = depth.squeeze().cpu().numpy()  # Convert depth tensor to NumPy array

    save_depth_map(depth_np, image_folder, image_name)


has_gpu = False
custom_config = DepthProConfig(
    patch_encoder_preset="dinov2l16_384",
    image_encoder_preset="dinov2l16_384",
    checkpoint_uri="../library/ml-depth-pro/checkpoints/depth_pro.pt",
    decoder_features=256,
    use_fov_head=True,
    fov_encoder_preset="dinov2l16_384",
)
frame_directory = "../media/frames"

for directory in os.listdir(frame_directory):
    frame_folder = os.path.join(frame_directory, directory)
    if os.path.isdir(frame_folder):
        print("Processing folder:", directory)
        for filename in os.listdir(frame_folder):
            print("     ", filename)
            extract_depth(filename, frame_folder)
