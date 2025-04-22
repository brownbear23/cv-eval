from enum import Enum
import os
import sys

from src.model_inference.gpt import evaluate_gpt4o
from src.model_inference.yolo import evaluate_yolo11m
from src.model_inference.detectron2 import evaluate_detectron2
from src.model_inference.grounding_dino import evaluate_grounding_dino_tiny
from src.frame_processing.frame_splitter import slice_video_to_frames
from src.frame_analysis.biqa.process_quality import evaluate_quality
from src.frame_analysis.depth_pro.get_depth import eval_depth


class Models(Enum):
    YOLO = 1
    DETECTRON2 = 2
    GROUNDING_DINO = 3  # grounding-dino-tiny
    GPT_O4 = 4


# --- Options ---
SLICE_VIDEO = False
QUALITY_CHECK = False
DEPTH_CALC = True
RUN_ALL = True
MODEL = Models.GPT_O4

OPEN_API_KEY = os.getenv("OPENAI_API_KEY")

IN_VIDEO_PATH = os.path.abspath("./media/videos/")
OUT_FRAME_DIR = os.path.abspath("./media/frames/")
OUT_FRAME_SCORE_DIR = os.path.abspath("./media/frames_scores/")
# OUT_FRAME_SCORE_DIR = os.path.abspath("./media/frames_scores_test/")
ML_DEPTH_PRO_CP_DIR = os.path.abspath("./library/ml-depth-pro/checkpoints/depth_pro.pt")
RESULT_DIR = os.path.abspath("./outputs/")
# RESULT_DIR = os.path.abspath("./outputs_test/")

# --- Model-Specific Detection Functions ---
def detect_objects(model_type: Models, frame_dir, result_dir):
    if model_type == Models.YOLO:
        print("YOLO11m")
        eval_ret = evaluate_yolo11m(frame_dir=frame_dir, output_dir=os.path.join(result_dir, "yolo11m"),
                                    model_weight_dir="./library/yolo-weights/yolo11m.pt")
        if not eval_ret:
            sys.exit("Stopping script due to condition.")
    elif model_type == Models.DETECTRON2:
        # Installation guide at detectron2.py
        print("Detectron2")
        eval_ret = evaluate_detectron2(frame_dir=frame_dir, output_dir=os.path.join(result_dir, "detectron2"))
        if not eval_ret:
            sys.exit("Stopping script due to condition.")
    elif model_type == Models.GROUNDING_DINO:
        print("grounding_dino_tiny")
        eval_ret = evaluate_grounding_dino_tiny(frame_dir=frame_dir, output_dir=os.path.join(result_dir, "grounding_dino_tiny"))
        if not eval_ret:
            sys.exit("Stopping script due to condition.")
    elif model_type == Models.GPT_O4:
        print("gpt_o4")

        prompt = (
            "Look at the image and list all clearly visible objects with a confidence score between 0 and 1. Format each result as:"
            "label(score)"
            "Only use your visual understanding of the image. Do not write any code. If no objects are visible, respond with:"
            "NULL"
        )
        eval_ret = evaluate_gpt4o(OPEN_API_KEY, prompt, frame_dir=frame_dir, output_dir=os.path.join(result_dir, "gpt_o4"), )
        if not eval_ret:
            sys.exit("Stopping script due to condition.")
    else:
        raise ValueError(f"Unsupported model: {model_type}")



# --- Run Pipeline ---
'''
mkdir /workspace/lib/
mkdir /workspace/lib/HF
mkdir /workspace/lib/PY

python3 -m venv /workspace/cv-eval/venv
source /workspace/cv-eval/venv/bin/activate  
pip install --upgrade pip
pip install -r requirements.txt

echo 'export HF_HOME=/workspace/lib/HF' >> ~/.bashrc
echo 'export PIP_CACHE_DIR=/workspace/lib/PY' >> ~/.bashrc

source ~/.bashrc

echo $HF_HOME
echo $PIP_CACHE_DIR
'''

if SLICE_VIDEO:
    print("### RUNNING: Slicing Video ###")
    ret = slice_video_to_frames(IN_VIDEO_PATH, OUT_FRAME_DIR, frame_interval_sec=0.25)
    if not ret:
        sys.exit("Stopping script due to condition.")

if QUALITY_CHECK:
    print("### RUNNING: Quality Check ###")
    ret = evaluate_quality(OUT_FRAME_DIR, OUT_FRAME_SCORE_DIR)
    if not ret:
        sys.exit("Stopping script due to condition.")

if DEPTH_CALC:
    print("### RUNNING: Depth Calculation ###")
    # Installation guide at get_depth.py
    eval_depth(OUT_FRAME_SCORE_DIR, ML_DEPTH_PRO_CP_DIR)

if RUN_ALL:
    print("### RUNNING: Inference for All Models ###")
    for model in Models:
        print(f"\n--- Running model: {model.name} ---")
        detect_objects(model, OUT_FRAME_SCORE_DIR, RESULT_DIR)
else:
    print("### RUNNING: Inference ###")
    detect_objects(MODEL, OUT_FRAME_SCORE_DIR, RESULT_DIR)



