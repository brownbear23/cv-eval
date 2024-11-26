# https://velog.io/@juneten/YOLO-v1v9
# https://github.com/ultralytics/ultralytics
# https://medium.com/@Nivitus./estimating-depth-and-focal-length-with-apple-depth-pro-e49a19392b47

from ultralytics import YOLO
import pandas as pd
import os
import cv2
import numpy as np


def draw_on_image(original_image, depth, cls_name=None, conf=None, bound=None):
    if bound:
        cv2.rectangle(original_image, bound[0], bound[1], (255, 0, 0), 2)

    text_x = bound[0][0]
    text_y = bound[0][1] - 5 if bound[0][1] - 5 > 10 else bound[0][1] + 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    if cls_name and conf:
        label = f"{cls_name} {conf:.2f} | {depth}m"
    else:
        label = "{depth}m"
    cv2.putText(
        original_image,
        label,
        (text_x, text_y),
        font,
        0.5,
        (255, 255, 255),
        1,
    )


def evaluate_yolo(batch, output_folder, data_rows):
    model = YOLO(MODEL)
    results = model(batch)
    for i in range(len(batch)):
        filename = os.path.basename(batch[i])
        original_image = cv2.imread(batch[i])
        depth_array = np.load(os.path.splitext(batch[i])[0] + "_depth.npy")

        detection = results[i]

        if len(detection.boxes) == 0:
            image = cv2.imread(batch[i])
            height, width, _ = image.shape
            center_distance = depth_array[height / 2, width / 2]
            draw_on_image(original_image, center_distance)
            continue

        detection.save()








MODEL = "../library/yolo-weights/yolo11m.pt"
frame_dir = "../media/frames"
output_dir = "../outputs/yolo"
excel_rows = []
for directory in os.listdir(frame_dir):
    frame_folder = os.path.join(frame_dir, directory)
    if os.path.isdir(frame_folder):
        print("Processing folder:", directory)

        valid_batch = [
            os.path.join(frame_folder, filename)
            for filename in os.listdir(frame_folder)
            if os.path.exists(os.path.join(frame_folder, filename)) and os.path.join(frame_folder, filename).endswith(
                "jpg")
        ]

        evaluate_yolo(valid_batch, output_dir, excel_rows)

df = pd.DataFrame(excel_rows)
df.to_excel("yolo_eval.xlsx", index=False)