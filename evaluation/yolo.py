# https://velog.io/@juneten/YOLO-v1v9
# https://github.com/ultralytics/ultralytics
# https://medium.com/@Nivitus./estimating-depth-and-focal-length-with-apple-depth-pro-e49a19392b47

from ultralytics import YOLO
import pandas as pd
import os
import cv2
import numpy as np
from media_processing.drawer.img_drawer import draw_on_image

MODEL = "../library/yolo11m-weights/yolo11m.pt"

def evaluate_yolo(batch, output_folder, excel_data):
    model = YOLO(MODEL)
    results = model(batch)
    for i in range(len(batch)):
        file_name = os.path.basename(batch[i])
        eval_img_path = str(os.path.join(output_folder, file_name))
        original_image = cv2.imread(batch[i])
        depth_array = np.load(os.path.splitext(batch[i])[0] + "_depth.npy")

        detection = results[i]

        new_row = {"File": file_name, "Classes(score/distance)": ""}

        if len(detection.boxes) == 0:
            height, width, _ = original_image.shape
            center_distance = round(float(depth_array[height // 2, width // 2]))
            draw_on_image(original_image, center_distance)
            cv2.imwrite(eval_img_path, original_image)
            new_row["Classes(score/distance)"] += f"None({center_distance}m)"
            excel_data.append(new_row)
            continue

        for box in detection.boxes:
            cls = int(box.cls[0])
            cls_name = detection.names[cls]
            score = round(float(box.conf[0]), 2)
            bound = list(box.xyxy[0])
            center_y = int((bound[1] + bound[3]) // 2)
            center_x = int((bound[0] + bound[2]) // 2)
            depth_value = round(float(depth_array[center_y, center_x]), 2)
            new_row["Classes(score/distance)"] += f"{cls_name}({score:.2f}/{depth_value}m),  "
            draw_on_image(original_image, depth_value, cls_name, score, bound)
        new_row["Classes(score/distance)"] = new_row["Classes(score/distance)"][:-3]
        excel_data.append(new_row)
        cv2.imwrite(eval_img_path, original_image)


frame_dir = "../media/frames"
output_dir = "../outputs/yolo11m"
excel_rows = []

os.makedirs(output_dir, exist_ok=True)
for directory in os.listdir(frame_dir):
    frame_folder = os.path.join(frame_dir, directory)
    if os.path.isdir(frame_folder):
        print("Processing folder:", directory)

        output_img_folder = os.path.join(output_dir, directory)
        os.makedirs(output_img_folder, exist_ok=True)

        valid_batch = [
            os.path.join(frame_folder, filename)
            for filename in os.listdir(frame_folder)
            if os.path.exists(os.path.join(frame_folder, filename)) and os.path.join(frame_folder, filename).endswith(
                "jpg")
        ]

        evaluate_yolo(valid_batch, output_img_folder, excel_rows)

df = pd.DataFrame(excel_rows)
df.to_excel(os.path.join(output_dir, "yolo_eval11m.xlsx"), index=False)
