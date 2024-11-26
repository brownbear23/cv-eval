# https://velog.io/@juneten/YOLO-v1v9
# https://github.com/ultralytics/ultralytics
'''
yolo11n.pt yolo11s.pt yolo11m.pt yolo11l.pt yolo11x.pt
'''
from ultralytics import YOLO
import pandas as pd
import os
import cv2

# photo_dir = "./photos_test/"
# used_model = "yolo11n"
used_model = "yolo11m"
# used_model = "yolo11x"
photo_dir = "../Media/photo"
output_dir = "./output_"+used_model
filenames = []
rows = []

os.makedirs("./output_"+used_model, exist_ok=True)

valid_batch = [
    os.path.join(photo_dir, filename)
    for filename in os.listdir(photo_dir)
    if os.path.exists(os.path.join(photo_dir, filename)) and os.path.join(photo_dir, filename).endswith("jpg")
]
model = YOLO(used_model+".pt")
results = model(valid_batch)

for i in range(len(valid_batch)):
    filename = os.path.basename(valid_batch[i])
    detection = results[i]
    # detection.save(filename=filename)
    save_path = os.path.join(output_dir, filename)
    annotated_image = detection.plot()
    cv2.imwrite(save_path, annotated_image)

    new_row = {"File": filename, "Classes (score, distance)": ""}

    for box in detection.boxes:
        cls = detection.names[int(box.cls[0])]
        score = round(float(box.conf[0]), 2)
        new_row["Classes (score)"] += f"{cls}({score}), "
    new_row["Classes (score)"] = new_row["Classes (score)"][:-2]
    rows.append(new_row)


df = pd.DataFrame(rows)
df.to_excel("data_" +used_model+ ".xlsx", index=False)