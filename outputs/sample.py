import time
import depth_pro
import cv2
import os
from ultralytics import YOLO


# https://medium.com/@Nivitus./estimating-depth-and-focal-length-with-apple-depth-pro-e49a19392b47






def process_image(image_dir, output_dir, save_photo):
    global transform, depth_model

    valid_batch = [
        os.path.join(image_dir, filename)
        for filename in os.listdir(image_dir)
        if os.path.exists(os.path.join(image_dir, filename)) and os.path.join(image_dir, filename).endswith(("jpg", "jpeg", "JPG", "PNG"))
    ]
    print(valid_batch)

    os.makedirs(output_dir, exist_ok=True)

    # Loading the YOLO model...
    model = YOLO("/home/yolo_code/yolo11n.pt")
    model.to("cuda")
    results = model(valid_batch)

    # save_path = os.path.join(output_dir, filename)
    # annotated_image = detection.plot()
    # cv2.imwrite(save_path, annotated_image)

    for i in range(len(valid_batch)):
        print(valid_batch[i])
        detection = results[i]

        save_path = None

        if save_photo:
            save_path = os.path.join(output_dir, os.path.basename(valid_batch[i]))

        if save_path is not None:
            annotated_image = detection.plot()
            cv2.imwrite(save_path, annotated_image)

        detected_boxes = []
        for box in detection.boxes:
            cls = int(box.cls[0])
            cls_name = detection.names[cls]
            xyxy = list(box.xyxy[0])
            conf = round(float(box.conf[0]), 2)
            detected_boxes.append((int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), cls_name, conf))

        if detected_boxes:
            depth_model, transform = depth_pro.create_model_and_transforms()
            depth_model = depth_model.to("cuda")  # Move depth model to GPU
            depth_model.eval()

            image, _, f_px = depth_pro.load_rgb(valid_batch[i])
            depth_input = transform(image).to("cuda")  # Move input tensor to GPU
            prediction = depth_model.infer(depth_input, f_px=f_px)
            depth = prediction["depth"]
            depth_np = depth.squeeze().cpu().numpy()

            for x1, y1, x2, y2, cls_name, conf in detected_boxes:
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                depth_value = round(float(depth_np[center_y, center_x]), 2)

                if save_path is not None:
                    image = cv2.imread(save_path)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(image, str(depth_value) + "m", (center_x, center_y), font, 1, (0, 255, 0), 2)
                    cv2.imwrite(save_path, image)


video_file = "../../final-splits/chair_deg0_light1.mp4"
frames_dir = "../frames"
extract_frames(video_file, frames_dir, frame_interval=1)


process_image(frames_dir,
              "../output",
              True)
