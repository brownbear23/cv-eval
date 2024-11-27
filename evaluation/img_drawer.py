import cv2

def draw_on_image(original_image, depth, cls_name=None, conf=None, bound=None):
    if bound:
        pt1 = (int(bound[0]), int(bound[1]))  # Top-left corner
        pt2 = (int(bound[2]), int(bound[3]))  # Bottom-right corner
        cv2.rectangle(original_image, pt1, pt2, (255, 0, 0), 2)
        text_x = int(bound[0])
        text_y = int(bound[1] - 5 if bound[1] - 5 > 10 else bound[1] + 20)
    else:
        height, width, _ = original_image.shape
        text_x = int(width // 2)
        text_y = int(height // 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    if cls_name and conf:
        label = f"{cls_name} {conf:.2f} | {depth}m"
    else:
        label = f"{depth}m"

    cv2.putText(
        original_image,
        label,
        (text_x, text_y),
        font,
        0.8, (0, 255, 0), 2)
