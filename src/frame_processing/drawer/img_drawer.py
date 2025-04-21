import cv2
import textwrap

def draw_on_image(original_image, depth, cls_name=None, conf_score=None, bound=None):
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
    if cls_name and conf_score:
        label = f"{cls_name} {conf_score:.2f} | {depth}m"
    else:
        label = f"{depth}m"

    cv2.putText(
        original_image,
        label,
        (text_x, text_y),
        font,
        0.8, (0, 255, 0), 2)



import cv2
import textwrap

def draw_on_image_gpt(original_image, depth, class_score_str):
    height, width, _ = original_image.shape
    center_x = width // 2
    center_y = height // 2

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Settings for the center asterisk
    asterisk_font_scale = 1.2
    asterisk_thickness = 2
    color = (0, 255, 0)

    # Draw the center '*'
    cv2.putText(original_image, "*", (center_x, center_y), font, asterisk_font_scale, color, asterisk_thickness)

    # Build and wrap the label string
    full_label = f"{class_score_str} | {depth}m"
    max_width_px = int(width * 0.9)
    approx_char_width = 16  # Adjust this as needed for the new font scale
    max_chars_per_line = max(1, max_width_px // approx_char_width)

    wrapped_lines = textwrap.wrap(full_label, width=max_chars_per_line)

    # Settings for the wrapped text
    text_font_scale = 0.9  # Bigger text
    text_thickness = 2     # Bolder text
    line_height = int(40 * text_font_scale)

    # Draw wrapped text lines underneath the asterisk
    for i, line in enumerate(wrapped_lines):
        y_offset = center_y + (i + 1) * line_height
        text_size = cv2.getTextSize(line, font, text_font_scale, text_thickness)[0]
        x_offset = center_x - text_size[0] // 2
        cv2.putText(original_image, line, (x_offset, y_offset), font, text_font_scale, color, text_thickness)

