from PIL import ImageDraw, ImageFont

def draw_on_image(original_image, depth, cls_name=None, conf=None, bound=None):
    """
    Draws a bounding box and optional text on a Pillow image.
    
    Args:
        original_image (PIL.Image.Image): The image as a Pillow Image object.
        depth (float): Depth value to display.
        cls_name (str): Class name to display.
        conf (float): Confidence score to display.
        bound (list): Bounding box coordinates [x1, y1, x2, y2].
    
    Returns:
        PIL.Image.Image: Annotated Pillow image.
    """
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    except IOError:
        print("TrueType font not found, using default font.")
        font = ImageFont.load_default()


    # Ensure the image is in RGB mode
    if original_image.mode != "RGB":
        original_image = original_image.convert("RGB")

    # Create a drawing context
    draw = ImageDraw.Draw(original_image)

    # Draw bounding box if `bound` is provided
    if bound:
        pt1 = (int(bound[0]), int(bound[1]))  # Top-left corner
        pt2 = (int(bound[2]), int(bound[3]))  # Bottom-right corner
        draw.rectangle([pt1, pt2], outline="red", width=2)
        text_x = pt1[0]
        text_y = pt1[1] - 15 if pt1[1] - 15 > 0 else pt1[1] + 10
    else:
        # Default text position at the center
        width, height = original_image.size
        text_x = width // 2
        text_y = height // 2

    # Prepare label text
    if cls_name and conf:
        label = f"{cls_name} {conf:.2f} | {depth}m"
    else:
        label = f"{depth}m"

    # Draw the text
    draw.text((text_x, text_y), label, fill="green", font=font)

    return original_image
