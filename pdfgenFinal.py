import os
import random
from PIL import Image, ImageStat, ImageFont, ImageDraw
from PIL.Image import Resampling
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from io import BytesIO
import colorsys
from shapely.geometry import Polygon
from math import radians, cos, sin
import math # Import math for sqrt
import copy
import string

# Import functions and constants from color_seg.py
from color_seg import hsv_shift_clipart_segmented, create_clipart_variants, get_clipart_files, CLIPART_FOLDER, logger

PAGE_WIDTH, PAGE_HEIGHT = A4
PAGE_WIDTH = int(PAGE_WIDTH)
PAGE_HEIGHT = int(PAGE_HEIGHT)
FONTS = ["Helvetica", "Courier", "Times-Roman"]
LINE_SPACING = 6 # Pixels of vertical space between lines in a potential sequence
OVERLAP_BUFFER = 5 # Pixels buffer for overlap checking

def rotate_point(x, y, cx, cy, angle_deg):
    """Rotates a point (x, y) around a center (cx, cy) by angle_deg."""
    angle_rad = radians(angle_deg)
    cos_a, sin_a = cos(angle_rad), sin(angle_rad)
    dx, dy = x - cx, y - cy
    rx = cx + dx * cos_a - dy * sin_a
    ry = cy + dx * sin_a + dy * cos_a
    return rx, ry

def get_rotated_bbox_polygon(bbox, origin, angle_deg):
    """
    Calculates the rotated bounding box polygon for a bbox rotated around an origin.

    Args:
        bbox: A tuple (x0, y0, x1, y1) representing the unrotated bounding box.
        origin: A tuple (cx, cy) representing the point of rotation.
        angle_deg: The rotation angle in degrees.

    Returns:
        A Shapely Polygon representing the rotated bounding box.
    """
    x0, y0, x1, y1 = bbox
    cx, cy = origin

    # Define the corners of the unrotated bounding box
    corners = [
        (x0, y0), # Top-left
        (x1, y0), # Top-right
        (x1, y1), # Bottom-right
        (x0, y1)  # Bottom-left
    ]
    # Rotate each corner around the origin
    rotated_corners = [rotate_point(x, y, cx, cy, angle_deg) for (x, y) in corners]
    return Polygon(rotated_corners)

def sample_region_average(img, x, y, w, h):
    """Samples the average RGB color of a region in an image, clamping coordinates."""
    # Clamp the coordinates to be within the image bounds
    img_w, img_h = img.size
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(img_w, int(x + w))
    y1 = min(img_h, int(y + h))

    # If the clamped region is invalid (width or height <= 0), return white
    if x1 <= x0 or y1 <= y0:
        return (255, 255, 255) # White

    cropped = img.crop((x0, y0, x1, y1))
    stat = ImageStat.Stat(cropped)
    # Use mean[:3] to get RGB values, handling potential alpha channel
    if len(stat.mean) < 3: # Handle grayscale or paletted images
         if len(stat.mean) == 1: # Grayscale
              val = int(stat.mean[0])
              return (val, val, val)
         return (255, 255, 255) # Default to white for other unexpected formats

    return tuple(map(int, stat.mean[:3]))


def get_contrasting_color(rgb):
    """Calculates a contrasting color given an RGB background color."""
    # Normalize RGB to 0-1 range
    r, g, b = [x / 255.0 for x in rgb]
    try:
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
    except ValueError: # Handle potential edge cases like pure black/white
        return (0, 0, 0) if (r + g + b) / 3 > 0.5 else (255, 255, 255) # Return black for light bg, white for dark bg

    # 1. Opposite hue for direct contrast
    h = (h + 0.5) % 1.0

    # 2. Invert brightness for luminosity contrast
    v = 1.0 - v
    v = max(0.4, min(v, 0.9))  # Clamp brightness to a visible range

    # 3. Boost saturation to ensure vividness
    s = max(0.6, s)

    # Convert back to RGB
    r_c, g_c, b_c = colorsys.hsv_to_rgb(h, s, v)
    return (int(r_c * 255), int(g_c * 255), int(b_c * 255))


def draw_text(c, text, x, y, angle, font_size, text_color, font_name="Helvetica"):
    """Draws text on the ReportLab canvas with rotation."""
    c.saveState() # Save the current canvas state
    # Set fill color (ReportLab uses RGB 0-1)
    c.setFillColorRGB(*[v / 255.0 for v in text_color])
    # Set font and size
    # Ensure font_name is one of the standard ReportLab fonts if not using registerFont
    if font_name not in c.getAvailableFonts():
         font_name = "Helvetica" # Fallback

    c.setFont(font_name, font_size)
    # Translate the origin to the text's desired position (x, y)
    c.translate(x, y)
    # Rotate the canvas around the new origin (x, y)
    c.rotate(angle)
    # Draw the string at the translated and rotated origin (which is now 0,0 relative to the text's position)
    c.drawString(0, 0, text)
    c.restoreState() # Restore the canvas state


def mutate_text(text):
    """Randomly add, remove, or change one character in the text."""
    if not text or len(text) == 0:
        return text
    op = random.choice(["add", "remove", "change"])
    idx = random.randint(0, len(text) - 1)
    if op == "remove" and len(text) > 1:
        return text[:idx] + text[idx+1:]
    elif op == "add":
        char = random.choice(string.ascii_letters)
        return text[:idx] + char + text[idx:]
    elif op == "change":
        char = random.choice(string.ascii_letters)
        return text[:idx] + char + text[idx+1:]
    return text

def generate_page_and_collect(c, num_placement_attempts=30):
    """Generates a page with cliparts and text, returning placed text items and the base image."""
    # --- Clipart Mutation Logic ---
    clipart_files = get_clipart_files()
    base_image_size = (PAGE_WIDTH, PAGE_HEIGHT)
    original_variants = create_clipart_variants(clipart_files, base_image_size)

    # Create a white base image
    base = Image.new("RGB", (int(PAGE_WIDTH), int(PAGE_HEIGHT)), (255, 255, 255))

    mutated_variants_info = []
    for i, var in enumerate(original_variants):
        mutation_info = {"original": var}
        mutation_info.update(var)

        if random.random() < 0.7:
            available_mutations = ["ROTATE", "RESIZE", "SHIFT", "COLOR_SHIFT"]
            num_mutations = random.choices([1, 2], weights=[0.6, 0.4], k=1)[0]
            num_mutations = min(num_mutations, len(available_mutations))
            chosen_mutations = random.sample(available_mutations, num_mutations)

            for mut_type in chosen_mutations:
                if mut_type == "ROTATE":
                    angle = random.choice([i for i in range(-45, 45) if i != 0])
                    if angle != mutation_info.get("angle", 0):
                        mutation_info["angle"] = angle
                elif mut_type == "RESIZE":
                    scale_factor = random.uniform(0.7, 1.3)
                    current_w = mutation_info.get("width", var["width"])
                    current_h = mutation_info.get("height", var["height"])
                    new_width = max(20, int(current_w * scale_factor))
                    new_height = max(20, int(current_h * scale_factor))
                    if new_width != current_w or new_height != current_h:
                        mutation_info["width"] = new_width
                        mutation_info["height"] = new_height
                        mutation_info["x"] = min(mutation_info.get("x", var["x"]), PAGE_WIDTH - mutation_info["width"])
                        mutation_info["y"] = min(mutation_info.get("y", var["y"]), PAGE_HEIGHT - mutation_info["height"])
                        mutation_info["x"] = max(0, mutation_info["x"])
                        mutation_info["y"] = max(0, mutation_info["y"])
                elif mut_type == "SHIFT":
                    shift = 40
                    original_x, original_y = mutation_info.get("x", var["x"]), mutation_info.get("y", var["y"])
                    current_w = mutation_info.get("width", var["width"])
                    current_h = mutation_info.get("height", var["height"])
                    new_x = min(max(0, original_x + random.randint(-shift, shift)), PAGE_WIDTH - current_w)
                    new_y = min(max(0, original_y + random.randint(-shift, shift)), PAGE_HEIGHT - current_h)
                    if new_x != original_x or new_y != original_y:
                        mutation_info["x"] = new_x
                        mutation_info["y"] = new_y
                elif mut_type == "COLOR_SHIFT":
                    if not mutation_info.get("color_shift", False):
                        mutation_info["color_shift"] = True

        mutated_variants_info.append(mutation_info)

    # Paste the cliparts onto the base image
    for i, mut_info in enumerate(mutated_variants_info):
        var = mut_info["original"]
        current_x = mut_info.get("x", var["x"])
        current_y = mut_info.get("y", var["y"])
        current_width = mut_info.get("width", var["width"])
        current_height = mut_info.get("height", var["height"])
        current_angle = mut_info.get("angle", var["angle"])
        apply_color_shift = mut_info.get("color_shift", False)

        current_width = max(1, current_width)
        current_height = max(1, current_height)

        paste_x = min(max(0, current_x), PAGE_WIDTH - current_width)
        paste_y = min(max(0, current_y), PAGE_HEIGHT - current_height)

        try:
            clip = Image.open(var["path"]).convert("RGBA")
            clip = clip.resize((current_width, current_height), Resampling.LANCZOS)

            if current_angle != 0:
                clip = clip.rotate(current_angle, expand=True, resample=Image.BICUBIC)
                paste_x = min(max(0, current_x), PAGE_WIDTH - clip.width)
                paste_y = min(max(0, current_y), PAGE_HEIGHT - clip.height)

            if apply_color_shift:
                clip = hsv_shift_clipart_segmented(clip, var["path"],
                                                   hue_range=(-50, 50),
                                                   sat_range=(-0.5, 0.5),
                                                   val_range=(-0.5, 0.5),
                                                   k_clusters=(2, 8))

            base.paste(clip, (paste_x, paste_y), clip)

        except Exception as e:
            logger.error(f"Error processing and pasting clipart (path: {var['path']}, variant {i}): {e}", exc_info=True)

    # --- Text Placement Logic ---
    pil_draw = ImageDraw.Draw(base)
    placed_items_data = []
    VISUAL_PADDING = 10
    INT_PADDING = int(VISUAL_PADDING + OVERLAP_BUFFER)

    for _ in range(num_placement_attempts):
        placed_this_attempt = False
        tries = 0

        while not placed_this_attempt and tries < 20:
            tries += 1
            attempt_is_follower = False
            chosen_prev_item = None

            if placed_items_data and random.random() < 0.3:
                chosen_prev_item = random.choice(placed_items_data)
                attempt_is_follower = True

            text_content = random.choice([
                "Capecitabina", "120 comprimidos", "Via oral", "Rx", "Uso hospitalar",
                "Advertencia", "Consulte Médico", "Manter fora do alcance", "Leia a bula", "Uso adulto"
            ])
            font_size = random.randint(10, 26)
            font_name = random.choice(FONTS)
            angle = random.choice([0, 90, 180, 270, 65, -45, 30, -60])

            if attempt_is_follower:
                angle = chosen_prev_item['angle']
                px = chosen_prev_item['x']
                py = chosen_prev_item['y']
                p_width = chosen_prev_item['width']
                p_height = chosen_prev_item['height']
                offset_distance = p_height + LINE_SPACING
                offset_angle_rad = radians(angle + 90)
                offset_x = offset_distance * cos(offset_angle_rad)
                offset_y = offset_distance * sin(offset_angle_rad)
                attempt_x = px + offset_x
                attempt_y = py + offset_y
            else:
                try:
                    temp_font_pil = ImageFont.truetype("arial.ttf", font_size)
                except IOError:
                    temp_font_pil = ImageFont.load_default()

                if not text_content:
                    temp_width = 0
                    temp_height = 0
                else:
                    temp_bbox_pil = pil_draw.textbbox((0, 0), text_content, font=temp_font_pil)
                    temp_width = temp_bbox_pil[2] - temp_bbox_pil[0]
                    temp_height = temp_bbox_pil[3] - temp_bbox_pil[1]

                temp_width = max(1, temp_width)
                temp_height = max(1, temp_height)
                max_valid_x = PAGE_WIDTH - temp_width - OVERLAP_BUFFER
                max_valid_y = PAGE_HEIGHT - temp_height - OVERLAP_BUFFER
                if max_valid_x < INT_PADDING: max_valid_x = INT_PADDING
                if max_valid_y < INT_PADDING: max_valid_y = INT_PADDING
                attempt_x = random.randint(INT_PADDING, int(max_valid_x))
                attempt_y = random.randint(INT_PADDING, int(max_valid_y))

            try:
                item_font_pil = ImageFont.truetype("arial.ttf", font_size)
            except IOError:
                item_font_pil = ImageFont.load_default()

            if not text_content:
                item_width = 0
                item_height = 0
            else:
                item_bbox_pil = pil_draw.textbbox((0, 0), text_content, font=item_font_pil)
                item_width = item_bbox_pil[2] - item_bbox_pil[0]
                item_height = item_bbox_pil[3] - item_bbox_pil[1]

            if item_width <= 0 or item_height <= 0:
                continue

            item_potential_bbox = (attempt_x, attempt_y, attempt_x + item_width, attempt_y + item_height)
            item_polygon = get_rotated_bbox_polygon(item_potential_bbox, (attempt_x, attempt_y), angle)

            try:
                buffered_item_polygon = item_polygon.buffer(OVERLAP_BUFFER)
            except Exception:
                continue

            if any(buffered_item_polygon.intersects(placed['polygon']) for placed in placed_items_data):
                continue

            minx, miny, maxx, maxy = buffered_item_polygon.bounds
            if minx < 0 or miny < 0 or maxx > PAGE_WIDTH or maxy > PAGE_HEIGHT:
                continue

            placed_this_attempt = True
            avg_color = sample_region_average(base, attempt_x, attempt_y, item_width, item_height)
            text_color = get_contrasting_color(avg_color)

            placed_items_data.append({
                'x': attempt_x,
                'y': attempt_y,
                'width': item_width,
                'height': item_height,
                'angle': angle,
                'font_size': font_size,
                'font_name': font_name,
                'text': text_content,
                'text_color': text_color,
                'polygon': item_polygon
            })

    return placed_items_data, base

def draw_text_items(c, text_items):
    for item in text_items:
        draw_text(
            c, item['text'], item['x'], item['y'], item['angle'],
            item['font_size'], item['text_color'], item['font_name']
        )

def mutate_text_items(text_items, mutation_rate=0.3):
    mutated = copy.deepcopy(text_items)
    n = int(len(mutated) * mutation_rate)
    idxs = random.sample(range(len(mutated)), n)
    for i in idxs:
        mutated[i]['text'] = mutate_text(mutated[i]['text'])
    return mutated

def generate_dual_pdfs(pages=1):
    output_file_orig = get_ordered_filename(base_name="generated_design_original")
    output_file_mut = get_ordered_filename(base_name="generated_design_mutated")
    c_orig = canvas.Canvas(output_file_orig, pagesize=A4)
    c_mut = canvas.Canvas(output_file_mut, pagesize=A4)

    for page_num in range(pages):
        print(f"Generating page {page_num + 1}...")

        # --- 1. Generate the clipart layout ONCE ---
        clipart_files = get_clipart_files()
        base_image_size = (PAGE_WIDTH, PAGE_HEIGHT)
        original_variants = create_clipart_variants(clipart_files, base_image_size)

        # --- 2. Create original base image ---
        base_orig = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), (255, 255, 255))
        for var in original_variants:
            try:
                clip = Image.open(var["path"]).convert("RGBA")
                clip = clip.resize((var["width"], var["height"]), Resampling.LANCZOS)
                if var.get("angle", 0) != 0:
                    clip = clip.rotate(var["angle"], expand=True, resample=Image.BICUBIC)
                    px = min(max(0, var["x"]), PAGE_WIDTH - clip.width)
                    py = min(max(0, var["y"]), PAGE_HEIGHT - clip.height)
                else:
                    px = min(max(0, var["x"]), PAGE_WIDTH - var["width"])
                    py = min(max(0, var["y"]), PAGE_HEIGHT - var["height"])
                base_orig.paste(clip, (px, py), clip)
            except Exception as e:
                logger.error(f"Error processing original clipart: {e}", exc_info=True)

        # --- 3. Create mutated base image ---
        base_mut = base_orig.copy()
        mutated_variants = copy.deepcopy(original_variants)
        for mut in mutated_variants:
            # Apply random mutations
            if random.random() < 0.7:
                available_mutations = ["ROTATE", "RESIZE", "SHIFT", "COLOR_SHIFT"]
                num_mutations = random.choices([1, 2], weights=[0.6, 0.4], k=1)[0]
                chosen_mutations = random.sample(available_mutations, num_mutations)
                for mut_type in chosen_mutations:
                    if mut_type == "ROTATE":
                        angle = random.choice([i for i in range(-45, 45) if i != 0])
                        mut["angle"] = angle
                    elif mut_type == "RESIZE":
                        scale = random.uniform(0.7, 1.3)
                        mut["width"] = max(20, int(mut["width"] * scale))
                        mut["height"] = max(20, int(mut["height"] * scale))
                    elif mut_type == "SHIFT":
                        shift = 40
                        mut["x"] = min(max(0, mut["x"] + random.randint(-shift, shift)), PAGE_WIDTH - mut["width"])
                        mut["y"] = min(max(0, mut["y"] + random.randint(-shift, shift)), PAGE_HEIGHT - mut["height"])
                    elif mut_type == "COLOR_SHIFT":
                        mut["color_shift"] = True

        # Clear the base_mut image (white background)
        base_mut = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), (255, 255, 255))
        for var in mutated_variants:
            try:
                clip = Image.open(var["path"]).convert("RGBA")
                clip = clip.resize((var["width"], var["height"]), Resampling.LANCZOS)
                if var.get("angle", 0) != 0:
                    clip = clip.rotate(var["angle"], expand=True, resample=Image.BICUBIC)
                    px = min(max(0, var["x"]), PAGE_WIDTH - clip.width)
                    py = min(max(0, var["y"]), PAGE_HEIGHT - clip.height)
                else:
                    px = min(max(0, var["x"]), PAGE_WIDTH - var["width"])
                    py = min(max(0, var["y"]), PAGE_HEIGHT - var["height"])
                if var.get("color_shift", False):
                    clip = hsv_shift_clipart_segmented(clip, var["path"],
                        hue_range=(-50, 50), sat_range=(-0.5, 0.5), val_range=(-0.5, 0.5), k_clusters=(2, 8))
                base_mut.paste(clip, (px, py), clip)
            except Exception as e:
                logger.error(f"Error processing mutated clipart: {e}", exc_info=True)

        # --- 4. Generate and mutate text items (same layout for both) ---
        text_items_orig, _ = generate_page_and_collect(c_orig)  # Only use text placement, ignore its image
        text_items_mut = copy.deepcopy(text_items_orig)
        mutated_items = mutate_text_items(text_items_mut, mutation_rate=0.3)

        # --- 5. Draw both PDFs ---
        # Original PDF
        buffer_orig = BytesIO()
        base_orig.save(buffer_orig, format="PNG")
        buffer_orig.seek(0)
        c_orig.drawImage(ImageReader(buffer_orig), 0, 0, width=PAGE_WIDTH, height=PAGE_HEIGHT)
        draw_text_items(c_orig, text_items_orig)

        # Mutated PDF
        buffer_mut = BytesIO()
        base_mut.save(buffer_mut, format="PNG")
        buffer_mut.seek(0)
        c_mut.drawImage(ImageReader(buffer_mut), 0, 0, width=PAGE_WIDTH, height=PAGE_HEIGHT)
        draw_text_items(c_mut, mutated_items)

        if page_num < pages - 1:
            c_orig.showPage()
            c_mut.showPage()

    print(f"Saving PDFs to {output_file_orig} and {output_file_mut}")
    c_orig.save()
    c_mut.save()

OUTPUT_FOLDER = "generated_pdfs"

def get_ordered_filename(base_name="generated_design", extension=".pdf"):
    """Generates a new ordered filename like generated_design1.pdf, generated_design1.pdf..."""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    index = 1
    while True:
        filename = os.path.join(OUTPUT_FOLDER, f"{base_name}{index}{extension}")
        if not os.path.exists(filename):
            return filename
        index += 1

if __name__ == "__main__":
    generate_dual_pdfs(pages=1)







