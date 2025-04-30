import os
import random
import math
from math import radians, cos, sin
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PIL import Image, ImageDraw, ImageFont
from PIL.Image import Resampling

from shapely.geometry import Polygon

# Constants
PAGE_WIDTH = 595
PAGE_HEIGHT = 842
FONTS = ["Helvetica"]
OVERLAP_BUFFER = 6
LINE_SPACING = 5


CLIPART_FOLDER = "clipart"
# Sample background function (replace with actual clipart logic)
# def paste_clipart(base_image):
#     # Example: Draw a gray rectangle as clipart placeholder
#     draw = ImageDraw.Draw(base_image)
#     draw.rectangle([50, 50, 200, 200], fill=(200, 200, 200))
#     return base_image

def paste_clipart(base_img):
    """Pastes random clipart images onto the base image."""
    clipart_files = [f for f in os.listdir(CLIPART_FOLDER) if f.lower().endswith('.png')]
    if not clipart_files:
        # print(f"No .png files found in {CLIPART_FOLDER}. Skipping clipart.")
        return base_img # Return base image unchanged if no clipart

    canvas_width = base_img.width
    canvas_height = base_img.height

    # Paste between 3 and 7 clipart images
    for _ in range(random.randint(3, 7)):
        try:
            clip_path = os.path.join(CLIPART_FOLDER, random.choice(clipart_files))
            clip = Image.open(clip_path).convert("RGBA")
            scale = random.uniform(0.2, 0.6) # Random scale
            # Use LANCZOS for downsampling for quality
            new_width = int(clip.width * scale)
            new_height = int(clip.height * scale)
             # Ensure dimensions are at least 1 pixel
            new_width = max(1, new_width)
            new_height = max(1, new_height)

            clip = clip.resize((new_width, new_height), Resampling.LANCZOS)

            # Ensure clipart doesn't go out of bounds - adjust max position
            max_x = max(0, canvas_width - clip.width)
            max_y = max(0, canvas_height - clip.height)

            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            base_img.paste(clip, (x, y), clip) # Use alpha channel for pasting
        except Exception as e:
            print(f"Error pasting clipart {clip_path}: {e}")
            continue # Continue with the next clipart

    return base_img

def draw_text(c, text, x, y, angle, font_size, color, font_name):
    c.saveState()
    c.setFillColorRGB(*[v / 255 for v in color])
    c.setFont(font_name, font_size)
    c.translate(x, y)
    c.rotate(angle)
    c.drawString(0, 0, text)
    c.restoreState()

def sample_region_average(image, x, y, width, height):
    cropped = image.crop((int(x), int(y), int(x + width), int(y + height)))
    pixels = list(cropped.getdata())
    if not pixels:
        return (0, 0, 0)
    avg = tuple(sum(c[i] for c in pixels) // len(pixels) for i in range(3))
    return avg

def get_contrasting_color(rgb):
    r, g, b = rgb
    return (0, 0, 0) if (r*0.299 + g*0.587 + b*0.114) > 186 else (255, 255, 255)

def get_rotated_bbox_polygon(bbox, origin, angle):
    x0, y0, x1, y1 = bbox
    ox, oy = origin
    angle_rad = math.radians(angle)
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    rotated = []
    for x, y in corners:
        dx, dy = x - ox, y - oy
        rx = ox + dx * cos(angle_rad) - dy * sin(angle_rad)
        ry = oy + dx * sin(angle_rad) + dy * cos(angle_rad)
        rotated.append((rx, ry))
    return Polygon(rotated)

def mutate_text(text):
    if len(text) <= 1:
        return text
    operation = random.choice(["add", "remove"])
    if operation == "remove":
        index = random.randint(0, len(text) - 1)
        return text[:index] + text[index+1:]
    elif operation == "add":
        index = random.randint(0, len(text))
        random_char = random.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
        return text[:index] + random_char + text[index:]
    return text

def generate_page(c=None, num_placement_attempts=30, return_data_only=False):
    base = Image.new("RGB", (int(PAGE_WIDTH), int(PAGE_HEIGHT)), (255, 255, 255))
    base = paste_clipart(base)
    buffer = BytesIO()
    base.save(buffer, format="PNG")
    buffer.seek(0)
    if c:
        c.drawImage(ImageReader(buffer), 0, 0, width=PAGE_WIDTH, height=PAGE_HEIGHT)
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

            text_content = random.choice(["Capecitabina", "120 comprimidos", "Via oral", "Rx", "Uso hospitalar", "Advertencia", "Consulte Médico", "Manter fora do alcance", "Leia a bula", "Uso adulto"])
            font_size = random.randint(10, 26)
            font_name = random.choice(FONTS)
            angle = random.choice([0, 90, 180, 270, 65, -45, 30, -60])

            if attempt_is_follower:
                angle = chosen_prev_item['angle']
                px, py = chosen_prev_item['x'], chosen_prev_item['y']
                p_width, p_height = chosen_prev_item['width'], chosen_prev_item['height']
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
            if not return_data_only:
                avg_color = sample_region_average(base, attempt_x, attempt_y, item_width, item_height)
                text_color = get_contrasting_color(avg_color)
                draw_text(c, text_content, attempt_x, attempt_y, angle, font_size, text_color, font_name)

            placed_items_data.append({
                'x': attempt_x,
                'y': attempt_y,
                'width': item_width,
                'height': item_height,
                'angle': angle,
                'polygon': item_polygon,
                'text': text_content,
                'font_size': font_size,
                'font_name': font_name
            })

    if return_data_only:
        return base, placed_items_data

def create_pdf(filename, base_image, placed_items, mutate=False):
    c = canvas.Canvas(filename, pagesize=(PAGE_WIDTH, PAGE_HEIGHT))
    buffer = BytesIO()
    base_image.save(buffer, format="PNG")
    buffer.seek(0)
    c.drawImage(ImageReader(buffer), 0, 0, width=PAGE_WIDTH, height=PAGE_HEIGHT)

    for item in placed_items:
        text = item['text']
        if mutate and random.random() > 0.7:  # Only mutate if condition is met
            text = mutate_text(text)
        avg_color = sample_region_average(base_image, item['x'], item['y'], item['width'], item['height'])
        text_color = get_contrasting_color(avg_color)
        draw_text(c, text, item['x'], item['y'], item['angle'], item['font_size'], text_color, item['font_name'])

    # for item in placed_items:
    #     text = item['text']
    #     if mutate:
    #         text = mutate_text(text)
    #     avg_color = sample_region_average(base_image, item['x'], item['y'], item['width'], item['height'])
    #     text_color = get_contrasting_color(avg_color)
    #     draw_text(c, text, item['x'], item['y'], item['angle'], item['font_size'], text_color, item['font_name'])

    c.showPage()
    c.save()

# MAIN EXECUTION
# if __name__ == "__main__":
#     base_image, placed_items = generate_page(return_data_only=True)
#     create_pdf("output_original.pdf", base_image, placed_items, mutate=False)
#     create_pdf("output_mutated.pdf", base_image, placed_items, mutate=True)

if __name__ == "__main__":
    OUTPUT_DIR = "output_pdfs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find next serial number
    existing_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("output_") and f.endswith(".pdf")]
    serial_numbers = [int(f.split("_")[1].split(".")[0]) for f in existing_files if f.split("_")[1].split(".")[0].isdigit()]
    next_serial = max(serial_numbers, default=0) + 1
    serial_str = f"{next_serial:03d}"  # Zero-padded like 001, 002...

    # File paths
    orig_path = os.path.join(OUTPUT_DIR, f"output_{serial_str}_original.pdf")
    mut_path = os.path.join(OUTPUT_DIR, f"output_{serial_str}_mutated.pdf")

    base_image, placed_items = generate_page(return_data_only=True)
    create_pdf(orig_path, base_image, placed_items, mutate=False)
    create_pdf(mut_path, base_image, placed_items, mutate=True)

    print(f"PDFs saved as:\n{orig_path}\n{mut_path}")
