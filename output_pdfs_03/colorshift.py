import os
import random
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from PIL.Image import Resampling
import colorsys

# Constants
PAGE_WIDTH = 595
PAGE_HEIGHT = 842
FONTS = ["Helvetica"]
CLIPART_FOLDER = "clipart"
OUTPUT_DIR = "output_pdfs_03"

def get_clipart_files():
    return [os.path.join(CLIPART_FOLDER, f) for f in os.listdir(CLIPART_FOLDER) if f.lower().endswith('.png')]

def create_clipart_variants(clipart_files, base_img_size):
    canvas_width, canvas_height = base_img_size
    variants = []
    for clip_path in random.sample(clipart_files, min(5, len(clipart_files))):
        try:
            clip = Image.open(clip_path).convert("RGBA")
            scale = random.uniform(0.2, 0.5)
            new_width = int(clip.width * scale)
            new_height = int(clip.height * scale)
            max_x = canvas_width - new_width
            max_y = canvas_height - new_height
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)

            variants.append({
                "path": clip_path,
                "x": x,
                "y": y,
                "width": new_width,
                "height": new_height,
                "angle": 0  # initially no rotation
            })
        except Exception as e:
            print(f"Error loading clipart {clip_path}: {e}")
    return variants

def paste_cliparts(base_img, variants):
    for var in variants:
        try:
            clip = Image.open(var["path"]).convert("RGBA")
            clip = clip.resize((var["width"], var["height"]), Resampling.LANCZOS)
            if var["angle"] != 0:
                clip = clip.rotate(var["angle"], expand=True, resample=Image.BICUBIC)
            base_img.paste(clip, (var["x"], var["y"]), clip)
        except Exception as e:
            print(f"Error pasting clipart: {e}")
    return base_img

def generate_text_items(num=25):
    text_samples = [
        "Capecitabina", "120 comprimidos", "Via oral", "Rx", "Uso hospitalar",
        "Advertencia", "Consulte Médico", "Manter fora do alcance", "Leia a bula", "Uso adulto"
    ]
    items = []
    for _ in range(num):
        text = random.choice(text_samples)
        font_size = random.randint(12, 24)
        angle = random.choice([0, 90, 180, 270, 30, -45])
        x = random.randint(20, PAGE_WIDTH - 150)
        y = random.randint(20, PAGE_HEIGHT - 50)
        items.append({
            'text': text, 'x': x, 'y': y,
            'font_size': font_size,
            'font_name': random.choice(FONTS),
            'angle': angle
        })
    return items

def draw_text(c, text, x, y, angle, font_size, color=(0, 0, 0), font_name="Helvetica"):
    c.saveState()
    c.setFont(font_name, font_size)
    c.setFillColorRGB(*[v / 255 for v in color])
    c.translate(x, y)
    c.rotate(angle)
    c.drawString(0, 0, text)
    c.restoreState()

def draw_all_text(c, text_items):
    for item in text_items:
        draw_text(c, item['text'], item['x'], item['y'], item['angle'], item['font_size'])

import numpy as np
import random
import colorsys
from PIL import Image

def hsv_shift_clipart(
    img: Image.Image,
    hue_range=(-30, 30),            # in degrees
    sat_range=(-0.2, 0.2),          # delta: -1.0 to 1.0
    val_range=(-0.2, 0.2)           # delta: -1.0 to 1.0
) -> Image.Image:
    """
    Applies a random HSV shift to the entire RGBA image,
    modifying only non-transparent pixels.

    Parameters:
        img (PIL.Image): RGBA image.
        hue_range (tuple): Range in degrees to shift hue.
        sat_range (tuple): Range to shift saturation.
        val_range (tuple): Range to shift value (brightness).

    Returns:
        PIL.Image: HSV-shifted image with alpha preserved.
    """
    img = img.convert("RGBA")
    arr = np.array(img).astype(np.float32)
    r, g, b, a = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]

    # Normalize RGB to 0–1
    r /= 255.0
    g /= 255.0
    b /= 255.0

    # Create mask of non-transparent pixels
    mask = a > 0

    # Convert to HSV
    h, s, v = np.vectorize(colorsys.rgb_to_hsv)(r[mask], g[mask], b[mask])

    # Random shifts
    hue_shift = random.uniform(*hue_range) / 360.0
    sat_shift = random.uniform(*sat_range)
    val_shift = random.uniform(*val_range)

    # Apply shifts with clamping
    h = (h + hue_shift) % 1.0
    s = np.clip(s + sat_shift, 0, 1)
    v = np.clip(v + val_shift, 0, 1)

    # Convert back to RGB
    r_new, g_new, b_new = np.vectorize(colorsys.hsv_to_rgb)(h, s, v)

    # Scale back to 0–255
    r[mask] = r_new * 255.0
    g[mask] = g_new * 255.0
    b[mask] = b_new * 255.0

    # Recombine
    arr[..., 0] = r
    arr[..., 1] = g
    arr[..., 2] = b
    result_img = Image.fromarray(np.uint8(arr), 'RGBA')

    return result_img

def create_pdf(filename, text_items, clipart_variants, mutation):
    mutated_variants = []
    color_shift_index = random.randint(0, len(clipart_variants) - 1) if mutation else -1

    for i, var in enumerate(clipart_variants):
        mutated = var.copy()

        if mutation and random.random() < 0.3:
            mutation_type = int(random.random() * 3) % 3 
            mutation_name = ["ROTATE", "RESIZE", "SHIFT"][mutation_type]
            print(f"Mutation applied: {mutation_name}")

            if mutation_type == 0:
                mutated["angle"] = random.choice([-45, -30, 30, 45])
            elif mutation_type == 1:
                scale_factor = random.uniform(0.75, 1.25)
                mutated["width"] = max(10, int(var["width"] * scale_factor))
                mutated["height"] = max(10, int(var["height"] * scale_factor))
            elif mutation_type == 2:
                shift = 30
                mutated["x"] = min(max(0, var["x"] + random.randint(-shift, shift)), PAGE_WIDTH - var["width"])
                mutated["y"] = min(max(0, var["y"] + random.randint(-shift, shift)), PAGE_HEIGHT - var["height"])

        if mutation and i == color_shift_index:
            mutated["color_shift"] = True  # Mark this one for color shifting
            print(f"Color shift applied to clipart: {mutated['path']}")
        else:
            mutated["color_shift"] = False

        mutated_variants.append(mutated)

    clipart_variants = mutated_variants

    c = canvas.Canvas(filename, pagesize=(PAGE_WIDTH, PAGE_HEIGHT))
    base_img = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), (255, 255, 255))

    for var in clipart_variants:
        try:
            clip = Image.open(var["path"]).convert("RGBA")
            clip = clip.resize((var["width"], var["height"]), Resampling.LANCZOS)
            if var.get("angle", 0) != 0:
                clip = clip.rotate(var["angle"], expand=True, resample=Image.BICUBIC)
            if var.get("color_shift"):
                clip = hsv_shift_clipart(clip, hue_range=(-30, 30), sat_range=(-0.2, 0.2), val_range=(-0.2, 0.2))
            base_img.paste(clip, (var["x"], var["y"]), clip)
        except Exception as e:
            print(f"Error processing mutated clipart: {e}")

    img_buffer = BytesIO()
    base_img.save(img_buffer, format="PNG")
    img_buffer.seek(0)
    c.drawImage(ImageReader(img_buffer), 0, 0, width=PAGE_WIDTH, height=PAGE_HEIGHT)

    draw_all_text(c, text_items)
    c.showPage()
    c.save()

# Main Execution
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Serial number handling
    existing = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".pdf")]
    numbers = [int(f.split("_")[1]) for f in existing if f.startswith("output_") and f.split("_")[1].isdigit()]
    serial = max(numbers, default=0) + 1
    serial_str = f"{serial:03d}"

    # PDF paths
    orig_pdf = os.path.join(OUTPUT_DIR, f"output_{serial_str}_original.pdf")
    mut_pdf = os.path.join(OUTPUT_DIR, f"output_{serial_str}_mutated.pdf")

    # Load cliparts and generate text
    clipart_files = get_clipart_files()
    base_image_size = (PAGE_WIDTH, PAGE_HEIGHT)
    text_items = generate_text_items()
    original_variants = create_clipart_variants(clipart_files, base_image_size)

    # Generate PDFs
    create_pdf(orig_pdf, text_items, original_variants, mutation=False)
    create_pdf(mut_pdf, text_items, original_variants, mutation=True)

    print(f"PDFs saved:\n{orig_pdf}\n{mut_pdf}")
