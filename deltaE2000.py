
import os
import random
import numpy as np
from PIL import Image
from PIL.Image import Resampling
import pytesseract
from reportlab.pdfgen import canvas
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

# Constants
PAGE_WIDTH = 595
PAGE_HEIGHT = 842
CLIPART_FOLDER = "clipart"
OUTPUT_FOLDER = "delta2000_output_01"
CLIPART_SIZE = (180, 180)
TEXT_COLOR = (0, 0, 0)  # Black
TEXT_ROTATION = 0
TEXT_FONT_SIZE = 16
TEXT_POSITION = (100, 700)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ========== CUSTOM FUNCTIONS ==========

def paste_clipart(base_img):
    clipart_files = [f for f in os.listdir(CLIPART_FOLDER) if f.lower().endswith('.png')]
    if not clipart_files:
        return base_img

    canvas_width, canvas_height = base_img.size

    for _ in range(random.randint(3, 7)):
        try:
            clip_path = os.path.join(CLIPART_FOLDER, random.choice(clipart_files))
            clip = Image.open(clip_path).convert("RGBA")
            scale = random.uniform(0.2, 0.6)
            new_width = max(1, int(clip.width * scale))
            new_height = max(1, int(clip.height * scale))
            clip = clip.resize((new_width, new_height), Resampling.LANCZOS)

            max_x = max(0, canvas_width - clip.width)
            max_y = max(0, canvas_height - clip.height)
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)

            base_img.paste(clip, (x, y), clip)
        except Exception as e:
            print(f"Error pasting clipart {clip_path}: {e}")
            continue

    return base_img

def draw_text(c, text, x, y, angle, font_size, text_color, font_name="Helvetica"):
    c.saveState()
    c.setFillColorRGB(*[v / 255.0 for v in text_color])
    if font_name not in c.getAvailableFonts():
        font_name = "Helvetica"
    c.setFont(font_name, font_size)
    c.translate(x, y)
    c.rotate(angle)
    c.drawString(0, 0, text)
    c.restoreState()

def extract_text_from_image(image):
    gray = image.convert("L")
    text = pytesseract.image_to_string(gray, config='--psm 6')
    return text.strip() or "No Text Found"

def hsv_shift(image, h_shift=0.1):
    img = image.convert("RGBA")
    arr = np.array(img)
    r, g, b, a = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
    mask = a > 0

    hsv = np.array(Image.fromarray(np.stack([r, g, b], axis=-1)).convert("HSV"))
    hsv[..., 0][mask] = (hsv[..., 0][mask].astype(np.float32) + h_shift * 255) % 255
    new_rgb = np.array(Image.fromarray(hsv.astype('uint8'), 'HSV').convert('RGB'))

    result = np.dstack([new_rgb, a])
    return Image.fromarray(result.astype('uint8'), 'RGBA')

def calculate_delta_e(img1, img2, output_path):
    img1 = img1.convert("RGB")
    img2 = img2.convert("RGB")

    arr1 = np.asarray(img1).astype(np.float32)
    arr2 = np.asarray(img2).astype(np.float32)

    gray_array = np.zeros((img1.height, img1.width), dtype=np.uint8)

    for y in range(img1.height):
        for x in range(img1.width):
            rgb1 = sRGBColor(*arr1[y, x] / 255, is_upscaled=False)
            rgb2 = sRGBColor(*arr2[y, x] / 255, is_upscaled=False)
            lab1 = convert_color(rgb1, LabColor)
            lab2 = convert_color(rgb2, LabColor)
            delta = delta_e_cie2000(lab1, lab2)
            gray = int(np.clip(delta * 12, 0, 255))  # Scaling for visibility
            gray_array[y, x] = gray

    # Create true grayscale image
    gray_img = Image.fromarray(gray_array, mode="L")
    gray_img.save(output_path)
    print(f"[✔] Grayscale Delta E image saved: {output_path}")    

def create_pdf(pdf_path, image_text_pairs):
    c = canvas.Canvas(pdf_path, pagesize=(PAGE_WIDTH, PAGE_HEIGHT))

    for image_path, text in image_text_pairs:
        draw_text(c, text, TEXT_POSITION[0], TEXT_POSITION[1], TEXT_ROTATION, TEXT_FONT_SIZE, TEXT_COLOR)
        c.drawImage(image_path, 100, 400, width=CLIPART_SIZE[0], height=CLIPART_SIZE[1], mask='auto')
        c.showPage()

    c.save()

# ========== MAIN ==========


def main():
    original_pages = []
    mutated_pages = []

    for idx in range(3):  # Generate 3 pages
        base_img = Image.new("RGBA", (300, 300), (255, 255, 255, 255))
        composited = paste_clipart(base_img)
        text = f"image {idx + 1}"

        mutated = hsv_shift(composited)

        orig_path = os.path.join(OUTPUT_FOLDER, f"original_{idx}.png")
        mutated_path = os.path.join(OUTPUT_FOLDER, f"mutated_{idx}.png")
        delta_path = os.path.join(OUTPUT_FOLDER, f"delta_diff_{idx}.png")

        composited.save(orig_path)
        mutated.save(mutated_path)

        # Generate Delta difference for each image
        calculate_delta_e(composited, mutated, delta_path)

        original_pages.append((orig_path, text))
        mutated_pages.append((mutated_path, text))

    create_pdf(os.path.join(OUTPUT_FOLDER, "original.pdf"), original_pages)
    create_pdf(os.path.join(OUTPUT_FOLDER, "mutated.pdf"), mutated_pages)

if __name__ == "__main__":
    main()


