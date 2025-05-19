
import os
import random
import numpy as np
import glob
from fpdf import FPDF
from PIL import Image
from PIL.Image import Resampling
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color


# Constants
CLIPART_FOLDER = "clipart"
OUTPUT_FOLDER = "deltaLAB_output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ========== FUNCTIONS ==========

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
            if new_width >= canvas_width or new_height >= canvas_height:
                continue  # Skip cliparts too large to fit
            clip = clip.resize((new_width, new_height), Resampling.LANCZOS)

            x = random.randint(0, canvas_width - clip.width)
            y = random.randint(0, canvas_height - clip.height)

            base_img.paste(clip, (x, y), clip)
        except Exception as e:
            print(f"Error pasting clipart {clip_path}: {e}")
            continue

    return base_img

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

def calculate_delta_lab(img1, img2, prefix):
    img1 = img1.convert("RGB")
    img2 = img2.convert("RGB")

    arr1 = np.asarray(img1).astype(np.float32)
    arr2 = np.asarray(img2).astype(np.float32)

    delta_l = np.zeros((img1.height, img1.width), dtype=np.uint8)
    delta_a = np.zeros((img1.height, img1.width), dtype=np.uint8)
    delta_b = np.zeros((img1.height, img1.width), dtype=np.uint8)

    for y in range(img1.height):
        for x in range(img1.width):
            rgb1 = sRGBColor(*arr1[y, x] / 255, is_upscaled=False)
            rgb2 = sRGBColor(*arr2[y, x] / 255, is_upscaled=False)
            lab1 = convert_color(rgb1, LabColor)
            lab2 = convert_color(rgb2, LabColor)

            dl = lab1.lab_l - lab2.lab_l
            da = lab1.lab_a - lab2.lab_a
            db = lab1.lab_b - lab2.lab_b

            delta_l[y, x] = int(np.clip(abs(dl), 0, 255))
            delta_a[y, x] = int(np.clip(abs(da), 0, 255))
            delta_b[y, x] = int(np.clip(abs(db), 0, 255))

    Image.fromarray(delta_l, mode="L").save(os.path.join(OUTPUT_FOLDER, f"{prefix}_delta_L.png"))
    Image.fromarray(delta_a, mode="L").save(os.path.join(OUTPUT_FOLDER, f"{prefix}_delta_A.png"))
    Image.fromarray(delta_b, mode="L").save(os.path.join(OUTPUT_FOLDER, f"{prefix}_delta_B.png"))
    print(f"[✔] ΔL, Δa, Δb images saved for: {prefix}")

from fpdf import FPDF
import glob
import os

def generate_numbered_pdf_report():
    # Auto-increment report number
    existing = sorted(glob.glob(os.path.join(OUTPUT_FOLDER, "report_*.pdf")))
    report_number = 1
    if existing:
        last = os.path.basename(existing[-1])
        report_number = int(last.split("_")[1].split(".")[0]) + 1

    pdf_path = os.path.join(OUTPUT_FOLDER, f"report_{report_number:03d}.pdf")
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=10)

    pair_indices = sorted(set(
        os.path.basename(f).split('_')[1]
        for f in glob.glob(os.path.join(OUTPUT_FOLDER, "pair_*_delta_L.png"))
    ))

    for idx in pair_indices:
        prefix = f"pair_{idx}"
        original = os.path.join(OUTPUT_FOLDER, f"original_{idx}.png")
        mutated = os.path.join(OUTPUT_FOLDER, f"mutated_{idx}.png")
        delta_L = os.path.join(OUTPUT_FOLDER, f"{prefix}_delta_L.png")
        delta_A = os.path.join(OUTPUT_FOLDER, f"{prefix}_delta_A.png")
        delta_B = os.path.join(OUTPUT_FOLDER, f"{prefix}_delta_B.png")

        # Add a new page
        pdf.add_page()
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, f"Report for Pair {idx}", ln=True)

        # Common parameters
        image_w = 90  # width in mm
        image_h = 60  # height in mm
        row_gap = 10  # vertical space between rows

        # Y positions
        y1 = pdf.get_y() + 5           # Start Row 1 (Original + Mutated)
        y2 = y1 + image_h + row_gap    # Start Row 2 (ΔL + Δa)
        y3 = y2 + image_h + row_gap    # Start Row 3 (Δb only)

        # Row 1: Original and Mutated
        pdf.image(original, x=10, y=y1, w=image_w, h=image_h)
        pdf.image(mutated, x=110, y=y1, w=image_w, h=image_h)

        # Row 2: ΔL and Δa
        pdf.image(delta_L, x=10, y=y2, w=image_w, h=image_h)
        pdf.image(delta_A, x=110, y=y2, w=image_w, h=image_h)

        # Row 3: Δb (centered)
        pdf.image(delta_B, x=60, y=y3, w=image_w, h=image_h)

    pdf.output(pdf_path)
    print(f"[✔] PDF report saved: {pdf_path}")

# ========== MAIN ==========

def main():
    for idx in range(3):
        base_img = Image.new("RGBA", (300, 300), (255, 255, 255, 255))
        composited = paste_clipart(base_img)
        mutated = hsv_shift(composited)

        orig_path = os.path.join(OUTPUT_FOLDER, f"original_{idx}.png")
        mutated_path = os.path.join(OUTPUT_FOLDER, f"mutated_{idx}.png")

        composited.save(orig_path)
        mutated.save(mutated_path)

        calculate_delta_lab(composited, mutated, f"pair_{idx}")

    generate_numbered_pdf_report()

if __name__ == "__main__":
    main()

