# --- MODIFIED FILE: color_seg.py ---

import os
import re
import random
import argparse
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PIL import Image
from PIL.Image import Resampling
import numpy as np
from io import BytesIO
from sklearn.cluster import KMeans
import colorsys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Mutator")

# Constants
PAGE_WIDTH = 595
PAGE_HEIGHT = 842
OUTPUT_DIR = "org_mut"
FONTS = ["Helvetica"]

# HSV Shift (Segmented)
def hsv_shift_clipart_segmented(img: Image.Image) -> Image.Image:
    try:
        img = img.convert("RGBA")
        arr = np.array(img).astype(np.float32)
        alpha = arr[..., 3]
        mask = alpha > 0
        if not np.any(mask):
            return img

        rgb = arr[mask, :3] / 255.0
        hsv = np.array([colorsys.rgb_to_hsv(*px) for px in rgb])
        kmeans = KMeans(n_clusters=min(5, len(hsv)), n_init=10)
        labels = kmeans.fit_predict(hsv)
        target_label = random.choice(np.unique(labels).tolist())

        seg_mask = np.zeros(mask.shape, dtype=bool)
        seg_mask[np.where(mask)] = labels == target_label

        hue_shift = random.uniform(-50, 50) / 360.0
        sat_shift = random.uniform(-0.5, 0.5)
        val_shift = random.uniform(-0.5, 0.5)

        hsv_seg = np.array([colorsys.rgb_to_hsv(*px) for px in arr[seg_mask, :3] / 255.0])
        hsv_seg[:, 0] = (hsv_seg[:, 0] + hue_shift) % 1.0
        hsv_seg[:, 1] = np.clip(hsv_seg[:, 1] + sat_shift, 0, 1)
        hsv_seg[:, 2] = np.clip(hsv_seg[:, 2] + val_shift, 0, 1)

        rgb_new = np.array([colorsys.hsv_to_rgb(*px) for px in hsv_seg]) * 255
        arr[seg_mask, :3] = rgb_new
        return Image.fromarray(arr.astype(np.uint8), 'RGBA')

    except Exception as e:
        logger.error(f"Color shift failed: {e}")
        return img

def draw_text(c, item):
    c.saveState()
    c.setFont(item["font_name"], item["font_size"])
    c.setFillColorRGB(0, 0, 0)
    c.translate(item["x"], item["y"])
    c.rotate(item["angle"])
    c.drawString(0, 0, item["text"])
    c.restoreState()

def generate_text_items(n=25):
    texts = ["Rx", "Capecitabina", "Via oral", "120 comprimidos", "Validade:"]
    return [{
        "text": random.choice(texts),
        "x": random.randint(20, PAGE_WIDTH - 150),
        "y": random.randint(20, PAGE_HEIGHT - 50),
        "font_size": random.randint(8, 24),
        "font_name": random.choice(FONTS),
        "angle": random.choice([0, 30, -30, 90, -90])
    } for _ in range(n)]

def draw_pdf(path, cliparts, text_items, mutate=False):
    base_img = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), (255, 255, 255))
    for file in cliparts:
        try:
            img = Image.open(file).convert("RGBA")
            w, h = random.randint(60, 150), random.randint(60, 150)
            x, y = random.randint(0, PAGE_WIDTH - w), random.randint(0, PAGE_HEIGHT - h)
            angle = 0
            img = img.resize((w, h), Resampling.LANCZOS)

            if mutate:
                if random.random() < 0.7:
                    if random.random() < 0.5:
                        angle = random.choice([30, -30, 45, -45])
                        img = img.rotate(angle, expand=True, resample=Image.BICUBIC)
                    img = hsv_shift_clipart_segmented(img)

            x = min(max(0, x), PAGE_WIDTH - img.width)
            y = min(max(0, y), PAGE_HEIGHT - img.height)
            base_img.paste(img, (x, y), img)
        except Exception as e:
            logger.warning(f"Clipart error {file}: {e}")

    buf = BytesIO()
    base_img.save(buf, format="PNG")
    buf.seek(0)
    c = canvas.Canvas(path, pagesize=(PAGE_WIDTH, PAGE_HEIGHT))
    c.drawImage(ImageReader(buf), 0, PAGE_HEIGHT - base_img.size[1], width=PAGE_WIDTH, height=PAGE_HEIGHT)
    for item in text_items:
        draw_text(c, item)
    c.showPage()
    c.save()
    logger.info(f"Saved: {path}")

def main():
    INPUT_FOLDER = "output_pdfs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = sorted(f for f in os.listdir(INPUT_FOLDER) if f.endswith(".pdf"))
    originals = [f for f in files if "original" in f]
    mutants = [f for f in files if "mutated" in f]

    for orig_pdf, mut_pdf in zip(originals, mutants):
        match = re.search(r'(\d+)', orig_pdf)
        serial = match.group(1) if match else "001"

        output_orig = os.path.join(OUTPUT_DIR, f"output_{serial}_original.pdf")
        output_mut = os.path.join(OUTPUT_DIR, f"output_{serial}_mutated.pdf")

        cliparts = [os.path.join(INPUT_FOLDER, f) for f in os.listdir(INPUT_FOLDER) if f.lower().endswith('.png')]
        if not cliparts:
            logger.error("No PNG cliparts found in input folder.")
            return

        text_items = generate_text_items()
        draw_pdf(output_orig, cliparts, text_items, mutate=False)
        draw_pdf(output_mut, cliparts, text_items, mutate=True)

if __name__ == "__main__":
    main()

# --- END MODIFIED FILE ---
