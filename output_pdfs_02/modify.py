# # import os
# # import random
# # from io import BytesIO
# # from reportlab.pdfgen import canvas
# # from reportlab.lib.utils import ImageReader
# # from PIL import Image, ImageDraw, ImageFont
# # from PIL.Image import Resampling

# # # Constants
# # PAGE_WIDTH = 595
# # PAGE_HEIGHT = 842
# # FONTS = ["Helvetica"]
# # CLIPART_FOLDER = "clipart"
# # OUTPUT_DIR = "output_pdfs_01"

# # def get_clipart_files():
# #     return [os.path.join(CLIPART_FOLDER, f) for f in os.listdir(CLIPART_FOLDER) if f.lower().endswith('.png')]

# # def create_clipart_variants(clipart_files, base_img_size, mutate=False):
# #     canvas_width, canvas_height = base_img_size
# #     variants = []
# #     for clip_path in random.sample(clipart_files, min(5, len(clipart_files))):
# #         try:
# #             clip = Image.open(clip_path).convert("RGBA")
# #             base_scale = random.uniform(0.2, 0.5)
# #             scale = base_scale + (random.uniform(-0.05, 0.05) if mutate else 0)
# #             scale = max(0.1, min(scale, 1.0))
# #             new_width = int(clip.width * scale)
# #             new_height = int(clip.height * scale)

# #             max_x = canvas_width - new_width
# #             max_y = canvas_height - new_height
# #             x = random.randint(0, max_x)
# #             y = random.randint(0, max_y)

# #             if mutate:
# #                 x = min(max(0, x + random.randint(-5, 5)), max_x)
# #                 y = min(max(0, y + random.randint(-5, 5)), max_y)

# #             angle = random.randint(-10, 10) if mutate else 0

# #             variants.append({
# #                 "path": clip_path,
# #                 "x": x,
# #                 "y": y,
# #                 "width": new_width,
# #                 "height": new_height,
# #                 "angle": angle
# #             })
# #         except Exception as e:
# #             print(f"Error loading clipart {clip_path}: {e}")
# #     return variants

# # def paste_cliparts(base_img, variants):
# #     for var in variants:
# #         try:
# #             clip = Image.open(var["path"]).convert("RGBA")
# #             clip = clip.resize((var["width"], var["height"]), Resampling.LANCZOS)
# #             if var["angle"] != 0:
# #                 clip = clip.rotate(var["angle"], expand=True, resample=Image.BICUBIC)
# #             base_img.paste(clip, (var["x"], var["y"]), clip)
# #         except Exception as e:
# #             print(f"Error pasting clipart: {e}")
# #     return base_img

# # def generate_text_items(num=25):
# #     text_samples = [
# #         "Capecitabina", "120 comprimidos", "Via oral", "Rx", "Uso hospitalar",
# #         "Advertencia", "Consulte Médico", "Manter fora do alcance", "Leia a bula", "Uso adulto"
# #     ]
# #     items = []
# #     for _ in range(num):
# #         text = random.choice(text_samples)
# #         font_size = random.randint(12, 24)
# #         angle = random.choice([0, 90, 180, 270, 30, -45])
# #         x = random.randint(20, PAGE_WIDTH - 150)
# #         y = random.randint(20, PAGE_HEIGHT - 50)
# #         items.append({
# #             'text': text, 'x': x, 'y': y,
# #             'font_size': font_size,
# #             'font_name': random.choice(FONTS),
# #             'angle': angle
# #         })
# #     return items

# # def draw_text(c, text, x, y, angle, font_size, color=(0, 0, 0), font_name="Helvetica"):
# #     c.saveState()
# #     c.setFont(font_name, font_size)
# #     c.setFillColorRGB(*[v / 255 for v in color])
# #     c.translate(x, y)
# #     c.rotate(angle)
# #     c.drawString(0, 0, text)
# #     c.restoreState()

# # def draw_all_text(c, text_items):
# #     for item in text_items:
# #         draw_text(c, item['text'], item['x'], item['y'], item['angle'], item['font_size'])

# # def create_pdf(filename, text_items, clipart_variants):
# #     c = canvas.Canvas(filename, pagesize=(PAGE_WIDTH, PAGE_HEIGHT))
# #     base_img = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), (255, 255, 255))
# #     base_img = paste_cliparts(base_img, clipart_variants)

# #     img_buffer = BytesIO()
# #     base_img.save(img_buffer, format="PNG")
# #     img_buffer.seek(0)
# #     c.drawImage(ImageReader(img_buffer), 0, 0, width=PAGE_WIDTH, height=PAGE_HEIGHT)

# #     draw_all_text(c, text_items)
# #     c.showPage()
# #     c.save()

# # # Main Execution
# # if __name__ == "__main__":
# #     os.makedirs(OUTPUT_DIR, exist_ok=True)

# #     # Serial number handling
# #     existing = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".pdf")]
# #     numbers = [int(f.split("_")[1]) for f in existing if f.startswith("output_") and f.split("_")[1].isdigit()]
# #     serial = max(numbers, default=0) + 1
# #     serial_str = f"{serial:03d}"

# #     # PDF paths
# #     orig_pdf = os.path.join(OUTPUT_DIR, f"output_{serial_str}_original.pdf")
# #     mut_pdf = os.path.join(OUTPUT_DIR, f"output_{serial_str}_mutated.pdf")

# #     # Common base data
# #     clipart_files = get_clipart_files()
# #     base_image_size = (PAGE_WIDTH, PAGE_HEIGHT)
# #     text_items = generate_text_items()
# #     original_variants = create_clipart_variants(clipart_files, base_image_size, mutate=False)

# # # Decide one mutation type to apply for this run
# # chosen_mutation = random.choice(["shift", "resize", "rotate"])
# # print(f"Chosen mutation type for this run: {chosen_mutation}")

# # mutated_variants = []
# # num_to_mutate = int(0.3 * len(original_variants))
# # indices_to_mutate = random.sample(range(len(original_variants)), num_to_mutate)

# # for idx, var in enumerate(original_variants):
# #     mutated = var.copy()

# #     if idx in indices_to_mutate:
# #         if chosen_mutation == "shift":
# #             shift_range = 15
# #             mutated["x"] = min(max(0, var["x"] + random.randint(-shift_range, shift_range)), PAGE_WIDTH - var["width"])
# #             mutated["y"] = min(max(0, var["y"] + random.randint(-shift_range, shift_range)), PAGE_HEIGHT - var["height"])

# #         elif chosen_mutation == "resize":
# #             resize_factor = 0.2
# #             mutated["width"] = max(1, int(var["width"] * random.uniform(1 - resize_factor, 1 + resize_factor)))
# #             mutated["height"] = max(1, int(var["height"] * random.uniform(1 - resize_factor, 1 + resize_factor)))

# #         elif chosen_mutation == "rotate":
# #             mutated["angle"] = var["angle"] + random.randint(-10, 10)

# #     mutated_variants.append(mutated)




# #     # Mutate original clipart slightly
# # # mutated_variants = []
# # # for var in original_variants:
# # #     mutated = var.copy()
# # #     shift_range = 15  # was 5
# # #     resize_factor = 0.2  # was ~0.05
# # #     mutated["x"] = min(max(0, var["x"] + random.randint(-shift_range, shift_range)), PAGE_WIDTH - var["width"])
# # #     mutated["y"] = min(max(0, var["y"] + random.randint(-shift_range, shift_range)), PAGE_HEIGHT - var["height"])
# # #     mutated["angle"] = var["angle"] + random.randint(-10, 10)
# # #     mutated["width"] = max(1, int(var["width"] * random.uniform(1 - resize_factor, 1 + resize_factor)))
# # #     mutated["height"] = max(1, int(var["height"] * random.uniform(1 - resize_factor, 1 + resize_factor)))
# # #     mutated_variants.append(mutated)


# #     # Mutate original clipart slightly
# #     # mutated_variants = []
# #     # for var in original_variants:
# #     #     mutated = var.copy()
# #     #     mutated["x"] = min(max(0, var["x"] + random.randint(-5, 5)), PAGE_WIDTH - var["width"])
# #     #     mutated["y"] = min(max(0, var["y"] + random.randint(-5, 5)), PAGE_HEIGHT - var["height"])
# #     #     mutated["angle"] = var["angle"] + random.randint(-5, 5)
# #     #     mutated["width"] = max(1, int(var["width"] * random.uniform(0.95, 1.05)))
# #     #     mutated["height"] = max(1, int(var["height"] * random.uniform(0.95, 1.05)))
# #     #     mutated_variants.append(mutated)

# #     # Generate PDFs
# #     create_pdf(orig_pdf, text_items, original_variants)
# #     create_pdf(mut_pdf, text_items, mutated_variants)

# #     print(f"PDFs saved:\n{orig_pdf}\n{mut_pdf}")

# ###############################################################################################################



# import os
# import random
# from io import BytesIO
# from reportlab.pdfgen import canvas
# from reportlab.lib.utils import ImageReader
# from PIL import Image, ImageDraw, ImageFont
# from PIL.Image import Resampling

# # Constants
# PAGE_WIDTH = 595
# PAGE_HEIGHT = 842
# FONTS = ["Helvetica"]
# CLIPART_FOLDER = "clipart"
# OUTPUT_DIR = "output_pdfs_02"

# def get_clipart_files():
#     return [os.path.join(CLIPART_FOLDER, f) for f in os.listdir(CLIPART_FOLDER) if f.lower().endswith('.png')]

# def create_clipart_variants(clipart_files, base_img_size, mutate=False):
#     canvas_width, canvas_height = base_img_size
#     variants = []
#     for clip_path in random.sample(clipart_files, min(5, len(clipart_files))):
#         try:
#             clip = Image.open(clip_path).convert("RGBA")
#             base_scale = random.uniform(0.2, 0.5)
#             scale = base_scale + (random.uniform(-0.05, 0.05) if mutate else 0)
#             scale = max(0.1, min(scale, 1.0))
#             new_width = int(clip.width * scale)
#             new_height = int(clip.height * scale)

#             max_x = canvas_width - new_width
#             max_y = canvas_height - new_height
#             x = random.randint(0, max_x)
#             y = random.randint(0, max_y)

#             if mutate:
#                 x = min(max(0, x + random.randint(-5, 5)), max_x)
#                 y = min(max(0, y + random.randint(-5, 5)), max_y)

#             angle = random.randint(-10, 10) if mutate else 0

#             variants.append({
#                 "path": clip_path,
#                 "x": x,
#                 "y": y,
#                 "width": new_width,
#                 "height": new_height,
#                 "angle": angle
#             })
#         except Exception as e:
#             print(f"Error loading clipart {clip_path}: {e}")
#     return variants

# def paste_cliparts(base_img, variants):
#     for var in variants:
#         try:
#             clip = Image.open(var["path"]).convert("RGBA")
#             clip = clip.resize((var["width"], var["height"]), Resampling.LANCZOS)
#             if var["angle"] != 0:
#                 clip = clip.rotate(var["angle"], expand=True, resample=Image.BICUBIC)
#             base_img.paste(clip, (var["x"], var["y"]), clip)
#         except Exception as e:
#             print(f"Error pasting clipart: {e}")
#     return base_img

# def generate_text_items(num=25):
#     text_samples = [
#         "Capecitabina", "120 comprimidos", "Via oral", "Rx", "Uso hospitalar",
#         "Advertencia", "Consulte Médico", "Manter fora do alcance", "Leia a bula", "Uso adulto"
#     ]
#     items = []
#     for _ in range(num):
#         text = random.choice(text_samples)
#         font_size = random.randint(12, 24)
#         angle = random.choice([0, 90, 180, 270, 30, -45])
#         x = random.randint(20, PAGE_WIDTH - 150)
#         y = random.randint(20, PAGE_HEIGHT - 50)
#         items.append({
#             'text': text, 'x': x, 'y': y,
#             'font_size': font_size,
#             'font_name': random.choice(FONTS),
#             'angle': angle
#         })
#     return items

# def draw_text(c, text, x, y, angle, font_size, color=(0, 0, 0), font_name="Helvetica"):
#     c.saveState()
#     c.setFont(font_name, font_size)
#     c.setFillColorRGB(*[v / 255 for v in color])
#     c.translate(x, y)
#     c.rotate(angle)
#     c.drawString(0, 0, text)
#     c.restoreState()

# def draw_all_text(c, text_items):
#     for item in text_items:
#         draw_text(c, item['text'], item['x'], item['y'], item['angle'], item['font_size'])

# def create_pdf(filename, text_items, clipart_variants):
#     c = canvas.Canvas(filename, pagesize=(PAGE_WIDTH, PAGE_HEIGHT))
#     base_img = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), (255, 255, 255))
#     base_img = paste_cliparts(base_img, clipart_variants)

#     img_buffer = BytesIO()
#     base_img.save(img_buffer, format="PNG")
#     img_buffer.seek(0)
#     c.drawImage(ImageReader(img_buffer), 0, 0, width=PAGE_WIDTH, height=PAGE_HEIGHT)

#     draw_all_text(c, text_items)
#     c.showPage()
#     c.save()

# # Main Execution
# if __name__ == "__main__":
#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     # Serial number handling
#     existing = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".pdf")]
#     numbers = [int(f.split("_")[1]) for f in existing if f.startswith("output_") and f.split("_")[1].isdigit()]
#     serial = max(numbers, default=0) + 1
#     serial_str = f"{serial:03d}"

#     # PDF paths
#     orig_pdf = os.path.join(OUTPUT_DIR, f"output_{serial_str}_original.pdf")
#     mut_pdf = os.path.join(OUTPUT_DIR, f"output_{serial_str}_mutated.pdf")

#     # Common base data
#     clipart_files = get_clipart_files()
#     base_image_size = (PAGE_WIDTH, PAGE_HEIGHT)
#     text_items = generate_text_items()
#     original_variants = create_clipart_variants(clipart_files, base_image_size, mutate=False)

#         # Generate mutated variants with conditional mutation
#     mutated_variants = []
#     for var in original_variants:
#         mutated = var.copy()
#         r = random.random()
#         if r > 0.2:  # 80% chance to mutate
#             selector = int(random.random() * 1000)
#             mod = selector % 6
#             if mod == 2:
#                 # Rotate visibly
#                 mutated["angle"] = var["angle"] + random.choice([-45, -30, 30, 45])
#             elif mod == 3:
#                 # Resize more dramatically
#                 resize_factor = random.choice([0.5, 0.75, 1.25, 1.5])
#                 mutated["width"] = max(10, int(var["width"] * resize_factor))
#                 mutated["height"] = max(10, int(var["height"] * resize_factor))
#             elif mod == 5:
#                 # Shift more noticeably
#                 shift_range = 100
#                 mutated["x"] = min(max(0, var["x"] + random.randint(-shift_range, shift_range)), PAGE_WIDTH - var["width"])
#                 mutated["y"] = min(max(0, var["y"] + random.randint(-shift_range, shift_range)), PAGE_HEIGHT - var["height"])
#         mutated_variants.append(mutated)


#     # Generate mutated variants with conditional mutation
#     # mutated_variants = []
#     # for var in original_variants:
#     #     mutated = var.copy()
#     #     r = random.random()
#     #     if r > 0.7:
#     #         selector = int(random.random() * 1000)
#     #         mod = selector % 6
#     #         if mod == 2:
#     #             # Rotate
#     #             mutated["angle"] = var["angle"] + random.randint(-10, 10)
#     #         elif mod == 3:
#     #             # Resize
#     #             resize_factor = 0.2
#     #             mutated["width"] = max(1, int(var["width"] * random.uniform(1 - resize_factor, 1 + resize_factor)))
#     #             mutated["height"] = max(1, int(var["height"] * random.uniform(1 - resize_factor, 1 + resize_factor)))
#     #         elif mod == 5:
#     #             # Shift
#     #             shift_range = 15
#     #             mutated["x"] = min(max(0, var["x"] + random.randint(-shift_range, shift_range)), PAGE_WIDTH - var["width"])
#     #             mutated["y"] = min(max(0, var["y"] + random.randint(-shift_range, shift_range)), PAGE_HEIGHT - var["height"])
#     #     # Append possibly mutated version
#     #     mutated_variants.append(mutated)

#     # Generate PDFs
#     create_pdf(orig_pdf, text_items, original_variants)
#     create_pdf(mut_pdf, text_items, mutated_variants)

#     print(f"PDFs saved:\n{orig_pdf}\n{mut_pdf}")


import os
import random
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PIL import Image, ImageDraw, ImageFont
from PIL.Image import Resampling

# Constants
PAGE_WIDTH = 595
PAGE_HEIGHT = 842
FONTS = ["Helvetica"]
CLIPART_FOLDER = "clipart"
OUTPUT_DIR = "output_pdfs_02"

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

def create_pdf(filename, text_items, clipart_variants, mutation):

    mutated_variants = []
    for var in clipart_variants:

        if mutation and random.random() < 0.3:
        # Determine if mutation should occur
            mutation_type = int(random.random() * 3) % 3 
            mutation_name = ["ROTATE", "RESIZE", "SHIFT"][mutation_type] 
            print(f"Mutation applied: {mutation_name}")      

            mutated = var.copy()
            if mutation_type == 0:
                # ROTATE
                mutated["angle"] = random.choice([-45, -30, 30, 45])
            elif mutation_type == 1:
                # RESIZE
                scale_factor = random.uniform(0.75, 1.25)
                mutated["width"] = max(10, int(var["width"] * scale_factor))
                mutated["height"] = max(10, int(var["height"] * scale_factor))
            elif mutation_type == 2:
                # SHIFT
                shift = 30
                mutated["x"] = min(max(0, var["x"] + random.randint(-shift, shift)), PAGE_WIDTH - var["width"])
                mutated["y"] = min(max(0, var["y"] + random.randint(-shift, shift)), PAGE_HEIGHT - var["height"])
            mutated_variants.append(mutated)
        else:
            mutated_variants.append(var.copy())
    clipart_variants = mutated_variants

    c = canvas.Canvas(filename, pagesize=(PAGE_WIDTH, PAGE_HEIGHT))
    base_img = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), (255, 255, 255))
    base_img = paste_cliparts(base_img, clipart_variants)

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

    # # Determine if mutation should occur
    # apply_mutation = random.random() < 0.3
    # mutation_type = int(random.random() * 3) % 3 if apply_mutation else -1
    # mutation_name = ["ROTATE", "RESIZE", "SHIFT"][mutation_type] if mutation_type != -1 else "NONE"
    # print(f"Mutation applied: {mutation_name}")

    # mutated_variants = []
    # for var in original_variants:
    #     mutated = var.copy()
    #     if mutation_type == 0:
    #         # ROTATE
    #         mutated["angle"] = random.choice([-45, -30, 30, 45])
    #     elif mutation_type == 1:
    #         # RESIZE
    #         scale_factor = random.choice([0.5, 0.75, 1.25, 1.5])
    #         mutated["width"] = max(10, int(var["width"] * scale_factor))
    #         mutated["height"] = max(10, int(var["height"] * scale_factor))
    #     elif mutation_type == 2:
    #         # SHIFT
    #         shift = 100
    #         mutated["x"] = min(max(0, var["x"] + random.randint(-shift, shift)), PAGE_WIDTH - var["width"])
    #         mutated["y"] = min(max(0, var["y"] + random.randint(-shift, shift)), PAGE_HEIGHT - var["height"])
    #     mutated_variants.append(mutated)

    # Generate PDFs
    create_pdf(orig_pdf, text_items, original_variants, mutation=False)
    create_pdf(mut_pdf, text_items, original_variants, mutation=True)

    print(f"PDFs saved:\n{orig_pdf}\n{mut_pdf}")
