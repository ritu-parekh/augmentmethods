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

CLIPART_FOLDER = "clipart"
PAGE_WIDTH, PAGE_HEIGHT = A4
PAGE_WIDTH = int(PAGE_WIDTH)
PAGE_HEIGHT = int(PAGE_HEIGHT)
#OUTPUT_FILE = "generated_design.pdf"
OUTPUT_FILE = f"generated_design_{random.randint(1000, 9999)}.pdf"
FONTS = ["Helvetica", "Courier", "Times-Roman"]
LINE_SPACING = 6 # Pixels of vertical space between lines in a potential sequence
OVERLAP_BUFFER = 5 # Pixels buffer for overlap checking

# Ensure clipart folder exists (create if it doesn't for the code to run without error)
if not os.path.exists(CLIPART_FOLDER):
    os.makedirs(CLIPART_FOLDER)
    print(f"Created clipart folder: {CLIPART_FOLDER}. Please add some .png images to this folder.")

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


def generate_page(c, num_placement_attempts=30): # Increased attempts slightly for potentially smaller placement area
    """Generates a single page with background clipart and text boxes."""
    # Create a white base image for background analysis
    base = Image.new("RGB", (int(PAGE_WIDTH), int(PAGE_HEIGHT)), (255, 255, 255))
    # Paste clipart onto the base image
    base = paste_clipart(base)

    # Save the base image with clipart to a buffer and draw it onto the PDF canvas
    buffer = BytesIO()
    base.save(buffer, format="PNG")
    buffer.seek(0)
    c.drawImage(ImageReader(buffer), 0, 0, width=PAGE_WIDTH, height=PAGE_HEIGHT)

    # Use PIL Draw object for text size calculations *before* drawing with ReportLab
    pil_draw = ImageDraw.Draw(base)

    # Store data for ALL successfully placed text items (individual or followers)
    placed_items_data = []

    # Add padding constant (integer)
    VISUAL_PADDING = 10
    INT_PADDING = int(VISUAL_PADDING + OVERLAP_BUFFER) # Use integer padding for randint

    # Try to place a number of text items
    for _ in range(num_placement_attempts):
        placed_this_attempt = False
        tries = 0

        while not placed_this_attempt and tries < 20: # Limit tries for placing a single item
            tries += 1

            # --- Determine if this item should follow a previous one or be standalone ---
            attempt_is_follower = False
            chosen_prev_item = None

            # ~30% chance to try being a follower, ONLY if there are items already placed
            if placed_items_data and random.random() < 0.3:
                 chosen_prev_item = random.choice(placed_items_data)
                 attempt_is_follower = True

            # --- Define the parameters for the current item ---
            text_content = random.choice(["Capecitabina", "120 comprimidos", "Via oral", "Rx", "Uso hospitalar", "Advertencia", "Consulte Médico", "Manter fora do alcance", "Leia a bula", "Uso adulto"])
            font_size = random.randint(10, 26) # Slightly adjusted size range
            font_name = random.choice(FONTS)
            angle = random.choice([0, 90, 180, 270, 65, -45, 30, -60]) # Common angles

            # If following a previous item, inherit angle and calculate position
            if attempt_is_follower:
                 angle = chosen_prev_item['angle']
                 px = chosen_prev_item['x']
                 py = chosen_prev_item['y']
                 p_width = chosen_prev_item['width'] # Width of the previous item
                 p_height = chosen_prev_item['height'] # Height of the previous item

                 # Calculate the vector offset (height + spacing) in the direction
                 # 90 degrees clockwise from the previous item's text orientation.
                 # Text orientation is given by the angle.
                 offset_distance = p_height + LINE_SPACING
                 offset_angle_rad = radians(angle + 90) # Offset direction

                 offset_x = offset_distance * cos(offset_angle_rad)
                 offset_y = offset_distance * sin(offset_angle_rad)

                 attempt_x = px + offset_x
                 attempt_y = py + offset_y

            # If not following (or no items placed yet), choose random position
            else:
                 # Calculate the unrotated dimensions for the current text/font/size using PIL
                 try:
                     temp_font_pil = ImageFont.truetype("arial.ttf", font_size)
                 except IOError:
                     temp_font_pil = ImageFont.load_default()

                 # Handle potential empty string text bbox issue
                 if not text_content:
                     temp_width = 0
                     temp_height = 0
                 else:
                     temp_bbox_pil = pil_draw.textbbox((0, 0), text_content, font=temp_font_pil)
                     temp_width = temp_bbox_pil[2] - temp_bbox_pil[0]
                     temp_height = temp_bbox_pil[3] - temp_bbox_pil[1]

                 # Ensure dimensions are positive before calculating max_dim
                 temp_width = max(1, temp_width)
                 temp_height = max(1, temp_height)


                 # Add padding and ensure space for rotated box (rough estimate)
                 # Use sqrt(w^2 + h^2) as max diagonal, which is always <= largest dim of rotated box
                 max_rotated_dim_estimate = math.sqrt(temp_width**2 + temp_height**2)

                 # Simple bounds check based on allowing the unrotated top-left to be placed
                 # within a region that allows the item + buffer to stay on page
                 max_valid_x = PAGE_WIDTH - temp_width - OVERLAP_BUFFER
                 max_valid_y = PAGE_HEIGHT - temp_height - OVERLAP_BUFFER

                 # Ensure max_valid_* is not less than padding (prevents issues if text is huge or padding is large)
                 # Use INT_PADDING for comparison and randint range
                 if max_valid_x < INT_PADDING: max_valid_x = INT_PADDING
                 if max_valid_y < INT_PADDING: max_valid_y = INT_PADDING

                 # Choose random position within this adjusted range (using integers)
                 attempt_x = random.randint(INT_PADDING, int(max_valid_x))
                 attempt_y = random.randint(INT_PADDING, int(max_valid_y))


            # --- Now calculate the actual dimensions and polygon for the candidate item ---
            # Use PIL for bbox calculation (approximation) based on final text/font/size
            try:
                item_font_pil = ImageFont.truetype("arial.ttf", font_size)
            except IOError:
                item_font_pil = ImageFont.load_default()

            # Handle potential empty string text bbox issue
            if not text_content:
                item_width = 0
                item_height = 0
            else:
                item_bbox_pil = pil_draw.textbbox((0, 0), text_content, font=item_font_pil)
                item_width = item_bbox_pil[2] - item_bbox_pil[0]
                item_height = item_bbox_pil[3] - item_bbox_pil[1]

            # If text is empty or dimensions are zero, skip placing
            if item_width <= 0 or item_height <= 0: # Ensure positive dimensions
                 # print("Skipping empty or zero/negative dimension text item.")
                 continue # Skip this attempt

            # Calculate the rotated polygon for this item at the attempted position and angle
            # Rotate around its own top-left corner (attempt_x, attempt_y)
            item_potential_bbox = (attempt_x, attempt_y, attempt_x + item_width, attempt_y + item_height)
            item_polygon = get_rotated_bbox_polygon(item_potential_bbox, (attempt_x, attempt_y), angle)

            # --- Check for Overlap ---
            # Check if the potential item's polygon *plus a buffer* intersects with ANY previously placed item's polygon
            # Expanding the new item's polygon by the buffer
            try:
                buffered_item_polygon = item_polygon.buffer(OVERLAP_BUFFER)
            except Exception as e:
                print(f"Error buffering polygon: {e}. Skipping item.")
                continue # Skip if buffering fails (can happen with invalid geometry, though rare here)


            if any(buffered_item_polygon.intersects(placed['polygon']) for placed in placed_items_data):
                # print("Overlap detected (with buffer), retrying placement...")
                continue # Try a different set of parameters/position/prev_item

            # --- Check if the item (with buffer) is reasonably within page bounds ---
            # Check if the buffered polygon's bounding box is within the page boundaries
            minx, miny, maxx, maxy = buffered_item_polygon.bounds

            # If any corner of the buffered bounding box is outside the page
            if minx < 0 or miny < 0 or maxx > PAGE_WIDTH or maxy > PAGE_HEIGHT:
                 # print("Item (with buffer) likely off-page, retrying placement...")
                 continue


            # --- If no overlap and within bounds, place the item ---
            placed_this_attempt = True

            # Sample background color under the unrotated bbox of THIS item
            avg_color = sample_region_average(base, attempt_x, attempt_y, item_width, item_height)
            text_color = get_contrasting_color(avg_color)

            # Draw the text
            draw_text(c, text_content, attempt_x, attempt_y, angle, font_size, text_color, font_name)

            # Store this item's data
            placed_items_data.append({
                'x': attempt_x,
                'y': attempt_y,
                'width': item_width,
                'height': item_height,
                'angle': angle,
                'polygon': item_polygon # Store the original (non-buffered) polygon for subsequent checks
            })
            # print(f"Placed item at ({attempt_x:.1f}, {attempt_y:.1f}) rotated {angle} deg")

        # End of tries loop for placing a single item


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


def generate_pdf(pages=1):
    """Generates a PDF with the specified number of pages in order."""
    output_file = get_ordered_filename()
    c = canvas.Canvas(output_file, pagesize=A4)
    
    for i in range(pages):
        print(f"Generating page {i + 1}...")
        generate_page(c)
        if i < pages - 1:
            c.showPage()
    
    print(f"Saving PDF to {output_file}")
    c.save()

if __name__ == "__main__":
    # Example: generate a 3-page PDF
    generate_pdf(pages=3)        







