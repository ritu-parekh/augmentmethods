# --- START OF FILE colorshift_seg_kmeans.py ---

import os
import random
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from PIL.Image import Resampling
import colorsys

# Import KMeans (back from SLIC)
from sklearn.cluster import KMeans

import warnings # To suppress potential warnings
import logging # Import the logging module
# import traceback # Optional: Uncomment traceback.print_exc() for detailed debugging
from delta import calculate_delta_e
from pdf2image import convert_from_path

# --- Configure Logging ---
# Create a logger
logger = logging.getLogger('ClipartSegmenter')
# Set the logging level (DEBUG is lowest, shows everything)
# Change to INFO or WARNING later if output is too verbose
logger.setLevel(logging.DEBUG)

# Create a console handler and set level to DEBUG
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# Create a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add formatter to ch
ch.setFormatter(formatter)

# Add ch to logger
# Prevent adding multiple handlers if the script is run multiple times (e.g., in an interactive session)
if not logger.handlers:
    logger.addHandler(ch)

# Suppress warning for n_init being deprecated in future versions (common in sklearn updates)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
# Suppress potential warnings from libraries if they become noisy (e.g., KMeans convergence)
warnings.filterwarnings("ignore", category=UserWarning)


# Constants
PAGE_WIDTH = 595
PAGE_HEIGHT = 842
FONTS = ["Helvetica"]
CLIPART_FOLDER = "clipart"
# Renamed output directory to reflect KMeans segmentation
OUTPUT_DIR = "output_kmeans_seg"

def get_clipart_files():
    """Gets list of PNG files in the clipart folder."""
    if not os.path.exists(CLIPART_FOLDER):
        logger.error(f"Error: Clipart folder '{CLIPART_FOLDER}' not found.")
        return []
    return [os.path.join(CLIPART_FOLDER, f) for f in os.listdir(CLIPART_FOLDER) if f.lower().endswith('.png')]

def create_clipart_variants(clipart_files, base_img_size):
    """Creates initial random placement and scaling for a few cliparts."""
    canvas_width, canvas_height = base_img_size
    variants = []
    num_cliparts_to_use = min(5, len(clipart_files))
    if num_cliparts_to_use == 0:
        logger.warning("No clipart files found. Cannot create variants.")
        return [] # Return empty list

    selected_cliparts = random.sample(clipart_files, num_cliparts_to_use)

    for clip_path in selected_cliparts:
        try:
            # Temporarily open just to get dimensions
            with Image.open(clip_path) as temp_clip:
                original_width, original_height = temp_clip.size

            # Scale must result in a size smaller than the page
            # Also ensure resulting size is not tiny
            # Add margin to ensure cliparts don't precisely fill the page edge-to-edge
            margin = 40
            max_possible_scale_w = (canvas_width - margin) / original_width if original_width > 0 else 1.0
            max_possible_scale_h = (canvas_height - margin) / original_height if original_height > 0 else 1.0
            max_scale = min(max_possible_scale_w, max_possible_scale_h, 0.5) # Cap at 0.5 scale from original logic
            min_scale = 0.2 # Original min scale
            scale = random.uniform(min_scale, max_scale) if max_scale > min_scale else min_scale # Handle case where max < min

            new_width = max(20, int(original_width * scale)) # Ensure min size 20
            new_height = max(20, int(original_height * scale)) # Ensure min size 20

            max_x = canvas_width - new_width
            max_y = canvas_height - new_height

            # Ensure placement is within bounds, respecting the element size
            x = random.randint(0, max(0, max_x)) # max(0, ...) prevents negative bounds if size > page
            y = random.randint(0, max(0, max_y))

            variants.append({
                "path": clip_path,
                "x": x,
                "y": y,
                "width": new_width,
                "height": new_height,
                "angle": 0  # initially no rotation
            })
        except Exception as e:
            logger.error(f"Error processing clipart {clip_path} for variant creation: {e}")
    return variants


# Modified function to use KMeans with robust segment selection and logging
def hsv_shift_clipart_segmented(
    img: Image.Image,
    clip_path: str, # Added clip_path for logging context
    hue_range=(-30, 30),            # in degrees
    sat_range=(-0.2, 0.2),          # delta: -1.0 to 1.0
    val_range=(-0.2, 0.2),          # delta: -1.0 to 1.0
    k_clusters=(2, 5)               # Range for number of KMeans clusters
) -> Image.Image:
    """
    Applies a random HSV shift to a single randomly selected segment
    of an RGBA image using KMeans on HSV values, modifying only non-transparent
    pixels within that segment. Includes internal error handling and falls back
    to shifting all non-transparent pixels if segmentation fails or no valid
    segment is chosen.

    Parameters:
        img (PIL.Image): RGBA image.
        clip_path (str): Path of the original clipart file (for logging context).
        hue_range (tuple): Range in degrees to shift hue.
        sat_range (tuple): Range to shift saturation.
        val_range (tuple): Range to shift value (brightness).
        k_clusters (tuple): Range for the number of clusters in K-Means.

    Returns:
        PIL.Image: HSV-shifted image with alpha preserved, or the original image on failure
                   of shifting or segmentation.
    """
    # Keep a copy of the original image to return on failure
    original_img = img.copy()
    clip_name = os.path.basename(clip_path)
    logger.debug(f"[{clip_name}] Starting segmented color shift (size: {img.size})")

    try:
        img = img.convert("RGBA")
        arr = np.array(img).astype(np.float32) # Use float32 for calculations
        height, width, channels = arr.shape

        # Separate alpha channel and create mask for non-transparent pixels
        alpha = arr[..., 3]
        mask = alpha > 0
        non_transparent_pixels_indices = np.where(mask) # Get indices (rows, cols)

        # If no non-transparent pixels, return original image immediately
        if not np.any(mask): # Check if the mask contains any True values
            logger.debug(f"[{clip_name}] No non-transparent pixels found. Returning original.")
            return original_img

        num_non_transparent_pixels = np.sum(mask)
        logger.debug(f"[{clip_name}] Found {num_non_transparent_pixels} non-transparent pixels.")


        # --- KMeans Segmentation ---
        # Extract non-transparent RGB pixels and normalize for HSV conversion
        non_transparent_rgb = arr[mask, :3] / 255.0 # Normalized RGB values

        # Convert non-transparent RGB pixels to HSV for KMeans
        v_rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)
        hsv_tuples = v_rgb_to_hsv(non_transparent_rgb[:, 0], non_transparent_rgb[:, 1], non_transparent_rgb[:, 2])
        non_transparent_hsv = np.stack(hsv_tuples, axis=-1) # Shape (N, 3)


        min_k, max_k = k_clusters
        # Choose a random number of clusters within the range
        k_requested = random.randint(min_k, max_k)

        # Ensure k is reasonable for the number of non-transparent pixels and unique colors
        num_unique_colors = len(np.unique(non_transparent_rgb, axis=0))
        effective_k = max(1, min(k_requested, num_unique_colors, num_non_transparent_pixels)) # Cannot have more clusters than unique colors or samples

        logger.debug(f"[{clip_name}] Requested K: {k_requested}, Effective K: {effective_k}")


        target_mask = None # Initialize target mask

        # If effective k is 1, clustering is trivial. Treat as shifting all non-transparent pixels.
        # Also if there are fewer than 2 non-transparent pixels, cannot segment meaningfully.
        if effective_k < 2 or num_non_transparent_pixels < 2:
             logger.debug(f"[{clip_name}] Segmentation not meaningful (effective K < 2 or sparse data). Applying shift to entire non-transparent area.")
             target_mask = mask # Fallback if data is too sparse/simple
        else:
             try:
                 logger.debug(f"[{clip_name}] Calling KMeans(n_clusters={effective_k})...")
                 # Run KMeans on the non-transparent HSV data
                 # Using n_init=10 for robustness; 'auto' requires recent sklearn
                 kmeans = KMeans(n_clusters=effective_k, n_init=10, random_state=None)
                 labels_flat = kmeans.fit_predict(non_transparent_hsv) # Labels for the flat array
                 logger.debug(f"[{clip_name}] KMeans succeeded. Labels shape: {labels_flat.shape}, dtype: {labels_flat.dtype}")

                 # Get unique labels from the KMeans result
                 unique_labels_flat = np.unique(labels_flat)
                 logger.debug(f"[{clip_name}] Found {unique_labels_flat.size} unique labels from KMeans: {unique_labels_flat}")

                 # --- FIX: Convert NumPy array to list before random.choice ---
                 logger.debug(f"[{clip_name}] Choosing random segment from {unique_labels_flat.size} labels.")
                 target_label_flat = random.choice(unique_labels_flat.tolist()) # <-- .tolist() is the fix!
                 logger.debug(f"[{clip_name}] Chosen segment label for shift (from flat labels): {target_label_flat}")


                 # Map the chosen label back to the original image dimensions using the mask
                 # Create a label image initialized to -1 (for transparent/unassigned)
                 label_img = -np.ones((height, width), dtype=int)
                 # Place the KMeans labels for non-transparent pixels at their original coordinates
                 label_img[non_transparent_pixels_indices] = labels_flat

                 # Create a mask for the chosen segment in the original image dimensions
                 # This mask implicitly excludes transparent pixels because label_img is -1 there
                 potential_target_mask = (label_img == target_label_flat)
                 # logger.debug(f"[{clip_name}] Potential target mask created from label image. Shape: {potential_target_mask.shape}, dtype: {potential_target_mask.dtype}") # Too verbose


                 # Verify the chosen segment mask is not empty before using it
                 if not np.any(potential_target_mask):
                      # This means the randomly chosen label didn't actually map back to any pixels in the masked area. Should be rare.
                      logger.warning(f"[{clip_name}] Chosen segment label {target_label_flat} resulted in an empty mask ({np.sum(potential_target_mask)} pixels). Applying shift to entire non-transparent area.")
                      target_mask = mask # Fallback if chosen segment is empty after masking
                 else:
                      target_mask = potential_target_mask # Use the generated segment mask
                      logger.debug(f"[{clip_name}] Successfully created target mask for segment label {target_label_flat} ({np.sum(target_mask)} pixels).")


             except Exception as kmeans_e:
                 # Catch specific KMeans errors (like the ambiguous truth value error or others) and fallback gracefully
                 logger.error(f"[{clip_name}] Error during KMeans clustering or label processing: {kmeans_e}. Applying shift to entire non-transparent area.", exc_info=True) # Log traceback here
                 target_mask = mask # Fallback on KMeans failure


        # Final check: If target_mask wasn't set by any path above (shouldn't happen with current logic, but for safety)
        if target_mask is None:
            logger.error(f"[{clip_name}] Target mask generation logic failed unexpectedly. Final fallback to shifting all non-transparent pixels.")
            target_mask = mask

        # --- Apply HSV Shift to the target segment (defined by target_mask) ---
        # If the target segment is empty after masking (redundant check due to earlier np.any(), but safe), return original image (no shift applied)
        if not np.any(target_mask):
            logger.debug(f"[{clip_name}] Target segment became empty before shifting. No shift applied.")
            return original_img # Return original image

        logger.debug(f"[{clip_name}] Applying HSV shift to target mask with {np.sum(target_mask)} pixels.")
        # Get the RGB values for the target segment pixels based on the target_mask
        target_rgb = arr[target_mask, :3].astype(np.float32) / 255.0 # Normalized for conversion
        # logger.debug(f"[{clip_name}] Extracted {target_rgb.shape[0]} RGB pixels for shifting.") # Too verbose


        # Convert target RGB to HSV
        v_rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)
        target_hsv_tuples = v_rgb_to_hsv(target_rgb[:, 0], target_rgb[:, 1], target_rgb[:, 2])
        target_hsv = np.stack(target_hsv_tuples, axis=-1)
        # logger.debug(f"[{clip_name}] Converted {target_hsv.shape[0]} pixels to HSV.") # Too verbose


        # Generate random shifts (applied uniformly to the chosen segment)
        hue_shift = random.uniform(*hue_range) / 360.0 # Convert degrees to 0-1 range
        sat_shift = random.uniform(*sat_range)
        val_shift = random.uniform(*val_range)
        logger.debug(f"[{clip_name}] Shifts (H, S, V): ({hue_shift*360:.2f} deg, {sat_shift:.2f}, {val_shift:.2f})")


        # logger.debug(f"[{clip_name}] Applying shifts and clamping...") # Too verbose
        # Apply shifts with clamping
        target_hsv[:, 0] = (target_hsv[:, 0] + hue_shift) % 1.0 # Hue wraps around (0 to 1)
        target_hsv[:, 1] = np.clip(target_hsv[:, 1] + sat_shift, 0, 1) # Saturation clamps (0 to 1)
        target_hsv[:, 2] = np.clip(target_hsv[:, 2] + val_shift, 0, 1) # Value clamps (0 to 1)
        # logger.debug(f"[{clip_name}] Shifts applied.") # Too verbose


        # logger.debug(f"[{clip_name}] Converting shifted HSV back to RGB...") # Too verbose
        # Convert shifted HSV back to RGB
        v_hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)
        shifted_rgb_tuples = v_hsv_to_rgb(target_hsv[:, 0], target_hsv[:, 1], target_hsv[:, 2])
        shifted_rgb = np.stack(shifted_rgb_tuples, axis=-1) * 255.0 # Scale back to 0-255
        # logger.debug(f"[{clip_name}] Converted back to RGB.") # Too verbose

        # logger.debug(f"[{clip_name}] Updating original array with shifted pixels...") # Too verbose
        # Update the original array with the shifted RGB values for the target segment
        # Assign shifted values back to the pixels where target_mask is True
        arr[target_mask, :3] = shifted_rgb
        # logger.debug(f"[{clip_name}] Array updated.") # Too verbose

        # Recombine channels (including original alpha) and create the final image
        result_img = Image.fromarray(np.uint8(arr), 'RGBA')
        logger.debug(f"[{clip_name}] Shift applied successfully. Returning result image.")

        return result_img

    except Exception as e:
        # This outer catch block catches any *other* unexpected errors during the entire
        # hsv_shift_clipart_segmented process that weren't caught by the more specific blocks.
        # It's a final safety net.
        logger.error(f"[{clip_name}] An unhandled unexpected error occurred during segmented color shift process: {e}", exc_info=True) # Log traceback here too
        return original_img # Always return the original image on any failure


def generate_text_items(num=25):
    """Generates random text strings with positions, sizes, and angles."""
    text_samples = [
        "Capecitabina", "120 comprimidos", "Via oral", "Rx", "Uso hospitalar",
        "Advertencia", "Consulte Médico", "Manter fora do alcance", "Leia a bula", "Uso adulto",
        "Lote:", "Fabricação:", "Validade:", "Registro MS:", "Farmacêutico:",
        "Resp. Técnico:", "Indústria Brasileira", "Conservar em temperatura ambiente",
        "Proteger da luz e umidade", "Agite antes de usar", "Não injetar", "Uso externo",
        "Suspender em caso de reação", "Produto Estéril", "Não reutilize"
    ]
    items = []
    for _ in range(num):
        text = random.choice(text_samples)
        font_size = random.randint(8, 24) # Slightly wider range
        angle = random.choice([0, 90, 180, 270, 30, -30, 45, -45]) # More angles
        # Generate Y coordinate in ReportLab's bottom-up system
        # Roughly distribute text vertically, avoiding extreme edges
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
    """Draws text on the ReportLab canvas with rotation."""
    c.saveState()
    c.setFont(font_name, font_size)
    c.setFillColorRGB(*[v / 255 for v in color])
    c.translate(x, y) # Translate to the desired bottom-left point of the text block before rotation
    c.rotate(angle)
    c.drawString(0, 0, text) # Draw text at the new origin
    c.restoreState()

def draw_all_text(c, text_items):
    """Draws all text items on the ReportLab canvas."""
    for item in text_items:
         # item['x'] and item['y'] are generated assuming ReportLab's bottom-up coordinates.
         draw_text(c, item['text'], item['x'], item['y'], item['angle'], item['font_size'])


def create_pdf(filename, text_items, clipart_variants, mutation):
    """Creates a single PDF with text and cliparts, applying mutation if specified."""
    mutated_variants_info = []
    # Decide which cliparts (indices) get which mutation type
    for i, var in enumerate(clipart_variants):
        mutation_info = {"original": var} # Store original data
        mutation_info.update(var) # Start with original values, which will be overwritten if mutated

        if mutation:
            # Decide if *any* mutation happens to this clipart instance
            # Increased chance of *some* mutation
            if random.random() < 0.8:
                 # Choose mutation type(s) - could apply multiple
                 available_mutations = ["ROTATE", "RESIZE", "SHIFT", "COLOR_SHIFT"]
                 # More likely to have 1 or 2 mutations
                 num_mutations = random.choices([1, 2], weights=[0.6, 0.4], k=1)[0]
                 num_mutations = min(num_mutations, len(available_mutations)) # Cannot pick more than available

                 chosen_mutations = random.sample(available_mutations, num_mutations)

                 # Apply mutations sequentially, updating the mutation_info dictionary
                 for mut_type in chosen_mutations:
                    if mut_type == "ROTATE":
                        angle = random.choice([0, 90, 180, 270, 30, -30, 45, -45]) # More angle options
                        if angle != mutation_info.get("angle", 0): # Only apply if different from current/default
                            mutation_info["angle"] = angle
                            # logger.debug(f"Mutation applied to variant {i}: ROTATE {angle} deg")
                    elif mut_type == "RESIZE":
                        scale_factor = random.uniform(0.6, 1.6) # Wider resize range
                        # Use current dimensions in mutation_info, not original var, for chaining mutations
                        current_w = mutation_info.get("width", var["width"])
                        current_h = mutation_info.get("height", var["height"])
                        new_width = max(20, int(current_w * scale_factor)) # Ensure min size 20
                        new_height = max(20, int(current_h * scale_factor)) # Ensure min size 20

                        if new_width != current_w or new_height != current_h:
                             mutation_info["width"] = new_width
                             mutation_info["height"] = new_height
                             # Adjust position based on new size to keep it reasonably on page
                             mutation_info["x"] = min(mutation_info.get("x", var["x"]), PAGE_WIDTH - mutation_info["width"])
                             mutation_info["y"] = min(mutation_info.get("y", var["y"]), PAGE_HEIGHT - mutation_info["height"])
                             mutation_info["x"] = max(0, mutation_info["x"])
                             mutation_info["y"] = max(0, mutation_info["y"])
                             # logger.debug(f"Mutation applied to variant {i}: RESIZE (scale={scale_factor:.2f}) -> ({mutation_info['width']}x{mutation_info['height']})")
                    elif mut_type == "SHIFT":
                        shift = 60 # Increased shift range
                        original_x, original_y = mutation_info.get("x", var["x"]), mutation_info.get("y", var["y"])
                        current_w = mutation_info.get("width", var["width"]) # Need current size for bounds check
                        current_h = mutation_info.get("height", var["height"])
                        new_x = min(max(0, original_x + random.randint(-shift, shift)), PAGE_WIDTH - current_w)
                        new_y = min(max(0, original_y + random.randint(-shift, shift)), PAGE_HEIGHT - current_h)
                        if new_x != original_x or new_y != original_y:
                             mutation_info["x"] = new_x
                             mutation_info["y"] = new_y
                             # logger.debug(f"Mutation applied to variant {i}: SHIFT ({new_x - original_x},{new_y - original_y})")
                    elif mut_type == "COLOR_SHIFT":
                        if not mutation_info.get("color_shift", False): # Only mark for shift if not already marked
                             mutation_info["color_shift"] = True
                             # logger.debug(f"Mutation applied to variant {i}: SEGMENTED_COLOR_SHIFT")

        mutated_variants_info.append(mutation_info) # Store the final state (original + mutations)

    # Create the base image for cliparts using PIL
    # Use RGB base image as cliparts are pasted with alpha mask
    base_img = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), (255, 255, 255))

    for i, mut_info in enumerate(mutated_variants_info):
        var = mut_info["original"] # Original variant data (path is here)

        # Get the current state after potential mutations
        # Use .get() with defaults falling back to original variant data if mutation didn't apply
        current_x = mut_info.get("x", var["x"])
        current_y = mut_info.get("y", var["y"])
        current_width = mut_info.get("width", var["width"])
        current_height = mut_info.get("height", var["height"])
        current_angle = mut_info.get("angle", var["angle"])
        apply_color_shift = mut_info.get("color_shift", False)

        # Ensure size is positive after potential mutation or initial setup
        current_width = max(1, current_width)
        current_height = max(1, current_height)

        # Ensure position keeps the element on the page after potential mutation and size change
        # These clamps prevent the paste coordinates from being negative or pushing the image off the page
        # They represent the desired top-left corner of the element's bounding box on the base_img
        paste_x = min(max(0, current_x), PAGE_WIDTH - current_width)
        paste_y = min(max(0, current_y), PAGE_HEIGHT - current_height)


        try:
            # Open the original clipart image for this variant instance
            clip = Image.open(var["path"]).convert("RGBA")

            # Apply resize
            clip = clip.resize((current_width, current_height), Resampling.LANCZOS)

            # Apply rotation
            if current_angle != 0:
                # Expand=True to keep the whole rotated image, use BICUBIC for quality
                clip = clip.rotate(current_angle, expand=True, resample=Image.BICUBIC)
                # If expanded, the image size changes. Need to re-calculate paste position
                # to keep the top-left of the *new, expanded* image at the intended (paste_x, paste_y)
                # (paste_x, paste_y) should represent the desired top-left corner on the base_img.
                # Ensure the potentially larger rotated image's top-left is still within bounds.
                # Use the *original* (current_x, current_y) as the anchor for the *new* clip top-left
                paste_x = min(max(0, current_x), PAGE_WIDTH - clip.width)
                paste_y = min(max(0, current_y), PAGE_HEIGHT - clip.height)


            # Apply segmented color shift if marked
            if apply_color_shift:
                # hsv_shift_clipart_segmented now uses KMeans and handles its own errors, returning original on failure
                # Pass the clip_path for better logging context inside the function
                clip = hsv_shift_clipart_segmented(clip, var["path"],
                                                   hue_range=(-50, 50), # Slightly wider ranges for more noticeable effect
                                                   sat_range=(-0.5, 0.5),
                                                   val_range=(-0.5, 0.5),
                                                   k_clusters=(2, 8)) # Range for KMeans clusters


            # Paste the processed clipart onto the base image
            # PIL paste uses top-left coordinates (paste_x, paste_y)
            # The 'clip' image should be RGBA, its alpha channel is used as the mask
            base_img.paste(clip, (paste_x, paste_y), clip)

        except Exception as e:
            # This outer catch block handles errors NOT related to the color shift internal logic (which are handled internally),
            # such as failure to open the file, initial resize/rotate before color shift, or the paste operation itself.
            logger.error(f"Outer catch: Error processing and pasting clipart (path: {var['path']}, variant {i}): {e}", exc_info=True)
            # If an error happens here, the image for this variant WILL be missing from the base_img.
            # This is less likely now that color shift/segmentation errors are handled internally.

    # Convert PIL image to ReportLab ImageReader
    img_buffer = BytesIO()
    # Ensure base_img is RGB before saving to PNG buffer
    if base_img.mode != 'RGB':
         base_img = base_img.convert('RGB')

    base_img.save(img_buffer, format="PNG")
    img_buffer.seek(0)
    img_reader = ImageReader(img_buffer)

    c = canvas.Canvas(filename, pagesize=(PAGE_WIDTH, PAGE_HEIGHT))

    # Draw the PIL image onto the ReportLab canvas.
    # ReportLab's (0,0) is bottom-left. PIL's (0,0) is top-left.
    # To draw the PIL image upright filling the page, we draw its top-left (PIL 0,0)
    # at the PDF's top-left (PDF 0, PAGE_HEIGHT). The drawImage command takes the
    # *bottom-left* PDF coordinates for the image rectangle.
    # So, the bottom-left PDF coordinate should be (0, PAGE_HEIGHT - image_height).
    # This ensures the image fills the page correctly.
    c.drawImage(img_reader, 0, PAGE_HEIGHT - base_img.size[1], width=PAGE_WIDTH, height=PAGE_HEIGHT)


    # Draw text on top of the image
    draw_all_text(c, text_items)

    c.showPage()
    c.save()

# Main Execution
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Serial number handling
    existing = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".pdf")]
    # Find the highest serial number used
    serial_numbers = set()
    for f in existing:
         parts = f.replace('.pdf', '').split('_')
         if len(parts) >= 3 and parts[0] == 'output' and parts[1].isdigit():
             serial_numbers.add(int(parts[1]))

    serial = max(serial_numbers, default=0)
    serial_str = f"{serial:03d}"

    # PDF paths
    # Updated filenames to reflect KMeans segmentation
    orig_pdf = os.path.join(OUTPUT_DIR, f"output_{serial_str}_original_kmeans.pdf")
    mut_pdf = os.path.join(OUTPUT_DIR, f"output_{serial_str}_mutated_kmeans.pdf")

    original_img = convert_from_path(orig_pdf, dpi=300)[0]
    mutated_img = convert_from_path(mut_pdf, dpi=300)[0]

    calculate_delta_e(original_img, mutated_img, output_path=f"{OUTPUT_DIR}/delta_e_{serial_str}.png")
    # Load cliparts and generate text
    clipart_files = get_clipart_files()
    if not clipart_files:
        logger.critical("No clipart files found in the 'clipart' folder. Please add some PNG images. Exiting.")
        exit()

    base_image_size = (PAGE_WIDTH, PAGE_HEIGHT)
    # Generate text items ONCE to be the same for original and mutated
    text_items = generate_text_items()
    # Generate original clipart placements ONCE to be the base for both PDFs
    original_variants = create_clipart_variants(clipart_files, base_image_size)

    if not original_variants:
        logger.critical("Failed to create initial clipart variants (maybe no cliparts loaded or processing failed?). Exiting.")
        exit()

    # Generate PDFs
    logger.info(f"Generating original PDF: {orig_pdf}")
    # Pass the original variants directly, mutation=False means no mutations applied in create_pdf
    create_pdf(orig_pdf, text_items, original_variants, mutation=False)

    logger.info(f"Generating mutated PDF: {mut_pdf}")
    # Pass the original variants, mutation=True means create_pdf will apply mutations based on these originals
    create_pdf(mut_pdf, text_items, original_variants, mutation=True)

    logger.info(f"PDFs saved:\n{orig_pdf}\n{mut_pdf}")

# --- END OF FILE colorshift_seg_kmeans.py ---