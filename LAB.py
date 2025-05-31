import os
import cv2
import numpy as np
from pdf2image import convert_from_path

# Poppler path for Windows
poppler_path = r"C:/poppler/Library/bin"  # Change this to your poppler path

# Folders
input_folder = "output_kmeans_seg_01"
output_folder = "LAB_Diff_Images"
os.makedirs(output_folder, exist_ok=True)

# Collect all relevant PDF pairs
original_pdfs = sorted([f for f in os.listdir(input_folder) if f.endswith("_original_kmeans.pdf")])
mutated_pdfs = sorted([f for f in os.listdir(input_folder) if f.endswith("_mutated_kmeans.pdf")])

# --- Difference Function ---
def generate_lab_difference_map(original_bgr, mutated_bgr):
    if original_bgr.shape != mutated_bgr.shape:
        mutated_bgr = cv2.resize(mutated_bgr, (original_bgr.shape[1], original_bgr.shape[0]))
    original_lab = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2LAB)
    mutated_lab = cv2.cvtColor(mutated_bgr, cv2.COLOR_BGR2LAB)
    lab_diff = cv2.absdiff(original_lab, mutated_lab)
    return lab_diff

# --- Main Processing ---
for idx, (orig_pdf, mut_pdf) in enumerate(zip(original_pdfs, mutated_pdfs), start=1):
    # Convert original PDF to JPEG
    orig_image = convert_from_path(os.path.join(input_folder, orig_pdf), poppler_path=poppler_path)[0]
    orig_bgr = cv2.cvtColor(np.array(orig_image), cv2.COLOR_RGB2BGR)
    orig_path = os.path.join(output_folder, f"{idx:03d}_original.jpg")
    cv2.imwrite(orig_path, orig_bgr)

    # Convert mutated PDF to JPEG
    mut_image = convert_from_path(os.path.join(input_folder, mut_pdf), poppler_path=poppler_path)[0]
    mut_bgr = cv2.cvtColor(np.array(mut_image), cv2.COLOR_RGB2BGR)
    mut_path = os.path.join(output_folder, f"{idx:03d}_mutated.jpg")
    cv2.imwrite(mut_path, mut_bgr)

    # Compute LAB difference image
    lab_diff = generate_lab_difference_map(orig_bgr, mut_bgr)
    lab_diff_bgr = cv2.cvtColor(lab_diff, cv2.COLOR_LAB2BGR)  # For visualization only
    diff_path = os.path.join(output_folder, f"{idx:03d}_diff_lab.jpg")
    cv2.imwrite(diff_path, lab_diff_bgr)

    print(f"Processed set {idx:03d}")
    print(f"  Original: {orig_path}")
    print(f"  Mutated: {mut_path}")
    print(f"  LAB Diff: {diff_path}")
