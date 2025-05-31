
# import os
# import cv2
# import random
# import numpy as np
# from pdf2image import convert_from_path
# from augraphy import (
#     BrightnessTexturize, ColorPaper, ColorShift, DirtyDrum, Dithering,
#     DotMatrix, Gamma, InkBleed, InkMottling, Jpeg, LightingGradient,
#     LowInkRandomLines, LowLightNoise, NoiseTexturize, ReflectedLight,
#     ShadowCast, SubtleNoise
# )

# # Set the path to the Poppler bin directory
# poppler_path = r'C:/poppler/Library/bin'  # Update this path accordingly

# # Define the input and output directories
# input_folder = 'output_kmeans_seg_05'
# output_folder = 'Mutated_Augugraphy'
# os.makedirs(output_folder, exist_ok=True)

# def custom_color_tint(img, hue_shift=15, saturation_scale=1.2):
#     # Create a pure white image same size as input
#     white_image = np.ones_like(img) * 255

#     # Apply ColorPaper to the white image
#     color_paper = ColorPaper(hue_range=(20, 35), saturation_range=(90, 100), p=1)
#     tinted_paper = color_paper(white_image)

#     # Resize just to be sure (though shape should already match)
#     tinted_paper_resized = cv2.resize(tinted_paper, (img.shape[1], img.shape[0]))

#     # Blend the tinted paper with original image
#     alpha = 0.6  # Original image weight
#     beta = 0.4   # Tint weight
#     gamma = 0
#     blended = cv2.addWeighted(img, alpha, tinted_paper_resized, beta, gamma)
    
#     return blended
#     # cv2.imwrite(os.path.join(output_folder, "05_color_paper.jpg"), blended)

# # Define the list of augmentation functions
# augmentation_functions = [
#     ("BrightnessTexturize", lambda img: BrightnessTexturize(texturize_range=(0.9, 0.99), deviation=0.1)(img)),
#     # ("ColorPaper", lambda img: ColorPaper(hue_range=(20, 35), saturation_range=(90, 100), p=1)(img)),
#     ("ColorPaper", lambda img: custom_color_tint(img, hue_shift=random.randint(10, 30), saturation_scale=1.1)),
#     ("ColorShift", lambda img: ColorShift(
#         color_shift_offset_x_range=(1, 3),
#         color_shift_offset_y_range=(1, 3),
#         color_shift_iterations=(2, 3),
#         color_shift_brightness_range=(0.9, 1.1),
#         color_shift_gaussian_kernel_range=(3, 3)
#     )(img)),
#     ("DirtyDrum", lambda img: DirtyDrum(
#         line_width_range=(3, 5),
#         line_concentration=0.2,
#         direction=1,
#         noise_intensity=0.1,
#         noise_value=(0, 3),
#         ksize=(3, 3),
#         sigmaX=0
#     )(img)),
#     ("Dithering", lambda img: Dithering(order=(2, 5))(img)),
#     ("DotMatrix", lambda img: DotMatrix(
#         dot_matrix_shape="circle",
#         dot_matrix_dot_width_range=(5, 5),
#         dot_matrix_dot_height_range=(5, 5),
#         dot_matrix_min_width_range=(1, 1),
#         dot_matrix_max_width_range=(50, 50),
#         dot_matrix_min_height_range=(1, 1),
#         dot_matrix_max_height_range=(50, 50),
#         dot_matrix_min_area_range=(10, 10),
#         dot_matrix_max_area_range=(800, 800),
#         dot_matrix_median_kernel_value_range=(29, 29),
#         dot_matrix_gaussian_kernel_value_range=(1, 1),
#         dot_matrix_rotate_value_range=(0, 0)
#     )(img)),
#     ("Gamma", lambda img: Gamma(gamma_range=(2.0, 3.0))(img)),
#     ("InkBleed", lambda img: InkBleed(intensity_range=(0.4, 0.7), kernel_size=(5, 5), severity=(0.2, 0.4))(img)),
#     ("InkMottling", lambda img: InkMottling(
#         ink_mottling_alpha_range=(0.5, 0.5),
#         ink_mottling_noise_scale_range=(1, 1),
#         ink_mottling_gaussian_kernel_range=(3, 5)
#     )(img)),
#     ("Jpeg", lambda img: Jpeg(quality_range=(5, 10))(img)),
#     ("LightingGradient", lambda img: LightingGradient(
#         light_position=None,
#         direction=90,
#         max_brightness=255,
#         min_brightness=0,
#         mode="gaussian",
#         transparency=0.5
#     )(img)),
#     ("LowInkRandomLines", lambda img: LowInkRandomLines(count_range=(30, 50), use_consistent_lines=True, noise_probability=0.1)(img)),
#     ("LowLightNoise", lambda img: LowLightNoise(
#         num_photons_range=(50, 100),
#         alpha_range=(0.7, 0.10),
#         beta_range=(10, 30),
#         gamma_range=(1.0, 1.8)
#     )(img)),
#     ("NoiseTexturize", lambda img: NoiseTexturize(
#         sigma_range=(2, 3),
#         turbulence_range=(2, 5),
#         texture_width_range=(50, 500),
#         texture_height_range=(50, 500)
#     )(img)),
#     ("ReflectedLight", lambda img: ReflectedLight(
#         reflected_light_smoothness=0.8,
#         reflected_light_internal_radius_range=(0.1, 0.7),
#         reflected_light_external_radius_range=(0.8, 0.98),
#         reflected_light_minor_major_ratio_range=(0.9, 1.0),
#         reflected_light_color=(255, 255, 255),
#         reflected_light_internal_max_brightness_range=(0.4, 0.5),
#         reflected_light_external_max_brightness_range=(0.3, 0.4),
#         reflected_light_location="random",
#         reflected_light_ellipse_angle_range=(0, 360),
#         reflected_light_gaussian_kernel_size_range=(5, 310)
#     )(img)),
#     ("ShadowCast", lambda img: ShadowCast(
#         shadow_side="bottom",
#         shadow_vertices_range=(2, 3),
#         shadow_width_range=(0.5, 0.8),
#         shadow_height_range=(0.5, 0.8),
#         shadow_color=(0, 0, 0),
#         shadow_opacity_range=(0.5, 0.6),
#         shadow_iterations_range=(1, 2),
#         shadow_blur_kernel_range=(101, 301)
#     )(img)),
#     ("SubtleNoise", lambda img: SubtleNoise(subtle_range=25)(img))
# ]

# # Get all original and mutated PDF files
# original_pdfs = sorted([f for f in os.listdir(input_folder) if f.endswith('_original_kmeans.pdf')])
# mutated_pdfs = sorted([f for f in os.listdir(input_folder) if f.endswith('_mutated_kmeans.pdf')])

# # Process each pair of original and mutated PDFs
# for idx, (orig_pdf, mut_pdf) in enumerate(zip(original_pdfs, mutated_pdfs), start=1):
#     # Convert original PDF to image
#     orig_pages = convert_from_path(os.path.join(input_folder, orig_pdf), poppler_path=poppler_path)
#     orig_image = orig_pages[0]
#     orig_image_path = os.path.join(output_folder, f"{idx:03d}_original.jpg")
#     orig_image.save(orig_image_path, 'JPEG')

#     # Convert mutated PDF to image
#     mut_pages = convert_from_path(os.path.join(input_folder, mut_pdf), poppler_path=poppler_path)
#     mut_image = mut_pages[0]
#     mut_image_path = os.path.join(output_folder, f"{idx:03d}_mutated.jpg")
#     mut_image.save(mut_image_path, 'JPEG')

#     # Convert mutated image to OpenCV format
#     mut_image_cv = cv2.cvtColor(np.array(mut_image), cv2.COLOR_RGB2BGR)

#     # Randomly select 5 augmentation functions
#     selected_augmentations = random.sample(augmentation_functions, 5)
#     augmented_image = mut_image_cv.copy()
#     method_names = []

#     # Apply selected augmentations
#     for name, func in selected_augmentations:
#         augmented_image = func(augmented_image)
#         method_names.append(name)

#     # Save augmented image
#     methods_str = '_'.join(method_names)
#     aug_image_path = os.path.join(output_folder, f"{idx:03d}_augmented_{methods_str}.jpg")
#     cv2.imwrite(aug_image_path, augmented_image)

#     print(f"Processed set {idx:03d}:")
#     print(f"  Original Image: {orig_image_path}")
#     print(f"  Mutated Image: {mut_image_path}")
#     print(f"  Augmented Image: {aug_image_path}")






####################################################################################################################################





import os
import cv2
import random
import numpy as np
from pdf2image import convert_from_path
from augraphy import (
    BrightnessTexturize, ColorPaper, ColorShift, DirtyDrum, Dithering
    , Gamma, InkBleed, InkMottling, Jpeg, LightingGradient,
    LowInkRandomLines, LowLightNoise, NoiseTexturize, ReflectedLight,
    ShadowCast, SubtleNoise
)

# Poppler path
poppler_path = r'C:/poppler/Library/bin'

# I/O
input_folder = 'output_kmeans_seg_05'
output_folder = 'Mutated_Augugraphy_05'
os.makedirs(output_folder, exist_ok=True)

# Custom color tint
def custom_color_tint(img):
    white = np.ones_like(img) * 255
    tinted = ColorPaper((20, 35), (90, 100), p=1)(white)
    resized = cv2.resize(tinted, (img.shape[1], img.shape[0]))
    return cv2.addWeighted(img, 0.8, resized, 0.2, 0)

# PHASES with augmentations
phases = {
    "Pre": [
        ("Jpeg", lambda img: Jpeg((5, 10))(img)),
        ("Gamma", lambda img: Gamma((1.5, 2.0))(img))
    ],
    "Ink": [
        ("InkBleed", lambda img: InkBleed((0.2, 0.4), (5, 5), (0.1, 0.3))(img)),
        ("LowInkRandomLines", lambda img: LowInkRandomLines((20, 40), True, 0.1)(img)),
        ("Dithering", lambda img: Dithering(order=(2, 5))(img)),
        ("InkMottling", lambda img: InkMottling((0.5, 0.5), (1, 1), (3, 5))(img))
    ],
    "Paper": [
        ("ColorPaper", lambda img: custom_color_tint(img)),
        ("BrightnessTexturize", lambda img: BrightnessTexturize((0.95, 0.99), deviation=0.05)(img)),
        ("NoiseTexturize", lambda img: NoiseTexturize((2, 3), (2, 4), (100, 300), (100, 300))(img))
    ],
    "Post": [
        ("ShadowCast", lambda img: ShadowCast(
            shadow_side="bottom",
            shadow_vertices_range=(2, 3),
            shadow_width_range=(0.3, 0.4),
            shadow_height_range=(0.3, 0.4),
            shadow_opacity_range=(0.1, 0.2),
            shadow_blur_kernel_range=(51, 101)
        )(img)),
        ("LightingGradient", lambda img: LightingGradient(
            max_brightness=120,
            min_brightness=100,
            transparency=0.3,
            mode="linear"
        )(img)),
        ("ReflectedLight", lambda img: ReflectedLight(
            reflected_light_internal_max_brightness_range=(0.2, 0.3),
            reflected_light_external_max_brightness_range=(0.1, 0.2),
            reflected_light_color=(255, 255, 255)
        )(img)),
        ("ColorShift", lambda img: ColorShift(
            color_shift_offset_x_range=(1, 2),
            color_shift_offset_y_range=(1, 2),
            color_shift_iterations=(1, 2),
            color_shift_brightness_range=(0.95, 1.05)
        )(img)),
        ("LowLightNoise", lambda img: LowLightNoise(
            num_photons_range=(100, 150),
            alpha_range=(0.3, 0.5),
            beta_range=(2, 5),
            gamma_range=(1.0, 1.1)
        )(img)),
        ("SubtleNoise", lambda img: SubtleNoise(subtle_range=20)(img)),
        ("DirtyDrum", lambda img: DirtyDrum(
            line_width_range=(2, 4),
            line_concentration=0.2,
            direction=1,
            noise_intensity=0.1,
            noise_value=(0, 3),
            ksize=(3, 3),
            sigmaX=0
        )(img))
    ]
}

# Get PDF pairs
original_pdfs = sorted(f for f in os.listdir(input_folder) if f.endswith('_original_kmeans.pdf'))
mutated_pdfs = sorted(f for f in os.listdir(input_folder) if f.endswith('_mutated_kmeans.pdf'))

for idx, (orig_pdf, mut_pdf) in enumerate(zip(original_pdfs, mutated_pdfs), start=1):
    orig_img = convert_from_path(os.path.join(input_folder, orig_pdf), poppler_path=poppler_path)[0]
    mut_img = convert_from_path(os.path.join(input_folder, mut_pdf), poppler_path=poppler_path)[0]

    orig_img.save(os.path.join(output_folder, f"{idx:03d}_original.jpg"), "JPEG")
    mut_img.save(os.path.join(output_folder, f"{idx:03d}_mutated.jpg"), "JPEG")

    img = cv2.cvtColor(np.array(mut_img), cv2.COLOR_RGB2BGR)
    methods = []

    chosen_phases = random.sample(list(phases.keys()), random.randint(3, 4))
    for phase in chosen_phases:
        name, func = random.choice(phases[phase])
        img = func(img)
        methods.append(f"{phase}_{name}")

    filename = f"{idx:03d}_augmented_{'_'.join(methods)}.jpg"
    cv2.imwrite(os.path.join(output_folder, filename), img)

    print(f" Processed {idx:03d}: {', '.join(methods)}")
