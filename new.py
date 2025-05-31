import cv2 
import os
import random
from augraphy import *  
from augraphy import ShadowCast 
from augraphy import LowInkRandomLines 
from augraphy.augmentations.colorshift import ColorShift 
from augraphy.augmentations.dithering import Dithering 
from augraphy.augmentations.colorpaper import ColorPaper 
# Load your input image
input_image_path = "D:/ImgAgumentation/newImg.jpg"  # <-- Change if needed
image = cv2.imread(input_image_path)

# Create output folder if it doesn't exist
output_folder = "new_Images_02"
os.makedirs(output_folder, exist_ok=True)

# # #1. Bad Photography (Light Augmentation)
# # bad_photo = BadPhotoCopy(noise_type=2,
# #                                    noise_side="right",
# #                                    noise_iteration=(1,1),
# #                                    noise_size=(1,1),
# #                                    noise_sparsity=(0.4,0.5),
# #                                    noise_concentration=(0.2,0.2),
# #                                    blur_noise=1,
# #                                    blur_noise_kernel=(5, 5),
# #                                    wave_pattern=0,
# #                                    edge_effect=1)
# # out = bad_photo(image)
# # cv2.imwrite(os.path.join(output_folder, "01_bad_photography.jpg"), out)

# # #2. Bleed Through
# # bleed = BleedThrough(intensity_range=(0.1, 0.2),
# #                             color_range=(0, 224),
# #                             ksize=(17, 17),
# #                             sigmaX=0,
# #                             alpha=0.3,
# #                             offsets=(5, 10))
# # out = bleed(image)
# # cv2.imwrite(os.path.join(output_folder, "02_bleedthrough.jpg"), out)

# # #3. Brightness Adjustment
# # brightness = Brightness(brightness_range=(0.2, 0.8),
# #                                 min_brightness=1,
# #                                 min_brightness_value=(120, 150))
# # out = brightness(image)
# # cv2.imwrite(os.path.join(output_folder, "03_brightness.jpg"), out)

# #4. Brightness Texturize
brightness_texturize = BrightnessTexturize(texturize_range=(0.9, 0.99), 
                                           deviation=0.1 )
out = brightness_texturize = brightness_texturize(image)
cv2.imwrite(os.path.join(output_folder, "04_brightness_texturize.jpg"), out)

#5. Color Paper
#  In OpenCV, Hue values range from 0 to 179
# Yellow corresponds to hue values roughly between 20 and 35
color_paper= ColorPaper(hue_range=(20, 35), saturation_range=(90,100),p=1)
out = color_paper(image)

blend_image = cv2.imread("D:/ImgAgumentation/newImg.jpg", cv2.IMREAD_COLOR)
texture_image_resized = cv2.resize(out, (blend_image.shape[1], blend_image.shape[0]))
alpha = 0.6
beta = 0.4
gamma = 0
# Blend the images
blended = cv2.addWeighted(blend_image, alpha, texture_image_resized, beta, gamma)
cv2.imwrite(os.path.join(output_folder, "05_color_paper.jpg"), blended)

# #6. Color Shift (Lighting Gradient)
color_shift = ColorShift(color_shift_offset_x_range = (1,3),
                        color_shift_offset_y_range = (1,3),
                        color_shift_iterations = (2,3),
                        color_shift_brightness_range = (0.9,1.1),
                        color_shift_gaussian_kernel_range = (3,3),)
out = color_shift(image)
cv2.imwrite(os.path.join(output_folder, "06_color_shift.jpg"), out)


# # #7. Tessellation Noise
# # tessellation = VoronoiTessellation(mult_range = (50,80),
# #                          seed = 19829813472 ,
# #                          num_cells_range = (500,800),
# #                          noise_type = "random",
# #                          background_value = (100, 200))
# # out = tessellation(image)
# # cv2.imwrite(os.path.join(output_folder, "07_tessellation_noise.jpg"), out)

# #8. Dirty Drum Effect
dirty_drum = DirtyDrum(line_width_range=(3, 5), 
                      line_concentration=0.2,
                      direction=1,
                      noise_intensity=0.1,
                      noise_value=(0, 3),
                      ksize=(3, 3),
                      sigmaX=0)
out = dirty_drum(image)
cv2.imwrite(os.path.join(output_folder, "08_dirty_drum.jpg"), out)

#9. Dithering
dithering = Dithering(order=(2, 5))
out = dithering(image)
cv2.imwrite(os.path.join(output_folder, "09_dithering.jpg"), out)

# #10. Dot Matrix (Bayer Dithering)
dot_matrix = DotMatrix(dot_matrix_shape="circle", 
                      dot_matrix_dot_width_range=(5, 5),
                      dot_matrix_dot_height_range=(5, 5),
                      dot_matrix_min_width_range=(1, 1),
                      dot_matrix_max_width_range=(50, 50),
                      dot_matrix_min_height_range=(1, 1),
                      dot_matrix_max_height_range=(50, 50),
                      dot_matrix_min_area_range=(10, 10),
                      dot_matrix_max_area_range=(800, 800),
                      dot_matrix_median_kernel_value_range = (29,29),
                      dot_matrix_gaussian_kernel_value_range=(1, 1),
                      dot_matrix_rotate_value_range=(0, 0))
out = dot_matrix(image)
cv2.imwrite(os.path.join(output_folder, "10_dot_matrix.jpg"), out)

# # #11. Fax Machine Output
# # fax = Faxify(scale_range = (1,2),
# #                 monochrome = 1,
# #                 monochrome_method = "cv2.threshold",
# #                 monochrome_arguments = {"thresh":128, "maxval":128, "type":cv2.THRESH_BINARY},
# #                 halftone = 1,
# #                 invert = 1,
# #                 half_kernel_size = (2,2),
# #                 angle = (0, 360),
# #                 sigma = (1,3))
# # out = fax(image)
# # cv2.imwrite(os.path.join(output_folder, "11_fax_machine.jpg"), out)

# #12. Gamma Augmentation
gamma_aug = Gamma(gamma_range=(2.0, 3.0)) 
out = gamma_aug(image)
cv2.imwrite(os.path.join(output_folder, "12_gamma_correction.jpg"), out)

# #13. Ink Bleed
ink_bleed = InkBleed(intensity_range=(0.4, 0.7), 
                    kernel_size=(5, 5),
                    severity=(0.2, 0.4))
out = ink_bleed(image)
cv2.imwrite(os.path.join(output_folder, "13_ink_bleed.jpg"), out)

# #14. Ink mottling
ink_mottling= InkMottling(ink_mottling_alpha_range=(0.5, 0.5), 
                         ink_mottling_noise_scale_range=(1,1),
                         ink_mottling_gaussian_kernel_range=(3,5),
                         )
out = ink_mottling(image)
cv2.imwrite(os.path.join(output_folder, "14_ink_mottling.jpg"), out)

# #15. JPEG Compression
jpeg_aug = Jpeg(quality_range=(5, 10)) 
out = jpeg_aug(image)
cv2.imwrite(os.path.join(output_folder, "15_jpeg_compression.jpg"), out)

# # #16. Letterpress
# # letterpress = Letterpress(n_samples=(200, 500),
# #                           n_clusters=(300, 800),
# #                           std_range=(1500, 5000),
# #                           value_range=(200, 255),
# #                           value_threshold_range=(120, 128),
# #                           blur=1)
# # out = letterpress(image)
# # cv2.imwrite(os.path.join(output_folder, "16_letterpress.jpg"), out)

# #17. Lighting Gradient
lighting_grad = LightingGradient(light_position=None, 
                                 direction=90,
                                 max_brightness=255,
                                 min_brightness=0,
                                 mode="gaussian",
                                 transparency=0.5)
out = lighting_grad(image)
cv2.imwrite(os.path.join(output_folder, "17_lighting_gradient.jpg"), out)

# #18. Low Ink Lines
lowink_random = LowInkRandomLines(count_range=(30, 50),
                                     use_consistent_lines=True,
                                     noise_probability=0.1)
out = lowink_random(image)
cv2.imwrite(os.path.join(output_folder, "18_lowink_lines.jpg"), out)

# #19. Low Light Noise
low_light = LowLightNoise(num_photons_range = (50, 100), 
    alpha_range = (0.7, 0.10),
    beta_range = (10, 30),
    gamma_range = (1.0 , 1.8))
out = low_light(image)
cv2.imwrite(os.path.join(output_folder, "19_low_light.jpg"), out)

# #20. Texture Noise
texture_noise = NoiseTexturize(sigma_range=(2, 3), 
                                 turbulence_range=(2, 5),
                                 texture_width_range=(50, 500),
                                 texture_height_range=(50, 500))
out = texture_noise(image)
cv2.imwrite(os.path.join(output_folder, "20_texture_noise.jpg"), out)

#21. Reflected Light
reflected_light = ReflectedLight(reflected_light_smoothness = 0.8, 
                                 reflected_light_internal_radius_range=(0.1, 0.7),
                                 reflected_light_external_radius_range=(0.8, 0.98),
                                 reflected_light_minor_major_ratio_range = (0.9, 1.0),
                                 reflected_light_color = (255,255,255),
                                 reflected_light_internal_max_brightness_range=(0.4,0.5),
                                 reflected_light_external_max_brightness_range=(0.3,0.4),
                                 reflected_light_location = "random",
                                 reflected_light_ellipse_angle_range = (0, 360),
                                 reflected_light_gaussian_kernel_size_range = (5,310))
out = reflected_light(image)
cv2.imwrite(os.path.join(output_folder, "21_reflected_light.jpg"), out)

# #22. Shadow
shadow_cast = ShadowCast(shadow_side = "bottom",
                        shadow_vertices_range = (2, 3),
                        shadow_width_range=(0.5, 0.8),
                        shadow_height_range=(0.5, 0.8),
                        shadow_color = (0, 0, 0),
                        shadow_opacity_range=(0.5,0.6),
                        shadow_iterations_range = (1,2),
                        shadow_blur_kernel_range = (101, 301))
out = shadow_cast(image)
cv2.imwrite(os.path.join(output_folder, "22_shadowCast.jpg"), out)

# #23. Subtle Noise
subtle_noise = SubtleNoise(subtle_range=25) 
out = subtle_noise(image)
cv2.imwrite(os.path.join(output_folder, "23_subtle_noise.jpg"), out)

print("\n✅ All augmentations done! Check the 'new_Images_03' folder.")

