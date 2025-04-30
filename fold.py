from augraphy import *
import random
import cv2
import numpy as np
ink_phase = [
    # Geometric(translation=(0, 0.2))
            #  Squish(squish_direction=1,
            #         squish_location='random',
            #         squish_number_range=(5, 10),
            #         squish_distance_range=(5, 7),
            #         squish_line=0, squish_line_thickness_range=(1, 1))
                    ]

paper_phase = []

post_phase = [Folding(fold_x=None,
                      fold_deviation=(0, 0),
                      fold_count=random.randint(4,7),
                      fold_noise=0,
                      fold_angle_range=(0, 45),
                    #   gradient_width=(0.1, 0.2),
                    #   gradient_height=(0.01, 0.02),
                      gradient_width=(0.1, 0.2),
                      gradient_height=(0.01, 0.02),
                      backdrop_color=(0, 0, 0)
                      )]


pipeline = AugraphyPipeline(ink_phase=ink_phase, paper_phase=paper_phase, post_phase=post_phase)

image = cv2.imread("C:/Users/RD001/OneDrive - SPAN INSPECTION SYSTEMS PVT LTD/Desktop/raj/ImgAugmentation/nepokare.png")

image_augmented = pipeline(image)
cv2.imwrite("crumpled.png", image_augmented)