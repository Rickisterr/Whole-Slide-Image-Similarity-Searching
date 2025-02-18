import os
import numpy as np
from PIL import Image



class tissueRegionization:
    def __init__(self, empty_threshold):
        self.empty_threshold = empty_threshold
        

    def getPixelValues(self, images_array, image_size):
        img_features = []
        
        for img_path in images_array:
            with Image.open(img_path) as img:
                img = img.resize(image_size).convert('L')               # Convert to grayscale
                pixels = np.array(img)                                  # Convert to NumPy array
                img_features.append(pixels)                             # Append to the list
        
        return img_features


    def calculateEmptyPercentage(self, images_array):
        percentages = {}

        # Get grayscale images using getPixelValues
        grayscale_images = self.getPixelValues(images_array, (64, 64))

        for img_path, grayscale_pixels in zip(images_array, grayscale_images):
            
            # Define empty regions (black, white, or near-gray based on threshold)
            empty_mask = (grayscale_pixels < self.empty_threshold) | (grayscale_pixels > (255 - self.empty_threshold))

            percentage_empty = np.sum(empty_mask) / (64 * 64) * 100     # Percentage of empty pixels
            percentages[img_path] = round(percentage_empty, 2)

        return percentages
