import cv2
import time
import numpy as np

def preprocess_image(image_path, target_size=(640, 640)):
    """
    CPU-side preprocessing stage.
    Simulates work done before sending data to DSP.
    """

    # Start timing CPU preprocessing
    start_time = time.time()

    # Read image from disk (CPU I/O)
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Failed to load image")

    # Resize image (CPU compute)
    image = cv2.resize(image, target_size)

    # Normalize image (CPU compute)
    image = image.astype(np.float32) / 255.0

    # End timing
    end_time = time.time()

    preprocess_time_ms = (end_time - start_time) * 1000

    return image, preprocess_time_ms
