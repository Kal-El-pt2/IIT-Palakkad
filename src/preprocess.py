# src/preprocess.py
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

def preprocess_image(img_path, size=(224, 224)):
    img = cv2.imread(img_path)
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = normalize_brightness_contrast(img)  # Apply normalization
    
    if needs_inversion(img):
        img = cv2.flip(img, 0)
    return img


def needs_inversion(img):
    # Step 1: Check for Notch
    if contains_notch(img):
        return False  # Not inverted if a notch is present

    # Step 2: Identify the macula and optic nerve
    macula_position, optic_nerve_position = find_macula_and_optic_nerve(img)
    
    if macula_position and optic_nerve_position:
        # Compare their vertical positions
        if macula_position[1] < optic_nerve_position[1]:  # macula is higher than optic nerve
            return True  # Image is inverted
    return False  # Not inverted if macula is not higher


def is_notch_shape(contour):
    # Approximate the contour shape
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    
    # Check for specific shapes
    if len(approx) == 3:  # Triangle
        return True
    elif len(approx) == 4:  # Rectangle
        aspect_ratio = cv2.boundingRect(contour)[2] / cv2.boundingRect(contour)[3]
        if 0.9 <= aspect_ratio <= 1.1:  # Check if it's roughly square
            return True
    elif len(approx) > 4:  # Circle or approximate circle
        area = cv2.contourArea(contour)
        ((x, y), radius) = cv2.minEnclosingCircle(contour)
        circle_area = np.pi * (radius ** 2)
        if 0.8 * circle_area < area < 1.2 * circle_area:  # Circular shape
            return True
    return False


def contains_notch(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 10 and h > 10 and (x < 10 or x + w > img.shape[1] - 10):
            if is_notch_shape(cnt):
                return True
    return False


def find_macula_and_optic_nerve(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img_clahe = cv2.merge((l, a, b))
    img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_LAB2RGB)

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    _, bright_thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    _, dark_thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)

    optic_contours, _ = cv2.findContours(bright_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    macula_contours, _ = cv2.findContours(dark_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    optic_nerve_position = get_largest_contour_center(optic_contours)
    macula_position = get_largest_contour_center(macula_contours)
    
    return macula_position, optic_nerve_position




def get_largest_contour_center(contours):
    """Find the center of the largest contour."""
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return (cX, cY)
    return None



def normalize_brightness_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img_normalized = cv2.merge((l, a, b))
    return cv2.cvtColor(img_normalized, cv2.COLOR_LAB2RGB)



def preprocess_and_save_images(data_dir, output_dir, batch_size=32):
    """Preprocess images in batches and save them to an output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    image_paths = glob(os.path.join(data_dir, '*.jpeg'))
    print(f"Found {len(image_paths)} images in {data_dir}.")
    
    # Process in batches
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i+batch_size]
        print(f"Processing batch {i // batch_size + 1} of {len(image_paths) // batch_size + 1}")
        
        for img_path in batch:
            processed_img = preprocess_image(img_path)
            
            if processed_img is not None:
                output_path = os.path.join(output_dir, f"processed_{os.path.basename(img_path)}")
                cv2.imwrite(output_path, processed_img)
                print(f"Saved processed image to: {output_path}")
            else:
                print(f"Image {img_path} could not be processed.")

# Example usage
if __name__ == "__main__":
    data_dir = "data/sampleimages"  # Your input directory
    output_dir = "data/processed_images"  # Output directory for processed images
    preprocess_and_save_images(data_dir, output_dir, batch_size=32)
