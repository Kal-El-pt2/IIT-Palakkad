# src/preprocess.py
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

def preprocess_image(img_path, size=(224, 224)):
    img = cv2.imread(img_path)
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Example inversion check (simplified for macula detection)
    if needs_inversion(img):
        img = cv2.flip(img, 0)  # Vertical flip if inverted

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


def contains_notch(img):
    """Check if there is a noticeable notch on the side."""
    # Convert to grayscale and apply edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        # Check for large contours near image edges, which could indicate a notch
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 10 and h > 10 and (x < 10 or x + w > img.shape[1] - 10):
            return True  # Notch detected
    return False  # No notch detected


def find_macula_and_optic_nerve(img):
    """Identify approximate positions of macula and optic nerve."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # Threshold to identify bright (optic nerve) and dark (macula) areas
    _, bright_thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    _, dark_thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)

    # Get contours of optic nerve (bright areas) and macula (dark areas)
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



def preprocess_and_save_images(data_dir, output_dir, num_samples=5):
    """Preprocess images and save them to a specified output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create output directory if it doesn't exist
        print(f"Created output directory: {output_dir}")
    
    sample_images = glob(os.path.join(data_dir, '*.jpeg'))
    print(f"Found {len(sample_images)} images in {data_dir}.")
    
    for i, img_path in enumerate(sample_images[:num_samples]):
        print(f"Processing image: {img_path}")  # Debugging statement
        processed_img = preprocess_image(img_path)
        
        if processed_img is not None:  # Check if the image was processed successfully
            output_path = os.path.join(output_dir, f"processed_{os.path.basename(img_path)}")
            cv2.imwrite(output_path, processed_img)  # Save the processed image
            print(f"Saved processed image to: {output_path}")
        else:
            print(f"Image {img_path} could not be processed.")


# Call the function with your input and output directory
if __name__ == "__main__":
    data_dir = "data/sampleimages"  # Adjust path to your data directory
    output_dir = "data/processed_images"  # Adjust path for saving processed images
    preprocess_and_save_images(data_dir, output_dir)
