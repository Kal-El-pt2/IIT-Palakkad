# src/dataAugmentation.py
import cv2
import numpy as np
import os
from glob import glob

def augment_image(img):
    """Applies a series of augmentations to an image."""
    augmented_images = []

    # Rotation (e.g., -30, 0, 30 degrees)
    angles = [-30, 0, 30]
    for angle in angles:
        rotated = rotate_image(img, angle)
        augmented_images.append(rotated)

    # Flip (horizontal and vertical)
    flipped_h = cv2.flip(img, 1)
    flipped_v = cv2.flip(img, 0)
    augmented_images.extend([flipped_h, flipped_v])

    # Brightness and contrast adjustments
    brightened = adjust_brightness_contrast(img, brightness=30, contrast=50)
    darkened = adjust_brightness_contrast(img, brightness=-30, contrast=30)
    augmented_images.extend([brightened, darkened])

    return augmented_images

def rotate_image(img, angle):
    """Rotates the image by a specified angle."""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1)
    rotated_img = cv2.warpAffine(img, rotation_matrix, (w, h))
    return rotated_img

def adjust_brightness_contrast(img, brightness=0, contrast=0):
    """Adjusts brightness and contrast of an image."""
    return cv2.convertScaleAbs(img, alpha=1 + (contrast / 100), beta=brightness)

def augment_and_save_images(data_dir, output_dir, batch_size=10):
    """Performs data augmentation on images in batches and saves them."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_paths = glob(os.path.join(data_dir, '*.jpeg'))
    print(f"Found {len(image_paths)} images in {data_dir}.")

    # Process in batches
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1} of {len(image_paths) // batch_size + 1}")

        for img_path in batch:
            img = cv2.imread(img_path)
            augmented_images = augment_image(img)

            for j, aug_img in enumerate(augmented_images):
                output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_aug_{j}.jpeg")
                cv2.imwrite(output_path, aug_img)
                print(f"Saved augmented image to: {output_path}")

# Example usage
if __name__ == "__main__":
    data_dir = "data/sampleimages"  # Path to the original images
    output_dir = "data/augmented_images"  # Path for saving augmented images
    augment_and_save_images(data_dir, output_dir, batch_size=10)

