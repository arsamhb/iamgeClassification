import cv2
import argparse
import os
from PIL import Image
import numpy as np

def log_message(message, log_file="../data/cleaned/cropped_images/activity_log.txt"):
    with open(log_file, "a") as f:
        f.write(message + "\n")

def crop_image(input_image_path, output_image_path):
    image = cv2.imread(input_image_path)

    if image is None:
        log_message(f"Error: Unable to load the image {input_image_path}. Please check the file path.")
        print("Error: Unable to load the image. Please check the file path.")
        return

    # Get the dimensions of the image
    height, width, _ = image.shape

    # Calculate the cropping bounds
    height_crop = 65

    if height <= (height_crop + height_crop):
        log_message(f"Error: Image height is too small to crop for {input_image_path}.")
        print("Error: Image height is too small to crop by the specified amounts.")
        return

    # Perform the cropping
    cropped_image = image[height_crop:height - height_crop, :]

    # Save the cropped image
    cv2.imwrite(output_image_path, cropped_image)
    # log_message(f"Cropped image saved to {output_image_path}")
    print(f"Cropped image saved to {output_image_path}")

def detect_horizontal_lines(image_path):
    # Read the image using OpenCV
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 250, 252, apertureSize=3)

    # Use Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=250, minLineLength=50, maxLineGap=10)

    horizontal_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y1 - y2) < 10:  # Check if the line is approximately horizontal
                horizontal_lines.append(y1)

    # Sort horizontal lines by their y-coordinate and remove duplicates
    horizontal_lines = sorted(set(horizontal_lines))
    
    # log_message(f"Detected horizontal lines: {horizontal_lines}")
     # Add a zero at the start if the first value is greater than 15
    if horizontal_lines[0] > 15:
        horizontal_lines.insert(0, 0)

    # Add the height of the image at the end if the last value is far from the image height
    img_height = img.shape[0]
    if img_height - horizontal_lines[-1] > 50:
        horizontal_lines.append(img_height)
    
    # log_message(f"Adjusted horizontal lines after adding bounds: {horizontal_lines}")
        # Merge close lines by averaging their positions
    merged_lines = []
    i = 0
    while i < len(horizontal_lines):
        if i < len(horizontal_lines) - 1 and abs(horizontal_lines[i] - horizontal_lines[i + 1]) < 10:
            avg = (horizontal_lines[i] + horizontal_lines[i + 1]) // 2
            merged_lines.append(avg)
            i += 2  # Skip the next line since it's merged
        else:
            merged_lines.append(horizontal_lines[i])
            i += 1
    # log_message(f"Merged horizontal lines: {merged_lines}")
    print(merged_lines)
    # Widen horizontal lines to match the image width
    return [(0, y, img.shape[1], y) for y in merged_lines]
    
def split_image_into_grid(image_path, directory_name, image_name):
    # Detect horizontal lines
    horizontal_lines = detect_horizontal_lines(image_path)

    # Open the image
    img = Image.open(image_path)
    width, height = img.size

    # Hardcoded vertical lines since they are known
    vertical_lines = [0, width // 3, 2*(width // 3), width]  # Example for splitting into three columns

    slice_count = 0

    # Loop through each grid cell and save it as a separate image
    for i in range(len(horizontal_lines) - 1):
        for j in range(len(vertical_lines) - 1):
            # Calculate the coordinates of the current grid cell
            left = vertical_lines[j]
            right = vertical_lines[j + 1]
            upper = horizontal_lines[i][1]  # Use the y-coordinate of the current horizontal line
            lower = horizontal_lines[i + 1][1]  # Use the y-coordinate of the next horizontal line

            # Crop the image
            cropped_img = img.crop((left, upper, right, lower))

            # Save the cropped image
            output_file = f"../data/cleaned/cropped_images/{directory_name}_{image_name}_{i}_{j}.jpg"
            cropped_img.save(output_file)
            slice_count += 1
            # log_message(f"Saved slice: {output_file}")

    log_message(f"Total slices for {image_path}: {slice_count}")

def process_all_images(root_directory):
    subdirectories = [d for d in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, d))]

    for subdirectory in subdirectories:
        subdirectory_path = os.path.join(root_directory, subdirectory)

        image_files = [f for f in os.listdir(subdirectory_path) if os.path.isfile(os.path.join(subdirectory_path, f))]

        for image_file in image_files:
            input_image_path = os.path.join(subdirectory_path, image_file)
            output_image_path = f"../data/cleaned/cropped_images/{subdirectory}_{image_file}"

            crop_image(input_image_path, output_image_path)
            split_image_into_grid(output_image_path, subdirectory, image_file)

root_directory = '../data/raw/test'

if __name__ == "__main__":
    process_all_images(root_directory)
