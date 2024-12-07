import cv2
import os
import numpy as np


image_path = '../data/raw/fashion/antoniofdez_.jpg' 
output_folder = '../data/cleaned/cropped_images/'  
os.makedirs(output_folder, exist_ok=True)

image = cv2.imread(image_path)

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_bound = np.array([0, 0, 250])  
upper_bound = np.array([180, 120, 255])  

mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (20, 20))
cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_BLACKHAT, kernel)
contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contour_image = image.copy()

image_count = 0
contour_count = 0
for contour in contours:
    contour_count += 1

    x, y, w, h = cv2.boundingRect(contour)

    cv2.rectangle(contour_image, (x, y), (x+w, y+h), (0, 0, 255), 2)  

    if w > 200 and w < 250 and h > 50 and h < 250 or w > 50 and w < 250 and h > 200 and h < 250:
        cropped_image = image[y:y+h, x:x+w]
        cv2.rectangle(contour_image, (x, y), (x+w, y+h), (0, 255, 0), 2) 

        output_path = os.path.join(output_folder, f'cropped_{image_count}.jpg')
        cv2.imwrite(output_path, cropped_image)
        image_count += 1

contour_image_path = '../data/cleaned/cropped_images/contours_drawn.jpg'
mask_image_path = '../data/cleaned/cropped_images/mask.jpg'
cleaned_mask_image_path = '../data/cleaned/cropped_images/cleaned_mask.jpg'

cv2.imwrite(contour_image_path, contour_image)
cv2.imwrite(mask_image_path, mask)
cv2.imwrite(cleaned_mask_image_path, cleaned_mask)

print(f"Found {contour_count} COUNTORS")
print(f"Saved {image_count} cropped images")
