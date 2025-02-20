#!pip install opencv-python opencv-python-headless
#!pip install ultralytics

##!sudo apt install tesseract-ocr
#!pip install pytesseract opencv-python opencv-python-headless

import cv2
from ultralytics import YOLO
import numpy as np
#from google.colab.patches import cv2_imshow
import os

def blur_license_plates_in_folder(folder_path, output_folder='Output_2', model_path='license_plate_detector.pt'):
    """
    Detects and blurs license plates in all images within a specified folder.
    
    Parameters:
    - folder_path (str): Path to the folder containing images.
    - output_folder (str): Path where modified images will be saved. Default: 'Output_2'.
    - model_path (str): Path to the YOLO model used for license plate detection. Default: 'license_plate_detector.pt'.
    """

    # Load the YOLO model for license plate detection
    model = YOLO(model_path)

    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all image files in the folder (JPG, JPEG, PNG)
    images = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

    for image_name in images:
        image_path = os.path.join(folder_path, image_name)

        # Read the image
        image = cv2.imread(image_path)
        print(f"Processing: {image_name}")

        # Perform inference using the YOLO model
        results = model(image)
        # Alternative prediction method (commented out)
        # results = model.predict(source=images, batch=8)

        # Loop through each detected object in the image
        for result in results:
            # Extract bounding box coordinates (x1, y1, x2, y2)
            boxes = result.boxes.xyxy.cpu().numpy()  # Convert to NumPy array
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)  # Convert float values to integers
                
                # Apply blur effect to the detected license plate region
                image[y1:y2, x1:x2] = cv2.blur(image[y1:y2, x1:x2], (50, 50))

                # Alternative approach: replace the plate area with a white rectangle (commented out)
                # cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), -1)

        # Save the modified image in the output folder with a prefix 'mod_'
        output_path = os.path.join(output_folder, 'mod_' + image_name)
        cv2.imwrite(output_path, image)
        print(f"Image saved in {output_path}")

# Example of usage
blur_license_plates_in_folder(folder_path='foto')
