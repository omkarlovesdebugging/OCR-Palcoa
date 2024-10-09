import cv2
import numpy as np
import os
import cv2
import numpy as np
from google.cloud import vision
from decimal import Decimal, InvalidOperation
from shapely.geometry import Polygon

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'vision.json'

# Load the first image (blue screen)
blue_screen_image = cv2.imread('mini-proj-assets/Device1/26.3.jpeg')
# Load the second image (light screen)
# light_screen_image = cv2.imread('mini-proj-assets/180(2).jpeg')

def detect_blue_screen(image):
    """Detect the blue screen in the image."""
    # Convert the image to HSV (Hue, Saturation, Value) color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the color blue in HSV
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])

    # Create a mask that isolates the blue areas of the image
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find contours in the mask to detect the blue region
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour which should correspond to the blue screen
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the bounding box coordinates of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Crop the image using the bounding box coordinates
        cropped_screen = image[y:y+h, x:x+w]

        # Draw a rectangle around the detected blue screen (for visualization)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        return cropped_screen, image
    else:
        print("No blue screen detected.")
        return None, image

# def detect_light_screen(image):
#     """Detect the light-colored screen in the image."""
#     # Convert the image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Apply a threshold to detect bright areas (light-colored screen)
#     _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

#     # Find contours in the thresholded image
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     if contours:
#         # Find the largest contour which should correspond to the light screen
#         largest_contour = max(contours, key=cv2.contourArea)

#         # Get the bounding box coordinates of the largest contour
#         x, y, w, h = cv2.boundingRect(largest_contour)

#         # Crop the image using the bounding box coordinates
#         cropped_screen = image[y:y+h, x:x+w]

#         # Draw a rectangle around the detected light screen (for visualization)
#         cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

#         return cropped_screen, image
#     else:
#         print("No light screen detected.")
#         return None, image

# Detect blue screen in the first image
cropped_blue_screen, blue_screen_image_with_box = detect_blue_screen(blue_screen_image)

# Detect light screen in the second image
# cropped_light_screen, light_screen_image_with_box = detect_light_screen(light_screen_image)

# Display the original image with the detected screen outlined (for both images)
cv2.imshow('Original Image with Detected Blue Screen', blue_screen_image_with_box)
# cv2.imshow('Original Image with Detected Light Screen', light_screen_image_with_box)

# Display the cropped screen (if detected)
if cropped_blue_screen is not None:
    cv2.imshow('Cropped Blue Screen', cropped_blue_screen)
    cv2.imwrite('cropped_blue_screen.jpeg',cropped_blue_screen)

# if cropped_light_screen is not None:
#     cv2.imshow('Cropped Light Screen', cropped_light_screen)

# Wait for a key press and then close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()

#------------------------------------------------------------------------------------------------------------------

def detect_text(path):
    client = vision.ImageAnnotatorClient()
    with open(path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    
    ocr_data = []  # List to hold text and bounding box data
    for text in texts:
        ocr_text = text.description
        vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
        ocr_data.append({"text": ocr_text, "bounds": vertices})

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )

    return ocr_data  # Return structured data

def draw_bounding_boxes(image_path, ocr_data):
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    for data in ocr_data:
        text = data['text']
        bounds = data['bounds']

        # Convert the bounds to a polygon and draw the bounding box
        points = [(int(x), int(y)) for x, y in bounds]
        cv2.polylines(image, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)

        # Optionally, put the text near the bounding box
        cv2.putText(image, text, (points[0][0], points[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the image with bounding boxes
    cv2.imshow("Detected Text with Bounding Boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # cropped_image_path = 'cropped_largest_rectangle.jpeg'
    ocr_data = detect_text(r'C:\Users\ghola\OneDrive\Desktop\Mini project\OCR-Project-SEM-5-\cropped_blue_screen.jpeg')  # Detect text and get bounding boxes
    draw_bounding_boxes(r'C:\Users\ghola\OneDrive\Desktop\Mini project\OCR-Project-SEM-5-\cropped_blue_screen.jpeg', ocr_data)  # Draw bounding boxes on the image
    preout =[]
    areas=[]
    for data in ocr_data:
        # print(data)
        vertices = data['bounds']
        rectangle = Polygon(vertices)
        # print(data['text']," Area = ",rectangle.area)
        areas.append(int(rectangle.area))
        preout.append({"text": data['text'], "area": int(rectangle.area)})

    # areas.sort()    
    print(areas)
    sorted_area = sorted(areas,reverse=True)
    print(sorted_area)
    # print(ocr_data)

    print("The Output is :",next((item['text'] for item in preout if item['area'] == sorted_area[1]), None))

    # for data in ocr_data:
    #     try:
    #         num = Decimal(data['text'])  # Try to convert text to a number
    #         print(data['text'])  # Print the detected text
    #     except InvalidOperation:
    #         continue

if __name__ == "__main__":
    main()
