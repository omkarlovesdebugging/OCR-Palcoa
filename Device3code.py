import cv2 
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import numpy as np
from google.cloud import vision
from decimal import Decimal, InvalidOperation
from shapely.geometry import Polygon

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'vision.json'

# Read and resize the image
img_color = cv2.imread(r"mini-proj-assets\Device3\797.1.jpeg")  # Updated image path
img_color = cv2.resize(img_color, None, None, fx=0.7, fy=0.7)  # Resize image

# Convert to grayscale
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# Use adaptive thresholding to focus on the bright area
# Tweak the parameters for better region detection
thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8)

# Dilate the thresholded image to better capture the region
kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(thresh, kernel, iterations=2)

# Find contours on the dilated image
cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter out contours that are too small or not rectangular (we expect the region to be large and roughly rectangular)
filtered_contours = [c for c in cnts if cv2.contourArea(c) > 500]  # Adjusted threshold for smaller areas
filtered_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)

# Assuming the display screen is one of the larger areas, select the largest contour
largest_contour = filtered_contours[0]

# Create a mask for the selected contour (assuming it's the display area)
mask = np.zeros_like(img_gray)
cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

# Extract the region of interest (ROI) using the mask
roi = cv2.bitwise_and(img_color, img_color, mask=mask)

# Find the bounding box of the selected contour to crop the ROI
x, y, w, h = cv2.boundingRect(largest_contour)
cropped_green_screen = roi[y:y+h, x:x+w]

# Display the cropped region (ROI) without performing text detection
cv2.imshow("ROI Cropped ", cropped_green_screen)

# Display the final image
# plt.imshow(cv2.cvtColor(cropped_green_screen, cv2.COLOR_BGR2RGB))
# plt.show()
cv2.imwrite("cropped_green_screen.jpeg",cropped_green_screen)

# ---------------------------------------------------------------------------------------------------------
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
    ocr_data = detect_text(r'C:\Users\ghola\OneDrive\Desktop\Mini project\OCR-Project-SEM-5-\cropped_green_screen.jpeg')  # Detect text and get bounding boxes
    draw_bounding_boxes(r'C:\Users\ghola\OneDrive\Desktop\Mini project\OCR-Project-SEM-5-\cropped_green_screen.jpeg', ocr_data)  # Draw bounding boxes on the image
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

    try:
        output = next((item['text'] for item in preout if item['area'] == sorted_area[1]), None) 
        int(output)
        print("The Output is :",next((item['text'] for item in preout if item['area'] == sorted_area[1]), None))
    except: 
        print("No Output.")
    # for data in ocr_data:
    #     try:
    #         num = Decimal(data['text'])  # Try to convert text to a number
    #         print(data['text'])  # Print the detected text
    #     except InvalidOperation:
    #         continue

if __name__ == "__main__":
    main()
