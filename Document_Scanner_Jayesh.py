import cv2
import numpy as np

def order_points(pts):
    # Initialize a list of coordinates that will be ordered such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype=np.float32)
    
    # Summing the coordinates to find the top-left and bottom-right corners
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Computing the difference between the points to find the top-right and bottom-left corners
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def perspective_transform(image, pts):
    # Define the dimensions of the target window
    width, height = 800, 1000
    target_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    
    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(pts, target_pts)
    warped = cv2.warpPerspective(image, M, (width, height))
    
    return warped

# Load the input image
image = cv2.imread(r"C:\Users\JAYESH AHIRE\Downloads\WhatsApp Image 2023-11-23 at 8.19.49 PM.jpeg")
#image = cv2.imread(r"C:\Users\JAYESH AHIRE\Downloads\1718213187084.jpg")

# Resize the image to fit within a portrait frame
image = cv2.resize(image, (800, 1000))

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)

# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imshow("Blurred", blurred)

# Detect edges using Canny edge detector
edged = cv2.Canny(blurred, 30, 50)
cv2.imshow("Edges", edged)

# Find contours in the edged image and sort by area
contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Loop over the contours to find the document boundary (assuming the largest contour is the document)
for c in contours:
    perimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
    
    if len(approx) == 4:
        target_points = approx
        break

if len(target_points) != 4:
    print("Document corners not detected!")
    exit(1)

# Order the points to get them in top-left, top-right, bottom-right, and bottom-left order
ordered_points = order_points(target_points.reshape(-1, 2))

# Apply perspective transformation to get the scanned image
scanned_image = perspective_transform(image, ordered_points)

# Display the scanned image
cv2.imshow("Scanned", scanned_image)

# Wait for user input to close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
