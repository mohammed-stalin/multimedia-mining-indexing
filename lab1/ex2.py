import cv2
import numpy as np

# Load the image with the green background
image = cv2.imread('fondvert.png')

# Load the new background image
background = cv2.imread('newback.jpeg')  # Replace with your background image

# Ensure the background is the same size as the input image
background = cv2.resize(background, (image.shape[1], image.shape[0]))

# Convert the image to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the range for the green color in HSV
lower_green = np.array([35, 40, 40])  # Adjust these values as needed
upper_green = np.array([85, 255, 255])

# Create a mask for the green color
mask = cv2.inRange(hsv_image, lower_green, upper_green)

# Invert the mask to get the non-green parts
mask_inv = cv2.bitwise_not(mask)

# Extract the foreground (non-green parts of the original image)
foreground = cv2.bitwise_and(image, image, mask=mask_inv)

# Extract the background where the green parts are
background = cv2.bitwise_and(background, background, mask=mask)

# Combine the foreground and the new background
result = cv2.add(foreground, background)

# Display the result
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
