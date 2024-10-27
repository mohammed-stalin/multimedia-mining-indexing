import cv2
import matplotlib.pyplot as plt

# Load the image
path='../assets/rose.jpg'
image = cv2.imread(path)
B, G, R = cv2.split(image)

# Merge channels back to form the original image in RGB for matplotlib
original_rgb = cv2.merge([R, G, B])  # Convert BGR to RGB

# Display all images in a 2x2 grid
plt.figure(figsize=(10, 8))

# Original image
plt.subplot(2, 2, 1)
plt.imshow(original_rgb)
plt.title("Original")
plt.axis("off")

# Blue channel
plt.subplot(2, 2, 2)
plt.imshow(B, cmap='gray')
plt.title("Blue Channel")
plt.axis("off")

# Green channel
plt.subplot(2, 2, 3)
plt.imshow(G, cmap='gray')
plt.title("Green Channel")
plt.axis("off")

# Red channel
plt.subplot(2, 2, 4)
plt.imshow(R, cmap='gray')
plt.title("Red Channel")
plt.axis("off")

plt.show()
