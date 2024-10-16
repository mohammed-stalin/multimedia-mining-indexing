import cv2
from sklearn.cluster import KMeans
import numpy as np


# Function to extract dominant colors using KMeans
def extract_dominant_colors(img, n_colors=10):
    imgLab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    height, width, _ = imgLab.shape
    dim = (int(width / 5), int(height / 5))
    imgExemples = cv2.resize(imgLab, dim)
    examples = imgExemples.reshape((imgExemples.shape[0] * imgExemples.shape[1], 3))
    kmeans = KMeans(n_clusters=n_colors, n_init=10)
    kmeans.fit(examples)
    colors = kmeans.cluster_centers_.astype(int)
    colors = np.sort(colors, axis=0)
    return colors


# Function to compute similarity between two sets of dominant colors
def calculate_color_similarity(colors1, colors2):
    assert colors1.shape == colors2.shape, "Color arrays must have the same shape"
    distances = np.linalg.norm(colors1 - colors2, axis=1)
    mean_distance = np.mean(distances)
    return mean_distance


# Function to create a visual representation of dominant colors
def create_color_bar(colors, width, height):
    img_colors = np.zeros((height, width, 3), dtype=np.uint8)
    bar_color_width = width // len(colors)

    for i, color in enumerate(colors):
        cv2.rectangle(img_colors, (i * bar_color_width, 0), ((i + 1) * bar_color_width, height), color.tolist(), -1)

    return img_colors


# Function to resize an image to 300x300 pixels
def resize_image(image, size=(300, 300)):
    return cv2.resize(image, size)


# Main function to extract and compare dominant colors of two images
def main():
    # Read two images
    img1 = cv2.imread("..\\lab2\\car.jpg", 1)
    img2 = cv2.imread("..\\lab2\\car.jpg", 1)

    if img1 is None:
        print("Error: Could not load image1. Check the path.")
        return
    if img2 is None:
        print("Error: Could not load image2. Check the path.")
        return

    # Extract dominant colors for both images
    nbreDominantColors = 10
    dominant_colors_img1 = extract_dominant_colors(img1, nbreDominantColors)
    dominant_colors_img2 = extract_dominant_colors(img2, nbreDominantColors)

    # Compute the similarity between the two sets of dominant colors
    similarity = calculate_color_similarity(dominant_colors_img1, dominant_colors_img2)
    print(f"Color similarity between the two images: {similarity:.2f}")

    # Create color bars for each set of dominant colors
    color_bar_height = 50
    color_bar_img1 = create_color_bar(dominant_colors_img1, 300, color_bar_height)
    color_bar_img2 = create_color_bar(dominant_colors_img2, 300, color_bar_height)

    # Resize images to 300x300 pixels
    img1_resized = resize_image(img1, (300, 300))
    img2_resized = resize_image(img2, (300, 300))

    # Combine images and their color bars
    combined_image = np.vstack((img1_resized, color_bar_img1, img2_resized, color_bar_img2))

    # Display the combined image
    cv2.imshow("Combined Image with Dominant Colors", combined_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
