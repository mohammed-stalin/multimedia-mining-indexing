import cv2
import numpy as np
from matplotlib import pyplot as plt

def build_gabor_filters():
    """
    Creates a bank of Gabor filters with 4 orientations and 3 scales.
    These filters help in detecting specific textures in the image.
    """
    filters = []
    ksize = 31  # Kernel size
    lambd = 10.0  # Wavelength of the sinusoidal factor
    sigma = 5.0  # Standard deviation of the Gaussian function
    gamma = 0.5  # Spatial aspect ratio

    orientations = [0, 45, 90, 135]  # Filter orientations in degrees
    scales = [3, 5, 7]  # Different scales for the Gabor filters

    for theta in orientations:
        for scale in scales:
            kernel = cv2.getGaborKernel(
                (ksize, ksize),
                sigma=sigma * scale / 3,
                theta=np.deg2rad(theta),
                lambd=lambd,
                gamma=gamma,
                psi=0,
                ktype=cv2.CV_32F
            )
            kernel /= 1.5 * kernel.sum()  # Normalizing filter for consistency
            filters.append(kernel)
    return filters

def apply_gabor_filters(img, filters):
    """
    Applies a list of Gabor filters to an image and returns the filtered images.
    """
    filtered_images = [cv2.filter2D(img, cv2.CV_8UC3, kern) for kern in filters]
    return filtered_images

def extract_region_features(region, filtered_images):
    """
    Extracts statistical features (mean and standard deviation) from a region.
    These features characterize the texture in the region.
    """
    stats = []
    for img in filtered_images:
        stats.append(np.mean(img))
        stats.append(np.std(img))
    return np.array(stats)

def calculate_similarity(vec1, vec2):
    """
    Calculates similarity between two feature vectors using Euclidean distance.
    """
    return np.linalg.norm(vec1 - vec2)

def texture_extraction(image_path, x, y, width, height, threshold=100):
    """
    Main function for texture extraction based on a selected region of interest (ROI).
    Generates a binary output image with matching texture regions highlighted.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not load the image")

    filters = build_gabor_filters()
    full_image_features = apply_gabor_filters(img, filters)

    roi = img[y:y+height, x:x+width]
    roi_filtered = apply_gabor_filters(roi, filters)
    roi_stats = extract_region_features(roi, roi_filtered)

    result = np.zeros_like(img)
    window_size = (height, width)

    for i in range(0, img.shape[0] - window_size[0], 4):
        for j in range(0, img.shape[1] - window_size[1], 4):
            window = img[i:i+window_size[0], j:j+window_size[1]]
            window_filtered = apply_gabor_filters(window, filters)
            window_stats = extract_region_features(window, window_filtered)

            similarity = calculate_similarity(roi_stats, window_stats)
            if similarity < threshold:
                result[i:i+window_size[0], j:j+window_size[1]] = 255

    return result

def select_region(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    roi = cv2.selectROI("Select ROI", img)
    cv2.destroyAllWindows()
    return roi

def main():
    image_paths = ["a.jpg", "b.jpg", "c.jpg"]

    for image_path in image_paths:
        x, y, w, h = select_region(image_path)
        result = texture_extraction(image_path, x, y, w, h, threshold=80)  # Adjust threshold as needed

        plt.figure(figsize=(12, 4))

        plt.subplot(131)
        plt.imshow(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), cmap='gray')
        plt.title('Original Image')

        plt.subplot(132)
        roi = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)[y:y+h, x:x+w]
        plt.imshow(roi, cmap='gray')
        plt.title('Selected Region')

        plt.subplot(133)
        plt.imshow(result, cmap='gray')
        plt.title('Extracted Texture')

        plt.show()

if __name__ == "__main__":
    main()
