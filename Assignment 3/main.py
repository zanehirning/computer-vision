import matplotlib.pyplot as plt
import numpy as np
import cv2


# Question 1.1
def averaging_filter(im, mask):
    # im: image to be filtered
    # mask: filter mask (odd size)
    if np.all(mask) < 0:
        raise ValueError('Mask has negative values')
    if len(mask) != len(mask[0]):
        raise ValueError('Mask is not square')
    if len(mask) % 2 == 0:
        raise ValueError('Mask size is not odd')
    if np.sum(mask) != 1:
        raise ValueError('Mask does not sum to 1')
    #Check if mask is symmetric across center
    if not np.allclose(mask, np.flip(mask, axis=None)):
        raise ValueError('Mask is not symmetric across center')
    
    padding_size = int((len(mask) - 1) / 2)
    padded_im = np.pad(im, padding_size, mode='constant')
    filtered_im = padded_im.copy()
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            filtered_im[i, j] = np.sum(mask * padded_im[i:i+len(mask), j:j+len(mask)])
    return filtered_im[1:-1, 1:-1]


circuit_im = cv2.imread('Circuit.jpg', cv2.IMREAD_GRAYSCALE)

standard_circuit_filtered = averaging_filter(circuit_im, np.ones((5, 5), dtype=np.float32)/25)
weighted_circuit_filtered = averaging_filter(circuit_im, np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])/16)
plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(circuit_im, cmap='gray')
plt.title('Original Image')
plt.subplot(1, 3, 2)
plt.imshow(standard_circuit_filtered, cmap='gray')
plt.title('5x5 Standard Averaging Filter')
plt.subplot(1, 3, 3)
plt.imshow(weighted_circuit_filtered, cmap='gray')
plt.title('3x3 Weighted Averaging Filter')
plt.tight_layout()
plt.show()

# Question 1.2
def median_filter(im, mask):
    if np.all(mask) < 0:
        raise ValueError('Mask has negative values')
    if len(mask) != len(mask[0]):
        raise ValueError('Mask is not square')
    if len(mask) % 2 == 0:
        raise ValueError('Mask size is not odd')
    
    padding_size = int((len(mask) - 1) / 2)
    padded_im = np.pad(im, padding_size, mode='constant')
    filtered_im = padded_im.copy()
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            filtered_im[i, j] = np.median(padded_im[i:i+len(mask), j:j+len(mask)])
    return filtered_im[1:-1, 1:-1]



# Apply the standard 3x3 median filter using OpenCV's cv2.medianBlur
standard_median_circuit = median_filter(circuit_im, np.ones((3, 3), dtype=np.float32))
weighted_median_circuit = median_filter(circuit_im, [[1, 2, 1], [2, 4, 2], [1, 2, 1]])

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(circuit_im, cmap='gray')
plt.title('Original Image')
plt.subplot(1, 3, 2)
plt.imshow(standard_median_circuit, cmap='gray')
plt.title('3x3 Standard Median Filter')
plt.subplot(1, 3, 3)
plt.imshow(weighted_median_circuit, cmap='gray')
plt.title('3x3 Weighted Median Filter')
plt.tight_layout()
plt.show()

# Question 1.3

moon_im = cv2.imread('Moon.jpg')

# 3x3 strong laplacean filter
laplacian_filter = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float64)
moon_filtered_im = cv2.filter2D(moon_im, cv2.CV_32F, kernel=laplacian_filter)

moon_clipped_im = np.clip(moon_filtered_im, 0, 255).astype(np.uint8)

min_value = np.min(moon_filtered_im)
scaled_filtered_image = (moon_filtered_im - min_value)
max_value = np.max(scaled_filtered_image)
scaled_filtered_image = (scaled_filtered_image / max_value) * 255

# Convert to integer values
scaled_moon_im = scaled_filtered_image.astype(np.uint8)

# scaled_moon_im = cv2.convertScaleAbs(moon_filtered_im)
moon_enhanced_im = cv2.add(moon_im, moon_filtered_im, dtype=cv2.CV_8U)

plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(moon_im, cmap='gray')
plt.title('Original Image')
plt.subplot(2, 2, 2)
plt.imshow(moon_clipped_im, cmap='gray')
plt.title('Filtered Image Clipped')
plt.subplot(2, 2, 3)
plt.imshow(scaled_moon_im, cmap='gray')
plt.title('Scaled Filtered Image')
plt.subplot(2, 2, 4)
plt.imshow(moon_enhanced_im, cmap='gray')
plt.title('Enhanced Image')
plt.tight_layout()
plt.show()

# Question 2
def find_edge_info(im, bin):
    G_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    G_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    padding_size = int((len(G_x) - 1) / 2)
    padded_im = np.pad(im, padding_size, mode='constant')
    vertical_edges = padded_im.copy()
    horizontal_edges = padded_im.copy()

    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            vertical_edges[i, j] = np.sum(G_x * padded_im[i:i+len(G_x), j:j+len(G_x)])
            horizontal_edges[i, j] = np.sum(G_y * padded_im[i:i+len(G_y), j:j+len(G_y)])
    
    magnitude = np.sqrt(np.square(vertical_edges) + np.square(horizontal_edges))
    direction = (np.arctan2(horizontal_edges, vertical_edges) * 180) / np.pi
    direction = direction + 180

    #compute histogram with desired number of bins
    edgeHist = np.zeros(bin)
    for i in range(bin):
        bin_start_value = i * 360 / bin
        bin_end_value = ((i + 1) * 360) / bin
        bin_range = (bin_start_value, bin_end_value)
        #check if the value in direction is >= start and < end, if so, add to bin
        edgeHist[i] = np.sum((direction >= bin_range[0]) & (direction < bin_range[1]))

    return magnitude, edgeHist

rice_im = cv2.imread('Rice.jpg', cv2.IMREAD_GRAYSCALE)
magnitude, edgeHist = find_edge_info(rice_im, 30)

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(rice_im, cmap='gray')
plt.title('Original Image')
plt.subplot(1, 3, 2)
plt.imshow(magnitude, cmap='gray')
plt.title('Magnitude')
plt.subplot(1, 3, 3)
plt.bar(np.arange(30), edgeHist)
plt.title('Edge Histogram')
plt.tight_layout()
plt.show()

# Question 3
def remove_stripes(im):
    #removing stripes without frequency domain
    original_image = im.copy()
    histogram = [0] * 256

    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            histogram[im[i][j]] += 1

    # Apply the mask to the original image to remove the stripes
    cleaned_image = original_image.copy()

    #threshold
    cleaned_image[im >= 70] = 255
    cleaned_image[im < 70] = 0

    return cleaned_image #+clipped_im
    #compute

#read in text images
video = cv2.VideoCapture('Text.gif')
_, text_im = video.read()
video.release()

video = cv2.VideoCapture('Text1.gif')
_, text1_im = video.read()
video.release()

text_im = cv2.cvtColor(text_im, cv2.COLOR_BGR2GRAY)
text1_im = cv2.cvtColor(text1_im, cv2.COLOR_BGR2GRAY)

text_clipped = remove_stripes(text_im)
text1_clipped = remove_stripes(text1_im)

plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(text_im, cmap='gray')
plt.title('Text Original Image')
plt.subplot(2, 2, 2)
plt.imshow(text_clipped, cmap='gray')
plt.title('Text Filtered Image')
plt.subplot(2, 2, 3)
plt.imshow(text1_im, cmap='gray')
plt.title('Text1 Original Image')
plt.subplot(2, 2, 4)
plt.imshow(text1_clipped, cmap='gray')
plt.title('Text1 Filtered Image')
plt.tight_layout()
plt.show()