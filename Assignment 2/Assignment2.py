import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import time

# Read the image
foodIm = cv2.imread('food.jpg')
foodIm = cv2.cvtColor(foodIm, cv2.COLOR_BGR2GRAY)

# Question 1 incomplete
def scale(inputIm, inputRange):
    if inputRange[0] > inputRange[1]:
        print("Please provide a valid inputRange")
        return
    if inputRange[0] < 0 or inputRange[1] < 0:
        print("Please provide a positive inputRange")
        return
    if not isinstance(inputRange[0], int) or not isinstance(inputRange[1], int):
        print("Please provide an integer inputRange")
        return
    scaledIm = inputIm.copy()
    x_range = np.max(inputIm) - np.min(inputIm)
    y_range = inputRange[1] - inputRange[0]
    #y = mx + b
    m = y_range / x_range
    
    #for original image
    b = -(m * np.min(inputIm))
    # Transform the values in the input image
    for i in range(len(inputIm[0])):
        for j in range(len(inputIm[1])):
            scaledIm[i][j] = (m * inputIm[i][j]) + b

    transFunc = [(m * x) + b for x in range(np.min(inputIm), np.max(inputIm)+1)]
    return scaledIm, transFunc


scale(foodIm, [0, 210.7])
scale(foodIm, [200, 10])
scale(foodIm, [-100, 10])
scaledFoodIm, scaledTransFunc = scale(foodIm, [0, 255])

# Display the image
plt.imshow(scaledFoodIm, cmap='gray')
plt.show()
plt.plot(scaledTransFunc)
plt.title("Scaled Transformation Function")
plt.ylabel("Output Intensity")
plt.xlabel("Input Intensity")
plt.show()
# plt.imshow(scaledFoodim, cmap='gray')
# plt.show()

# Question 2
def CalHist(inputIm, normalized=False):
    if inputIm.ndim != 2:
        raise ValueError("Input image must be grayscale (2D array).")

    # Calculate the histogram
    histogram = [0] * 256

    # Iterate through the image and count pixel values
    for row in inputIm:
        for pixel_value in row:
            histogram[pixel_value] += 1

    if normalized:
        total_pixels = np.max(histogram)
        histogram = [count / total_pixels for count in histogram]

    return histogram

hist = CalHist(scaledFoodIm, False)
# Calculate the normalized histogram
normalized_hist = CalHist(scaledFoodIm, True)

# Create a figure with two subplots for the histogram and normalized histogram
plt.figure(figsize=(12, 4))

# Plot the histogram
plt.subplot(1, 2, 1)
plt.bar(list(range(256)), hist, width=1)
plt.title("Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")

# Plot the normalized histogram
plt.subplot(1, 2, 2)
plt.bar(list(range(256)), normalized_hist, width=1)
plt.title("Normalized Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Normalized Frequency")

# Display the histograms side-by-side
plt.tight_layout()
plt.show()

# Question 3

def HistEqualization(inputIm):
    if inputIm.ndim != 2:
        raise ValueError("Input image must be grayscale (2D array).")

    histogram = CalHist(inputIm, normalized=False)
    # Compute the cumulative normalized histogram

    cdf = [sum(histogram[:i + 1])/(inputIm.shape[1]*inputIm.shape[0]) for i in range(len(histogram))] #T(k)
    #Compute the transformed intesity by g_k = T(k) * (L-1) where L is the maximum gray level

    transFunc = [(len(histogram) - 1) * cdf[i] for i in range(len(cdf))]
    transFunc = np.round(transFunc).astype(np.uint8) # rounding and converting to uint8

    # Scan the image and transform old values into transFunc values
    enhancedIm = inputIm.copy()
    for i in range(len(inputIm[0])):
        for j in range(len(inputIm[1])):
            enhancedIm[i][j] = transFunc[inputIm[i][j]]

    return enhancedIm, transFunc

#Display the running time of using the HistEqualization function to accomplish the task on the console.
start_time = time.time()
equalizedFoodIm, equalizedTransFunc = HistEqualization(foodIm)
execution_time = time.time() - start_time
print("Execution time: " + str(execution_time) + " seconds")


# Question 4
start_time = time.time()
# Perform histogram equalization using cv2.equalizeHist
enhancedFoodIm = cv2.equalizeHist(foodIm)
execution_time = time.time() - start_time
print("Execution time: " + str(execution_time) + " seconds")

#Question 5

def BBHE(inputIm):
    #Compute X_m, X_L, X_H
    
    x_m = np.mean(inputIm)

    # Initialize histograms for dark and bright regions
    hist_dark = np.zeros(256, dtype=np.uint32)
    hist_bright = np.zeros(256, dtype=np.uint32)

    # Compute histograms for dark and bright regions
    for pixel_value in inputIm.flatten():
        if pixel_value <= x_m:
            hist_dark[pixel_value] += 1
        else:
            hist_bright[pixel_value] += 1

    # Calculate CDF (cumulative distribution function) for dark and bright regions
    cdf_dark = np.zeros(256, dtype=np.uint32)
    cdf_bright = np.zeros(256, dtype=np.uint32)

    cdf_dark[0] = hist_dark[0]
    cdf_bright[0] = hist_bright[0]

    for i in range(1, 256):
        cdf_dark[i] = cdf_dark[i - 1] + hist_dark[i]
        cdf_bright[i] = cdf_bright[i - 1] + hist_bright[i]

    # Normalize CDFs to the range [0, 1]
    cdf_dark_normalized = cdf_dark / cdf_dark[-1]
    cdf_bright_normalized = cdf_bright / cdf_bright[-1]

    # Calculate transformation functions
    transFuncL = np.zeros(256, dtype=np.uint8)
    transFuncH = np.zeros(256, dtype=np.uint8)

    for i in range(256):
        transFuncL[i] = round(x_m * cdf_dark_normalized[i])
        transFuncH[i] = round(((x_m + 1) + (255 - x_m - 1)) * cdf_bright_normalized[i])


    enhanced_image = np.zeros_like(inputIm, dtype=np.uint8)
    for i in range(inputIm.shape[0]):
        for j in range(inputIm.shape[1]):
            if inputIm[i, j] <= x_m:
                enhanced_image[i, j] = transFuncL[inputIm[i, j]]
            else:
                enhanced_image[i, j] = transFuncH[inputIm[i, j]]

    return enhanced_image, transFuncL, transFuncH
    
start_time = time.time()
BBHEFoodIm, transFuncL, transFuncH = BBHE(foodIm)
execution_time = time.time() - start_time
print("Execution time: " + str(execution_time) + " seconds")

#plotting question 3 and 5 transformation functions side by side. Question 4 does not return a transformation function
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(equalizedTransFunc)
plt.title("Equalized Transformation Function")
plt.ylabel("Output Intensity")
plt.xlabel("Input Intensity")

plt.subplot(1, 3, 2)
plt.plot(transFuncL)
plt.title("BBHE Dark Histogram Transformation Function")
plt.ylabel("Output Intensity")
plt.xlabel("Input Intensity")

plt.subplot(1, 3, 3)
plt.plot(transFuncH)
plt.title("BBHE Light Histogram Transformation Function")
plt.ylabel("Output Intensity")
plt.xlabel("Input Intensity")

plt.tight_layout()
plt.show()

#PSNR stuff
def PSNR(inputIm, enhancedIm):
    if inputIm.shape != enhancedIm.shape:
        raise ValueError("Input and reference images must have the same dimensions.")

    mean_squared_error = np.mean((inputIm - enhancedIm) ** 2)
    psnr = 10 * math.log10((255 ** 2) / mean_squared_error)

    return psnr

psnr_equalized = PSNR(foodIm, equalizedFoodIm)
psnr_BBHE = PSNR(foodIm, BBHEFoodIm)
print("PSNR of equalizedFoodIm: " + str(psnr_equalized))
print("PSNR of BBHEFoodIm: " + str(psnr_BBHE))
