# Name: Zane Hirning
# A-number: A02371568

import matplotlib.pyplot as plt
import numpy as np
import cv2

#Question 1.
pepperIm = plt.imread("peppers.bmp")
lenaIm = plt.imread("lena.jpg")

plt.figure()
plt.suptitle("Original Images")
plt.subplot(1, 2, 1)
plt.imshow(pepperIm)
plt.subplot(1, 2, 2)
plt.imshow(lenaIm, cmap="gray")
# plt.show()

#Question 2.
#convert pepperIm to grayscale
pepperGrayIm = cv2.cvtColor(pepperIm, cv2.COLOR_RGB2GRAY)
plt.imshow(pepperGrayIm, cmap="gray")

#transpose pepperGrayIm
pepperGrayImT = pepperGrayIm.T
plt.imshow(pepperGrayImT, cmap="gray")

#flip pepperGrayIm across the x axis
pepperGrayImF = np.flip(pepperGrayIm, 1)
#flip pepperGrayIm across the y axis
pepperGrayImH = np.flip(pepperGrayIm, 0)

#display all 4 images
plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(pepperGrayIm, cmap="gray")
plt.title("pepperGrayIm")

plt.subplot(2, 2, 2)
plt.imshow(pepperGrayImT, cmap="gray")
plt.title("pepperGrayImT")

plt.subplot(2, 2, 3)
plt.imshow(pepperGrayImH, cmap="gray")
plt.title("pepperGrayImH")

plt.subplot(2, 2, 4)
plt.imshow(pepperGrayImF, cmap="gray")
plt.title("pepperGrayImF")

#add space
plt.tight_layout()
# plt.show()

#Question 3.
#inbuilt functions
lenaMax = np.max(lenaIm)
lenaMin = np.min(lenaIm)
lenaMean = np.mean(lenaIm)
lenaMedian = np.median(lenaIm)

def quickSort(im):
    if len(im) <= 1:
        return im  # Base case: if the array has 0 or 1 elements, it's already sorted.

    pivot = im[len(im) // 2]  # Choose a pivot element (middle of the array in this example)
    left = [x for x in im if x < pivot]  # Elements less than the pivot
    middle = [x for x in im if x == pivot]  # Elements equal to the pivot
    right = [x for x in im if x > pivot]  # Elements greater than the pivot

    return quickSort(left) + middle + quickSort(right)

#manual functions
def findInfo(im) -> int:
    max_value=0 #min value
    min_value=255 #max value
    mean_value = 0
    #flatten, sort, take middle value
    flattened_array = [value for row in im for value in row]
    sorted_im = quickSort(flattened_array)
    median_value = sorted_im[len(sorted_im)//2]

    #find max, min, mean
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i][j] > max_value:
                max_value = im[i][j]
            if im[i][j] < min_value:
                min_value = im[i][j]
            mean_value += im[i][j]

    mean_value = mean_value/(im.shape[0]*im.shape[1])
    return max_value, min_value, mean_value, median_value
#this could have been done without the for loops. Could just take sorted_im[0] and sorted_im[len(sorted_im)-1] for min and max

myLenaMax, myLenaMin, myLenaMean,myLenaMedian = findInfo(lenaIm)

#check max values via assignment description
if lenaMax == myLenaMax:
    print("Max values match")
else:   
    print("Max values do not match")

#check min values via assignment description
if lenaMin == myLenaMin:
    print("Min values match")
else:
    print("Min values do not match")

#check mean values via assignment description
if lenaMean == myLenaMean:
    print("Mean values match")
else:
    print("Mean values do not match")

#check median values via assignment description
if lenaMedian == myLenaMedian:
    print("Median values match")
else:
    print("Median values do not match")


#Question 4.
normalizedLenaIm = lenaIm.astype(float) / lenaMax
plt.figure()
plt.suptitle("Normalized Lena Image")
plt.imshow(normalizedLenaIm, cmap="gray")
# plt.show()

#getting the second and fourth quarters of the image
first_quarter_end = normalizedLenaIm.shape[0]//4
second_quarter_start = normalizedLenaIm.shape[0]//2
third_quarter_end = 3*(normalizedLenaIm.shape[0]//4)
fourth_quarter_start = 4*(normalizedLenaIm.shape[0])

processedNormalizedLenaIm = normalizedLenaIm.copy()
processedNormalizedLenaIm[first_quarter_end:second_quarter_start] = processedNormalizedLenaIm[first_quarter_end:second_quarter_start] ** 1.25
processedNormalizedLenaIm[third_quarter_end:fourth_quarter_start] = processedNormalizedLenaIm[third_quarter_end:fourth_quarter_start] ** 0.25

plt.figure()
plt.suptitle("Processed Grayscale Image")
plt.imshow(processedNormalizedLenaIm, cmap="gray")
# plt.show()

#write image
cv2.imwrite("Zane_processedNormalizedLenaIm.jpg", processedNormalizedLenaIm)

#Question 5.
pepperGrayImN = pepperGrayIm.astype(float) / np.max(pepperGrayIm)
threshold = .37
bw1 = (pepperGrayImN > threshold) * 1
bw2 = (pepperGrayImN > threshold).astype(int)
#inbuilt function
bw3 = cv2.threshold(pepperGrayImN, threshold, 1, cv2.THRESH_BINARY)[1]

#check if equal
if (bw1 == bw2).all() and (bw2 == bw3).all():
    print("Both of my methods worked")
elif (bw1 == bw3).all() and (bw2 != bw3).all():
    print("My method 1 works but not my method 2")
elif (bw2 == bw3).all() and (bw1 != bw3).all():
    print("My method 2 works but not my method 1")
else:
    print("Both of my methods did not work")
#Note: ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all() // also from assignment hint:)
#display bw1, bw2, bw3
plt.figure()
plt.suptitle("Binary Images")
plt.subplot(1, 3, 1)
plt.imshow(bw1, cmap="gray")
plt.title("My First Method")

plt.subplot(1, 3, 2)
plt.imshow(bw2, cmap="gray")
plt.title("My Second Method")

plt.subplot(1, 3, 3)
plt.imshow(bw3, cmap="gray")
plt.title("Built-in Method")
plt.tight_layout()
# plt.show()

#Question 6.

def generateBlurImage(im, n):
    blurred_im = im.copy()
    for i in range(0, im.shape[0], n):
        for j in range(0, im.shape[1], n):
            section = im[i:i+n, j:j+n]
            average = np.mean(section, axis=(0, 1))
            blurred_im[i:i+n,j:j+n] = average

    return blurred_im




pepperImBlur = generateBlurImage(pepperIm, 4)
lenaImBlur = generateBlurImage(lenaIm, 8)

plt.figure()
plt.suptitle("Normal and Blurred Images")
plt.subplot(2, 2, 1)
plt.imshow(pepperIm)
plt.title("Original Pepper Image")
plt.subplot(2, 2, 2)
plt.imshow(lenaIm, cmap="gray")
plt.title("Original Lena Image")
plt.subplot(2, 2, 3)
plt.imshow(pepperImBlur)
plt.title("Blurred Pepper Image")
plt.subplot(2, 2, 4)
plt.imshow(lenaImBlur, cmap="gray")
plt.title("Blurred Lena Image")
plt.tight_layout()
plt.show()
#display blurred images
            

# if __name__ == "__main__":
#     print("Hello World!")
