import cv2
import numpy as np
import matplotlib.pyplot as plt

def cal_normalized_hsv_hist(im, hBinNum, sBinNum, vBinNum):
    hsv_im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    hist = np.zeros((hBinNum, sBinNum, vBinNum))
    # h has 180 values in opencv, s has 256 values, v has 256 values
    # calculate the size of each bin
    hBinSize = 180 / hBinNum
    sBinSize = 256 / sBinNum
    vBinSize = 256 / vBinNum

    for i in range(im.shape[0]):
        for j in range(im.shape[1]):

            h = int(hsv_im[i, j, 0] / hBinSize)
            s = int(hsv_im[i, j, 1] / sBinSize)
            v = int(hsv_im[i, j, 2] / vBinSize)
            hist[h, s, v] += 1

    total_pixels = im.shape[0] * im.shape[1]
    hist = hist / total_pixels
    hist = hist.flatten()
    return hist

elephant_1 = cv2.imread('Elephant1.jpg')
elephant_2 = cv2.imread('Elephant2.jpg')
horse_1 = cv2.imread('Horse1.jpg')
horse_2 = cv2.imread('Horse2.jpg')

elephant_1_hist = cal_normalized_hsv_hist(elephant_1, 4, 4, 4)
elephant_2_hist = cal_normalized_hsv_hist(elephant_2, 4, 4, 4)
horse_1_hist = cal_normalized_hsv_hist(horse_1, 4, 4, 4)
horse_2_hist = cal_normalized_hsv_hist(horse_2, 4, 4, 4)

def hist_intersection(im1, hist1, im2, hist2):
    # magnitude is equal to number of pixels in corresponding image
    magnitude1 = im1.shape[0] * im1.shape[1]
    magnitude2 = im2.shape[0] * im2.shape[1]
    numerator = np.sum(np.minimum(hist1 * magnitude1, hist2 * magnitude2))
    denominator = np.minimum(magnitude1, magnitude2)
    return numerator / denominator

query_images= [elephant_1, elephant_2, horse_1, horse_2]
query_histograms = [elephant_1_hist, elephant_2_hist, horse_1_hist, horse_2_hist]
query_names = ['Elephant 1', 'Elephant 2', 'Horse 1', 'Horse 2']

fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for i, query in enumerate(query_images):
    hist = query_histograms[i]
    scores = []
    #test all images and store the scores
    index_order = []
    for j, test in enumerate(query_images):
        test_hist = query_histograms[j]
        score = hist_intersection(query, hist, test, test_hist)
        scores.append((score, j))
    #sort the scores
    sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)

    for k, (score, index) in enumerate(sorted_scores):
        ax = axes[k, i]
        ax.imshow(cv2.cvtColor(query_images[index], cv2.COLOR_BGR2RGB))
        ax.set_title(f"Rank {k+1}\nScore: {score:.2f}\n{query_names[index]}")
        ax.axis('off')
plt.tight_layout()
plt.show()
