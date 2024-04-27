import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt

# Question 1.1
ball_im = cv2.imread('ball.bmp')

ball_hsv = cv2.cvtColor(ball_im, cv2.COLOR_BGR2HSV)

# separate orange ball from background
# https://en.wikipedia.org/wiki/HSL_and_HSV
lower_orange = np.array([0, 40, 40])
upper_orange = np.array([40, 255, 255])

ball = cv2.inRange(ball_hsv, lower_orange, upper_orange)
kernel = np.ones((5, 5), np.uint8)
ball = cv2.morphologyEx(ball, cv2.MORPH_CLOSE, kernel)

# make the countour smoother by blurring and thresholding
blurred_ball = cv2.GaussianBlur(ball, (0, 0), 3)
_, thresh = cv2.threshold(blurred_ball, 200, 255, cv2.THRESH_BINARY)

# find contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

ball_result = ball_im.copy()

if len(contours)>0:
    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    if M['m00'] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv2.drawContours(ball_result, [largest_contour], -1, (0, 0, 0), 2)
        cv2.line(ball_result, (cx - 10, cy), (cx + 10, cy), (255, 0, 0), 2)
        cv2.line(ball_result, (cx, cy - 10), (cx, cy + 10), (255, 0, 0), 2)

ball_result = cv2.cvtColor(ball_result, cv2.COLOR_BGR2RGB)

plt.figure()
plt.imshow(ball)
plt.show()

plt.figure()
plt.imshow(ball_result)
plt.show()

# Question 1.2
#remove the shadow from the ball
lower_shadow = np.array([50, 20, 0])
upper_shadow = np.array([110, 255, 100])

shadow = cv2.inRange(ball_hsv, lower_shadow, upper_shadow)
shadow = cv2.morphologyEx(shadow, cv2.MORPH_CLOSE, kernel)

blurred_shadow = cv2.GaussianBlur(shadow, (0, 0), 3)
_, thresh = cv2.threshold(blurred_shadow, 100, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

shadow_result = ball_im.copy()

shadow_largest = max(contours, key=cv2.contourArea)
shadow_filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > 6000] #magic number

cv2.drawContours(shadow_result, shadow_filtered_contours, -1, (0, 0, 255), 1)

plt.figure()
plt.imshow(shadow)
plt.show()

plt.figure()
shadow_result = cv2.cvtColor(shadow_result, cv2.COLOR_BGR2RGB)
plt.imshow(shadow_result)
plt.show()

# Question 2.1
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

plt.figure()
plt.subplot(2, 2, 1)
plt.bar(range(len(elephant_1_hist)), elephant_1_hist)
plt.title('Elephant 1')
plt.subplot(2, 2, 2)
plt.bar(range(len(elephant_2_hist)), elephant_2_hist)
plt.title('Elephant 2')
plt.subplot(2, 2, 3)
plt.bar(range(len(horse_1_hist)), horse_1_hist)
plt.title('Horse 1')
plt.subplot(2, 2, 4)
plt.bar(range(len(horse_2_hist)), horse_2_hist)
plt.title('Horse 2')
plt.tight_layout()
plt.show()

# Question 2.2
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

# Question 3.1

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt

lena_im = cv2.imread('Lena.jpg', cv2.IMREAD_GRAYSCALE)
lena_approx, lena_details_3, lena_details_2, lena_details_1 = pywt.wavedec2(lena_im, 'db9', level=3, mode='periodization')
np.random.seed(20)
b = np.random.randint(0, 2, lena_approx.shape)
beta = 30
for i in range(lena_approx.shape[0]):
    for j in range(lena_approx.shape[1]):
        if b[i, j] == 1 and (lena_approx[i, j]%beta) >= .25*beta:
            lena_approx[i, j] = (lena_approx[i, j] - (lena_approx[i, j]%beta)) + (.75*beta)
        elif b[i, j] == 1 and (lena_approx[i, j]%beta) < .25*beta:
            lena_approx[i, j] = ((lena_approx[i, j]-.25*beta) - ((lena_approx[i, j]-.25*beta)%beta)) + (.75*beta)
        elif b[i, j] == 0 and (lena_approx[i, j]%beta) <= .75*beta:
            lena_approx[i, j] = (lena_approx[i, j] - (lena_approx[i, j]%beta)) + (.25*beta)
        elif b[i, j] == 0 and (lena_approx[i, j]%beta) > .75*beta:
            lena_approx[i, j] = ((lena_approx[i, j]+.5*beta) - ((lena_approx[i, j]-.5*beta)%beta)) + (.25 * beta)

watermarked_lena = pywt.waverec2([lena_approx, lena_details_3, lena_details_2, lena_details_1], 'db9', mode='periodization')

watermarked_lena = np.clip(watermarked_lena, 0, 255).astype(np.uint8)

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(lena_im, cmap='gray')
plt.title('Original Lena')
plt.subplot(1, 3, 2)
plt.imshow(watermarked_lena, cmap='gray')
plt.title('Watermarked Lena')
plt.subplot(1, 3, 3)
plt.imshow(lena_im-watermarked_lena, cmap='gray')
plt.title('Difference')
plt.tight_layout()
plt.show()

new_b = np.zeros(b.shape, dtype=int)
lena_approx_2, _, _, _ = pywt.wavedec2(watermarked_lena, 'db9', level=3, mode='periodization')
for i in range(lena_approx_2.shape[0]):
    for j in range(lena_approx_2.shape[1]):
        if (lena_approx_2[i, j]%beta) > beta/2:
            new_b[i, j] = 1
        else:
            new_b[i, j] = 0

matching_bits = np.sum(b == new_b)
percentage_matching = (matching_bits / b.size) * 100

print(f"Percentage of matching bits: {percentage_matching}%")
if np.array_equal(b, new_b):
    print("The watermarked image is the same as the original image")
else:
    print("The watermarked image is not the same as the original image")

# Question 3.3

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt

lena_im = cv2.imread('Lena.jpg', cv2.IMREAD_GRAYSCALE)
lena_approx, lena_details_3, lena_details_2, lena_details_1 = pywt.wavedec2(lena_im, 'db9', level=3, mode='periodization')
np.random.seed(20)
b = np.random.randint(0, 2, lena_approx.shape)
beta = 60
for i in range(lena_approx.shape[0]):
    for j in range(lena_approx.shape[1]):
        if b[i, j] == 1 and (lena_approx[i, j]%beta) >= .25*beta:
            lena_approx[i, j] = (lena_approx[i, j] - (lena_approx[i, j]%beta)) + (.75*beta)
        elif b[i, j] == 1 and (lena_approx[i, j]%beta) < .25*beta:
            lena_approx[i, j] = ((lena_approx[i, j]-.25*beta) - ((lena_approx[i, j]-.25*beta)%beta)) + (.75*beta)
        elif b[i, j] == 0 and (lena_approx[i, j]%beta) <= .75*beta:
            lena_approx[i, j] = (lena_approx[i, j] - (lena_approx[i, j]%beta)) + (.25*beta)
        elif b[i, j] == 0 and (lena_approx[i, j]%beta) > .75*beta:
            lena_approx[i, j] = ((lena_approx[i, j]+.5*beta) - ((lena_approx[i, j]-.5*beta)%beta)) + (.25 * beta)

watermarked_lena = pywt.waverec2([lena_approx, lena_details_3, lena_details_2, lena_details_1], 'db9', mode='periodization')

watermarked_lena = np.clip(watermarked_lena, 0, 255).astype(np.uint8)

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(lena_im, cmap='gray')
plt.title('Original Lena')
plt.subplot(1, 3, 2)
plt.imshow(watermarked_lena, cmap='gray')
plt.title('Watermarked Lena')
plt.subplot(1, 3, 3)
plt.imshow(lena_im-watermarked_lena, cmap='gray')
plt.title('Difference')
plt.tight_layout()
plt.show()

new_b = np.zeros(b.shape, dtype=int)
lena_approx_2, _, _, _ = pywt.wavedec2(watermarked_lena, 'db9', level=3, mode='periodization')
for i in range(lena_approx_2.shape[0]):
    for j in range(lena_approx_2.shape[1]):
        if (lena_approx_2[i, j]%beta) > beta/2:
            new_b[i, j] = 1
        else:
            new_b[i, j] = 0

matching_bits = np.sum(b == new_b)
percentage_matching = (matching_bits / b.size) * 100

print(f"Percentage of matching bits: {percentage_matching}%")
if np.array_equal(b, new_b):
    print("The watermarked image is the same as the original image")
else:
    print("The watermarked image is not the same as the original image")