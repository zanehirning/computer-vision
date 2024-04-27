import matplotlib.pyplot as plt
import numpy as np
import cv2
import pywt # package for wavelet transform
import copy


# Question 1
def gaussian_low_pass(img, std1, std2):
    # Get the middle pixel
    u, v = img.shape[0]//2, img.shape[1]//2
    gaussian_filter = np.zeros(img.shape)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            gaussian_filter[x, y] = np.exp(-(((x-u)**2/(2*std1**2)) + ((y-v)**2/(2*std2**2))))
    #Normalize the filter
    gaussian_filter = gaussian_filter/np.sum(gaussian_filter)
    return gaussian_filter

def apply_filter(img, filter):
    # Apply the filter
    img_fft = np.fft.fft2(img)
    img_fft_shift = np.fft.fftshift(img_fft)
    img_fft_shift_filtered = img_fft_shift * filter
    img_fft_filtered = np.fft.ifftshift(img_fft_shift_filtered)
    img_filtered = np.fft.ifft2(img_fft_filtered)
    return img_filtered


sample_im = cv2.imread('sample.jpg', cv2.IMREAD_GRAYSCALE)
std1, std2 = 20, 70

gaussian_filter = gaussian_low_pass(sample_im, std1, std2)
sample_im_lp = apply_filter(sample_im, gaussian_filter)

plt.figure(num="Figure 1")
plt.subplot(1, 3, 1)
plt.imshow(sample_im, cmap='gray')
plt.title('Original Image')
plt.subplot(1, 3, 2)
plt.imshow(gaussian_filter, cmap='gray')
plt.title('Gaussian Low Pass Filter')
plt.subplot(1, 3, 3)
plt.imshow(np.abs(sample_im_lp), cmap='gray')
plt.title('Filtered Image')
plt.show()

# Question 2
def butterworth_high_pass(img, cutoff, order):
    u, v = img.shape[0]//2, img.shape[1]//2
    butterworth_filter = np.zeros(img.shape)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            distance = np.sqrt((x-u)**2+(y-v)**2)
            butterworth_filter[x, y] = (1/(1+(distance/cutoff)**(2*order)))
    butterworth_filter = 1 - butterworth_filter
    return butterworth_filter

butterworth_filter = butterworth_high_pass(sample_im, 50, 2)
sample_im_hp = apply_filter(sample_im, butterworth_filter)
plt.figure(num="Figure 2")
plt.subplot(1, 3, 1)
plt.imshow(sample_im, cmap='gray')
plt.title('Original Image')
plt.subplot(1, 3, 2)
plt.imshow(butterworth_filter, cmap='gray')
plt.title('Butterworth High Pass Filter')
plt.subplot(1, 3, 3)
plt.imshow(np.abs(sample_im_hp), cmap='gray')
plt.title('Filtered Image')
plt.show()

capital_im = cv2.imread('Capitol.jpg', cv2.IMREAD_GRAYSCALE)

sample_fft = np.fft.fft2(sample_im)
capital_fft = np.fft.fft2(capital_im)
sample_fft = np.fft.fftshift(sample_fft)
capital_fft = np.fft.fftshift(capital_fft)

# Calculate the magnitude and phase of the two images
# Must take log of magnitude for display purposes. Add 1 to avoid log(0)
# For the phase, use np.angle(). https://numpy.org/doc/stable/reference/generated/numpy.angle.html
sample_mag = np.log(np.abs(sample_fft)+1)
capital_mag = np.log(np.abs(capital_fft)+1)

sample_phase = np.angle(sample_fft)
capital_phase = np.angle(capital_fft)

# display the magnitude and phase of the two images
plt.figure(num="Figure 3")
plt.subplot(2, 2, 1)
plt.imshow(sample_mag, cmap='gray')
plt.title('Sample Magnitude')
plt.subplot(2, 2, 2)
plt.imshow(sample_phase, cmap='gray')
plt.title('Sample Phase')
plt.subplot(2, 2, 3)
plt.imshow(capital_mag, cmap='gray')
plt.title('Capital Magnitude')
plt.subplot(2, 2, 4)
plt.imshow(capital_phase, cmap='gray')
plt.title('Capital Phase')
plt.tight_layout()
plt.show()

# inverse the log of magnitude and multiply it by the phase. 1j is the complex number i
swapped_sample_fft = np.exp(sample_mag) * np.exp(1j*capital_phase)
swapped_capital_fft = np.exp(capital_mag) * np.exp(1j*sample_phase)

swapped_sample_image = np.abs(np.fft.ifft2(np.fft.ifftshift(swapped_sample_fft))).astype(np.uint8)
swapped_capital_image = np.abs(np.fft.ifft2(np.fft.ifftshift(swapped_capital_fft))).astype(np.uint8)

plt.figure(num="Figure 4")
plt.subplot(1, 2, 1)
plt.imshow(swapped_sample_image, cmap='gray')
plt.title('Swapped Sample Image')
plt.subplot(1, 2, 2)
plt.imshow(swapped_capital_image, cmap='gray')
plt.title('Swapped Capital Image')
plt.tight_layout()
plt.show()

# Question 3
video = cv2.VideoCapture('boy_noisy.gif')
_, boy_im = video.read()
video.release()
boy_im = cv2.cvtColor(boy_im, cv2.COLOR_BGR2GRAY)

def largest_magnitude_locations(magnitude_img, num):

    locations = []
    for i in range(magnitude_img.shape[0]):
        for j in range(magnitude_img.shape[1]):
            locations.append((i, j, magnitude_img[i, j]))

    # sort locations based on magnitude, reverse order to get largest first
    # locations contains center, so skip first one. 
    # Each location will appear twice in the image, so we need to get 2*num locations
    locations = sorted(locations, reverse=True, key=lambda x: x[2])[1:(2*num)+1]
    return locations

def replace_neighbors_with_average(img, locations):
    # add padding to image
    padded_img = np.pad(img, 1, mode='constant')
    padded_img_mag = np.abs(padded_img)
    img_copy = img.copy()
    img_mag = np.abs(img_copy)
    img_phase = np.angle(img_copy)
    for location in locations:
        x, y = location[0]+1, location[1]+1 # add 1 to account for padding
        neighbor_magnitudes = [
            padded_img_mag[x-1, y-1],
            padded_img_mag[x-1, y],
            padded_img_mag[x-1, y+1],
            padded_img_mag[x, y-1],
            padded_img_mag[x, y+1],
            padded_img_mag[x+1, y-1],
            padded_img_mag[x+1, y],
            padded_img_mag[x+1, y+1]
        ]
        average_mag = np.mean(neighbor_magnitudes)
        img_mag[x-1, y-1] = average_mag
    #reconstruct image
    return img_mag * np.exp(1j*img_phase)


boy_dft = np.fft.fftshift(np.fft.fft2(boy_im))
boy_mag = np.abs(boy_dft)
boy_phase = np.angle(boy_dft)
# 4 largest locations
boy_largest_locations = largest_magnitude_locations(boy_mag, 4)

# change location with average of its neighbors
average_boy_dft = replace_neighbors_with_average(boy_dft, boy_largest_locations)

inverse_boy_im = np.abs(np.fft.ifft2(np.fft.ifftshift(average_boy_dft)))

plt.figure(num="Figure 5")
plt.subplot(1, 2, 1)
plt.imshow(boy_im, cmap='gray')
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(inverse_boy_im, cmap='gray')
plt.title('Modified Image')
plt.tight_layout()
plt.show()

# replacing different number of locations
boy_largest_locations_2 = largest_magnitude_locations(boy_mag, 2)
average_boy_dft_2 = replace_neighbors_with_average(boy_dft, boy_largest_locations_2)
inverse_boy_im_2 = np.abs(np.fft.ifft2(np.fft.ifftshift(average_boy_dft_2)))

boy_largest_locations_3 = largest_magnitude_locations(boy_mag, 3)
average_boy_dft_3 = replace_neighbors_with_average(boy_dft, boy_largest_locations_3)
inverse_boy_im_3 = np.abs(np.fft.ifft2(np.fft.ifftshift(average_boy_dft_3)))

boy_largest_locations_5 = largest_magnitude_locations(boy_mag, 5)
average_boy_dft_5 = replace_neighbors_with_average(boy_dft, boy_largest_locations_5)
inverse_boy_im_5 = np.abs(np.fft.ifft2(np.fft.ifftshift(average_boy_dft_5)))

boy_largest_locations_6 = largest_magnitude_locations(boy_mag, 6)
average_boy_dft_6 = replace_neighbors_with_average(boy_dft, boy_largest_locations_6)
inverse_boy_im_6 = np.abs(np.fft.ifft2(np.fft.ifftshift(average_boy_dft_6)))

plt.figure(num="Figure 6")
plt.subplot(2, 2, 1)
plt.imshow(inverse_boy_im_2, cmap='gray')
plt.title('Modified Image 2 Locations')
plt.subplot(2, 2, 2)
plt.imshow(inverse_boy_im_3, cmap='gray')
plt.title('Modified Image 3 Locations')
plt.subplot(2, 2, 3)
plt.imshow(inverse_boy_im_5, cmap='gray')
plt.title('Modified Image 5 Locations')
plt.subplot(2, 2, 4)
plt.imshow(inverse_boy_im_6, cmap='gray')
plt.title('Modified Image 6 Locations')
plt.tight_layout()
plt.show()

# Question 4
lena_im = cv2.imread('Lena.jpg', cv2.IMREAD_GRAYSCALE)

#https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html
pywt_mode = 'periodic'

max_level = pywt.dwt_max_level(lena_im.shape[0], 'db2')
coefficients = pywt.wavedec2(lena_im, 'db2', level=max_level)
reconstructed_lena_im = pywt.waverec2(coefficients, 'db2')
# use np.allclose because rounding errors. 
if np.allclose(lena_im, reconstructed_lena_im):
    print('The original and restored images are the same')
else:
    print('The original and restored images are different')

lena_decomp_three = pywt.wavedec2(lena_im, 'db2', level=3)
lena_approx, lena_details_3, lena_details_2, lena_details_1 = lena_decomp_three

def set_subband_to_zero(level, subband):
    #level is decomp level details
    # subband 0 is horizontal, 1 is vertical, 2 is diagonal
    for x in range(level[subband].shape[0]):
        for y in range(level[subband].shape[1]):
            level[subband][x][y] = 0
    return level

# Set the 16 values of each 4×4 non-overlapping block in the approximation subband as its average
for x in range(0, lena_approx.shape[0], 4):
    for y in range(0, lena_approx.shape[1], 4):
        lena_approx[x:x+4, y:y+4] = np.mean(lena_approx[x:x+4, y:y+4])
# Independently reconstruct image one layer at a time
lena_approx_reconstructed = pywt.waverec2((lena_approx, lena_details_3, lena_details_2, lena_details_1), 'db2')
plt.figure(num="Figure 7")
plt.imshow(lena_approx_reconstructed, cmap='gray')
plt.title('Reconstructed Approximation Subband')
plt.show()

# Set the first level horizontal detail coefficients as 0’s.
lena_details_1 = set_subband_to_zero(lena_details_1, 0)
lena_details_1_reconstructed = pywt.waverec2((lena_approx, lena_details_3, lena_details_2, lena_details_1), 'db2')
plt.figure(num="Figure 8")
plt.imshow(lena_details_1_reconstructed, cmap='gray')
plt.title('Reconstructed First Level Horizontal Detail Coefficients')
plt.show()

# Set the second level diagonal detail coefficients as 0’s.
lena_details_2 = set_subband_to_zero(lena_details_2, 2)
lena_details_2_reconstructed = pywt.waverec2((lena_approx, lena_details_3, lena_details_2, lena_details_1), 'db2')
plt.figure(num="Figure 9")
plt.imshow(lena_details_2_reconstructed, cmap='gray')
plt.title('Reconstructed Second Level Diagonal Detail Coefficients')
plt.show()

# Set the third level vertical detail coefficients as 0’s.
lena_details_3 = set_subband_to_zero(lena_details_3, 1)
lena_details_3_reconstructed = pywt.waverec2((lena_approx, lena_details_3, lena_details_2, lena_details_1), 'db2')
plt.figure(num="Figure 10")
plt.imshow(lena_details_3_reconstructed, cmap='gray')
plt.title('Reconstructed Third Level Vertical Detail Coefficients')
plt.show()

print(
    """Truthfully, I could not truly tell much of a difference between the images. 
    However, the first operation was done on the approximation subband, this does not affect any of the vertical, horizontal, or diagonal details.
    The second operation was done on the first level horizontal detail coefficients, this only affects the horizontal details. Setting them to zero removed some horizontal details.
    The third operation was done on the second level diagonal detail coefficients, this only affects the diagonal details. Setting them to zero removed some diagonal details.
    The fourth operation was done on the third level vertical detail coefficients, this only affects the vertical details. Setting them to zero removed some vertical details.
    The fourth operation also affected the most details, so it makes sense that it was the most noticeable. Whereas the other operations affected much fewer details.
    """
)

# Question 5
lena_max = np.max(lena_im)
lena_im_n = lena_im/lena_max

row, col = lena_im_n.shape
mean = 0
var = 0.01
sigma = var**0.5
noise = np.random.normal(mean, sigma, (row, col))
lena_noise = cv2.add(lena_im_n, noise)
lena_noise = lena_noise*lena_max

cv2.imwrite("NoisyLena.bmp", lena_noise)

noisy_lena = cv2.imread('NoisyLena.bmp', cv2.IMREAD_GRAYSCALE)

noisy_lena_decomp = pywt.wavedec2(noisy_lena, 'db2', level=3, mode=pywt_mode)

noisy_lena_approx, noisy_lena_details_3, noisy_lena_details_2, noisy_lena_details_1 = noisy_lena_decomp

# Compute adaptive threshold t for 1st level wavelet subband
def adaptive_threshold(level, subbands):
    # t = std*sqrt(2*log(n))
    #level is decomp level details
    # subbands is a list of all subbands to be used
    all_subband_coeffs = np.concatenate([level[subband] for subband in subbands])
    std = np.median(np.abs(all_subband_coeffs))/.6745
    M = all_subband_coeffs.shape[0]*all_subband_coeffs.shape[1]
    t = std*np.sqrt(2*np.log(M))
    return t

# Apply soft thresholding to 1st level wavelet subband
def soft_threshold(value, t):
    if value >= t:
        return value - t
    elif value <= -t:
        return value + t
    else:
        return 0
    
# Modify the wavelet coefficients using soft thresholding
def modify_coefficients(level, subbands):
    #level is decomp level details
    # subband 0 is horizontal, 1 is vertical, 2 is diagonal, subbands is a list of subbands
    level_copy = copy.deepcopy(level)
    t = adaptive_threshold(level_copy, subbands)
    for subband in range(len(level_copy)):
        for x in range(level_copy[subband].shape[0]):
            for y in range(level_copy[subband].shape[1]):
                level_copy[subband][x][y] = soft_threshold(level_copy[subband][x][y], t)
    return level_copy

noisy_lena_details_1_modified = modify_coefficients(noisy_lena_details_1, [2])
noisy_lena_details_2_modified = modify_coefficients(noisy_lena_details_2, [2])
noisy_lena_details_3_modified = modify_coefficients(noisy_lena_details_3, [2])

# Reconstruct the image using the modified coefficients
noisy_lena_reconstructed = pywt.waverec2((noisy_lena_approx, noisy_lena_details_3_modified, noisy_lena_details_2_modified, noisy_lena_details_1_modified), 'db2')

noisy_lena_details_1_modified_2 = modify_coefficients(noisy_lena_details_1, [0, 1, 2])
noisy_lena_details_2_modified_2 = modify_coefficients(noisy_lena_details_2, [0, 1, 2])
noisy_lena_details_3_modified_2 = modify_coefficients(noisy_lena_details_3, [0, 1, 2])

noisy_lena_reconstructed_2 = pywt.waverec2((noisy_lena_approx, noisy_lena_details_3_modified_2, noisy_lena_details_2_modified_2, noisy_lena_details_1_modified_2), 'db2')

plt.figure(num="Figure 11")
plt.subplot(1,3,1)
plt.imshow(noisy_lena, cmap='gray')
plt.title('Noisy Lena')
plt.subplot(1,3,2)
plt.imshow(noisy_lena_reconstructed, cmap='gray')
plt.title('Denoised Lena First Method')
plt.subplot(1,3,3)
plt.imshow(noisy_lena_reconstructed_2, cmap='gray')
plt.title('Denoised Lena Second Method')
plt.tight_layout()
plt.show()

noisy_lena = noisy_lena.astype(np.float64)

first_method_psnr = cv2.PSNR(noisy_lena, noisy_lena_reconstructed)
second_method_psnr = cv2.PSNR(noisy_lena, noisy_lena_reconstructed_2)

print("First Method PSNR: ", first_method_psnr)
print("Second Method PSNR: ", second_method_psnr)
print("First Method PSNR is better than Second Method PSNR: ", first_method_psnr > second_method_psnr)
print("From these psnr values, we can see that the first method is *technically* better than the second method. However, the difference is so small that it is not noticeable to the human eye. At least, I was not able to tell the difference between the two")