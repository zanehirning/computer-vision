import matplotlib.pyplot as plt
import numpy as np
import cv2


# Question 1.1
wirebond_im = cv2.imread('Wirebond.tif', cv2.IMREAD_GRAYSCALE)

kernel = np.ones((9, 9), np.uint8)

desired_im_1 = cv2.erode(wirebond_im, kernel, iterations=2)
desired_im_2 = cv2.erode(wirebond_im, kernel, iterations=1)
desired_im_3 = cv2.erode(wirebond_im, kernel, iterations=4)

plt.figure(num="Figure 1")
plt.subplot(1, 4, 1)
plt.imshow(wirebond_im, cmap='gray')
plt.title('Original Image')
plt.subplot(1, 4, 2)
plt.imshow(desired_im_1, cmap='gray')
plt.title('Desired Image 1')
plt.subplot(1, 4, 3)
plt.imshow(desired_im_2, cmap='gray')
plt.title('Desired Image 2')
plt.tight_layout()
plt.subplot(1, 4, 4)
plt.imshow(desired_im_3, cmap='gray')
plt.title('Desired Image 3')
plt.tight_layout()
plt.show()

kernel = np.ones((5, 5), np.uint8)
shapes_im = cv2.imread('Shapes.tif', cv2.IMREAD_GRAYSCALE)

desired_im_1 = cv2.erode(shapes_im, kernel, iterations=5)
desired_im_1 = cv2.dilate(desired_im_1, kernel, iterations=5)
desired_im_2 = cv2.dilate(shapes_im, kernel, iterations=4)
desired_im_2 = cv2.erode(desired_im_2, kernel, iterations=4)
desired_im_3 = cv2.erode(shapes_im, kernel, iterations=5)
desired_im_3 = cv2.dilate(desired_im_3, kernel, iterations=20)
desired_im_3 = cv2.erode(desired_im_3, kernel, iterations=15)

plt.figure(num="Figure 2")
plt.subplot(1, 4, 1)
plt.imshow(shapes_im, cmap='gray')
plt.title('Original Image')
plt.subplot(1, 4, 2)
plt.imshow(desired_im_1, cmap='gray')
plt.title('Desired Image 1')
plt.subplot(1, 4, 3)
plt.imshow(desired_im_2, cmap='gray')
plt.title('Desired Image 2')
plt.tight_layout()
plt.subplot(1, 4, 4)
plt.imshow(desired_im_3, cmap='gray')
plt.title('Desired Image 3')
plt.tight_layout()
plt.show()

# Question 1.2

dowels_im = cv2.imread('Dowels.tif', cv2.IMREAD_GRAYSCALE)

radius = 5
structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ((2*radius), (2*radius)))

oc_dowels = cv2.morphologyEx(dowels_im, cv2.MORPH_OPEN, structuring_element)
oc_dowels = cv2.morphologyEx(oc_dowels, cv2.MORPH_CLOSE, structuring_element)

co_dowels = cv2.morphologyEx(dowels_im, cv2.MORPH_CLOSE, structuring_element)
co_dowels = cv2.morphologyEx(co_dowels, cv2.MORPH_OPEN, structuring_element)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4), num="Figure 3")
plt.subplot(1, 2, 1)
plt.title("Open-Close")
plt.imshow(oc_dowels, cmap='gray')
plt.subplot(1, 2, 2)
plt.title("Close-Open")
plt.imshow(co_dowels, cmap='gray')
plt.tight_layout()
plt.show()

radius = [2, 3, 4, 5]
for r in radius:
    structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ((2*r), (2*r)))
    oc_dowels = cv2.morphologyEx(dowels_im, cv2.MORPH_OPEN, structuring_element)
    oc_dowels = cv2.morphologyEx(oc_dowels, cv2.MORPH_CLOSE, structuring_element)
    co_dowels = cv2.morphologyEx(dowels_im, cv2.MORPH_CLOSE, structuring_element)
    co_dowels = cv2.morphologyEx(co_dowels, cv2.MORPH_OPEN, structuring_element)

plt.figure(figsize=(10, 4), num="Figure 4")
plt.subplot(1, 2, 1)
plt.title("Multiple Open-Close")
plt.imshow(oc_dowels, cmap='gray')
plt.subplot(1, 2, 2)
plt.title("Multiple Close-Open")
plt.imshow(co_dowels, cmap='gray')
plt.tight_layout()
plt.show()

# Question 1.3
squares_im = cv2.imread('SmallSquares.tif', cv2.IMREAD_GRAYSCALE)

# structuring element for pixels that have east and north neighbors
# but have no northwest, west, southwest, south, or southeast neighbors
structuring_element = np.array([[0, 1, 1],
                                [0, 1, 1],
                                [0, 0, 0]], dtype=np.uint8)

# apply the structuring element to the image
east_north_squares = cv2.erode(squares_im, structuring_element, iterations=1)

plt.figure(num="Figure 5")
plt.title("East and North Neighbors")
plt.imshow(east_north_squares, cmap='gray')
plt.show()

# count the number of pixels that have east and north neighbors
num_pixels = np.count_nonzero(east_north_squares)
print(f"Number of pixels with east and north neighbors: {num_pixels}")

# Question 2.1

ball_im = cv2.imread('Ball.tif', cv2.IMREAD_GRAYSCALE)

def find_component_labels(img, se):
    labelImg = np.zeros(img.shape, dtype=np.uint8)
    img_copy = img.copy()
    num = 0
    for x in range(img_copy.shape[0]):
        for y in range(img_copy.shape[1]):
            # check if it is a new connected component
            if img_copy[x, y] == 255 and labelImg[x, y] == 0:
                #update label
                num += 1
                #create a stack to keep track of pixels to visit
                stack = [(x, y)]
                while stack:
                    cur_x, cur_y = stack.pop()
                    labelImg[cur_x, cur_y] = num
                    # remove the pixel from the image so it is not re-visited
                    img_copy[cur_x, cur_y] = 0
                    # check the neighbors of the current pixel
                    for dx in range(-se.shape[0]//2+1, se.shape[0]//2 + 1):
                        for dy in range(-se.shape[1]//2+1, se.shape[1]//2 + 1):
                            # check if the neighbor is in the structuring element 
                            # This should be every element in this case, because all elements in se are 1
                            if se[dx + se.shape[0]//2, dy + se.shape[1]//2] == 1:
                                # Update our position by adding the original pixel we were looking at, plus the one that is its neighbor
                                new_x = cur_x + dx
                                new_y = cur_y + dy
                                # Ensure that the new position is within the image
                                # Ensure that the new position is a white pixel and has not been labeled yet
                                if 0 <= new_x < img_copy.shape[0] and 0 <= new_y < img_copy.shape[1] and img_copy[new_x, new_y] == 255 and labelImg[new_x, new_y] == 0:
                                    stack.append((new_x, new_y))

    return labelImg, num

se = np.ones((3, 3), dtype=np.uint8)
#it is slow, but it works
labelImg, number_components = find_component_labels(ball_im, se)

print(f"Number of components: {number_components}")

plt.figure(num="Figure 6")
plt.title("Components Labeled")
plt.imshow(labelImg, cmap='gray')
plt.show()

# Question 2.2
#inbuilt function for find_component_labels
number_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ball_im, connectivity=8)
num_connected_particles = number_labels - 1
print(f"Number of components: {num_connected_particles}")

labeled_im = np.uint8(labels)

plt.figure(figsize=(6, 6), num="Figure 7")
plt.imshow(cv2.cvtColor(labeled_im, cv2.COLOR_BGR2RGB))
plt.title("Connected Particles")
plt.show()

# Question 2.3
# find connected particals that reside on the border
se = np.ones((3, 3), dtype=np.uint8)

def find_component_labels_border(img, se):
    labelImg = np.zeros(img.shape, dtype=np.uint8)
    img_copy = img.copy()
    num = 0
    for x in range(img_copy.shape[0]):
        for y in range(img_copy.shape[1]):
            # check if it is a new connected component
            if (x == 0 or x == img_copy.shape[0] - 1) or (y == 0 or y == img_copy.shape[1] - 1):
                if img_copy[x, y] == 255 and labelImg[x, y] == 0:
                    #update label
                    num += 1
                    #create a stack to keep track of pixels to visit
                    stack = [(x, y)]
                    while stack:
                        cur_x, cur_y = stack.pop()
                        labelImg[cur_x, cur_y] = num
                        # remove the pixel from the image so it is not re-visited
                        img_copy[cur_x, cur_y] = 0
                        # check the neighbors of the current pixel
                        for dx in range(-se.shape[0]//2+1, se.shape[0]//2 + 1):
                            for dy in range(-se.shape[1]//2+1, se.shape[1]//2 + 1):
                                # check if the neighbor is in the structuring element 
                                # This should be every element in this case, because all elements in se are 1
                                if se[dx + se.shape[0]//2, dy + se.shape[1]//2] == 1:
                                    # Update our position by adding the original pixel we were looking at, plus the one that is its neighbor
                                    new_x = cur_x + dx
                                    new_y = cur_y + dy
                                    # Ensure that the new position is within the image
                                    # Ensure that the new position is a white pixel and has not been labeled yet
                                    if 0 <= new_x < img_copy.shape[0] and 0 <= new_y < img_copy.shape[1] and img_copy[new_x, new_y] == 255 and labelImg[new_x, new_y] == 0:
                                        stack.append((new_x, new_y))

    return labelImg, num

border_labeled_im, num_border_particles = find_component_labels_border(ball_im, se)
print(f"Number of border components: {num_border_particles}")
border_ball_im = np.where(border_labeled_im > 0, 255, 0)

plt.figure(num="Figure 8")
plt.title("Border Particles")
plt.imshow(border_ball_im, cmap='gray')
plt.show()

# Question 2.4
connected_particles_not_on_border = np.where((labelImg > 0) & (border_labeled_im == 0), 255, 0)
labeled_not_on_border, not_on_border_numbered = find_component_labels(connected_particles_not_on_border, se)

print(f"Number of connected particles not on border: {not_on_border_numbered}")

#estimate particle size
def estimate_particle_size(img, num_particles):
    particle_size = []
    for i in range(1, num_particles + 1):
        particle_size.append(np.count_nonzero(img == i))
    # the particle size in pixels is the minimum of the particle size
    print(particle_size)
    return np.min(particle_size)

def remove_individual(label_im, num_particles, individual_size):
    for i in range(1, num_particles + 1):
        if np.count_nonzero(label_im == i) <= individual_size:
            label_im = np.where(label_im == i, 0, label_im)
    return label_im

particle_size = estimate_particle_size(labeled_not_on_border, not_on_border_numbered) + 20 #slight offset
overlapping_particles = remove_individual(labeled_not_on_border, not_on_border_numbered, particle_size)

plt.figure(num="Figure 9")
plt.subplot(1, 2, 1)
plt.title("Connected Particles Not on Border")
plt.imshow(labeled_not_on_border, cmap='gray')
plt.subplot(1, 2, 2)
plt.title("Overlapping Particles")
plt.imshow(overlapping_particles, cmap='gray')
plt.show()

# Question 2.5
def remove_non_partial(label_im, num_particles, individual_size):
    for i in range(1, num_particles + 1):
        if np.count_nonzero(label_im == i) >= individual_size:
            label_im = np.where(label_im == i, 0, label_im)
    return label_im

#remove overlapping border particles

partial_particles = remove_non_partial(border_labeled_im, num_border_particles, particle_size)

plt.figure(num="Figure 10")
plt.title("Partial Particles on Border")
plt.imshow(partial_particles, cmap='gray')
plt.show()