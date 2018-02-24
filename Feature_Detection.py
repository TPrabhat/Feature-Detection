# do relevant imports

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import Functions as Func

# load image
img = mpimg.imread('test-image.jpg')
plt.subplot(231)
plt.gca().set_title('Original')
plt.imshow(img)

# convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.subplot(232)
plt.gca().set_title('Grayscale')
plt.imshow(gray)

# find gradient of the grayscale image in x direction
gradx = Func.abs_sobel_thresh(gray, 'x', sobel_kernel = 3, thresh=(10, 100))
plt.subplot(233)
plt.gca().set_title('Sobel X')
plt.imshow(gradx)

# find gradient of grayscale image in y direction
grady = Func.abs_sobel_thresh(gray, 'y', sobel_kernel = 3, thresh=(10, 100))
plt.subplot(234)
plt.gca().set_title('Sobel Y')
plt.imshow(grady)

# find the magnitude of the gradient
mag = Func.mag_thresh(gray, sobel_kernel = 3, mag_thresh = (10, 100))
plt.subplot(235)
plt.gca().set_title('Magnitude')
plt.imshow(mag)

# find the direction of the gradient and apply threshold
dir = Func.dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/16))
plt.subplot(236)
plt.gca().set_title('Direction')
plt.imshow(dir)

plt.savefig('output.jpg')
plt.show()
