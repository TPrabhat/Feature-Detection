import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import Functions as Func


img = mpimg.imread('test-image.jpg')
plt.subplot(231)
plt.gca().set_title('Original')
plt.imshow(img)

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.subplot(232)
plt.gca().set_title('Grayscale')
plt.imshow(gray)

gradx = Func.abs_sobel_thresh(gray, 'x', sobel_kernel = 3, thresh=(10, 100))
plt.subplot(233)
plt.gca().set_title('Sobel X')
plt.imshow(gradx)

grady = Func.abs_sobel_thresh(gray, 'y', sobel_kernel = 3, thresh=(10, 100))
plt.subplot(234)
plt.gca().set_title('Sobel Y')
plt.imshow(grady)

mag = Func.mag_thresh(gray, sobel_kernel = 3, mag_thresh = (10, 100))
plt.subplot(235)
plt.gca().set_title('Magnitude')
plt.imshow(mag)

dir = Func.dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/16))
plt.subplot(236)
plt.gca().set_title('Direction')
plt.imshow(dir)

plt.show()


