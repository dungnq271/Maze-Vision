import cv2
from PIL import Image
import numpy as np

resolution = (1600, 800)
img = np.array(Image.open('maze/16x9.png').convert('RGB'))
w, h, c = img.shape

# cv2.namedWindow("Output")
# cv2.moveWindow("Output", 40,30)  # Move it to (40,30)
# cv2.resizeWindow("Output", resolution[0], resolution[1])

new_w = int(w * 1.5)
new_h = int(h * 1.5)

# cv2.imshow('Original Image', img)
# cv2.waitKey(0)
# dim = (img.shape[1] * 5, img.shape[0] * 5)
img = cv2.resize(img, (1600, 800), interpolation=cv2.INTER_CUBIC)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = cv2.copyMakeBorder(img, int((800 - new_h) / 2), int((800 - new_h) / 2), int((1600 - new_w) / 2),
#                          int((1600 - new_h) / 2), 0)

cv2.imshow('Resized Image', img)
cv2.waitKey(0)
# cv2.imwrite('dung.jpg', img)
