import cv2
from PIL import Image
import numpy as np

resolution = (1600, 800)
img = np.array(Image.open('maze/16x9.png').convert('RGB'))

img = cv2.resize(img, (1600, 840), interpolation=cv2.INTER_CUBIC)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

cv2.namedWindow('Resized Image', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Resized Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow('Resized Image', img)
cv2.waitKey(2000)
cv2.destroyAllWindows()
# press any key to quit

