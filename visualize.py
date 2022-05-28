import matplotlib.pyplot as plt
import numpy as np
from matplotlib import image
from matplotlib.backend_bases import MouseButton
import cv2

img = image.imread('maze/8x6.png')
img = cv2.resize(img, (1600, 840), cv2.INTER_CUBIC)


def onclick(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))


fig = plt.figure()
fig.gca().imshow(img)

cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
