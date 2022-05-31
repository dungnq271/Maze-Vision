import matplotlib.pyplot as plt
import numpy as np
from matplotlib import image
from matplotlib.backend_bases import MouseButton
import cv2


def onclick(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))


for size in ['8x6', '16x9', '32x18']:
    img = image.imread(f'maze/{size}.png')
    img = cv2.resize(img, (1216, 830), cv2.INTER_CUBIC)

    fig = plt.figure()
    fig.gca().imshow(img)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
