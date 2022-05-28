import cv2
from PIL import Image
import numpy as np


def show_maze(img_path):
    img = np.array(Image.open(img_path).convert('RGB'))

    cv2.namedWindow('Resized Image', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Resized Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Resized Image', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# press any key to quit


if __name__ == '__main__':
    path = '../maze/16x9.png'
    show_maze(path)
