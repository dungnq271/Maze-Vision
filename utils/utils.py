from PIL import Image
import cv2
import numpy as np


def display_agent(img, agent, x, y, w, h):
    img_w = img.shape[0]
    img_h = img.shape[1]
    # left = x - w // 2 if x - w // 2 >= 0 else w // 2 + x
    # right = x + w // 2 if x + w // 2 <= img_w else img_w
    # upper = y - h // 2 if y - h // 2 >= 0 else 0
    # below = y + h // 2 if y + h // 2 <= img_h else img_h
    img[y, w] = agent
    return img


def load_image(path, w, h):
    image = np.array(Image.open(path).convert('RGB'))
    image = cv2.resize(image, (w, h), cv2.INTER_CUBIC)
    return image
