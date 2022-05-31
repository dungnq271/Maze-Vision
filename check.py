import numpy as np


def check_wall(x, y, image):
    if np.linalg.norm(image[y, x, :] - np.array([255, 255, 255])) < 1e-9:
        return True
    return False


def check_entrance(x, y, r, img=None):
    if img == '8x6':
        if 470-r < x < 600+r and r+1 < y < r+10:
            return True
    if img == '16x9':
        if 536-r < x < 601+r and r+2 < y < r+10:
            return True
    if img == '32x18':
        if 572-r < x < 605+r and r+1 < y < r+10:
            return True
    return False


def check_destination(x, y, r, img=None):
    if img == '8x6':
        if 617-r < x < 748+r and 822-r < y < 830:
            return True
    if img == '16x9':
        if 613-r < x < 679+r and 826-r < y < 830:
            return True
    if img == '32x18':
        if 610-r < x < 642+r and 826-r < y < 830:
            return True
    return False

