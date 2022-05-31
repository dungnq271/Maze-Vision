import numpy as np


def check_wall(x, y, image):
    if np.linalg.norm(image[y, x, :] - np.array([255, 255, 255])) < 1e-9:
        return True
    return False


def check_entrance(x, y, r, img=None):
    if img == '8x6':
        if 618-r < x < 789+r and r+2 < y < r+10:
            return True
    if img == '16x9':
        if 706-r < x < 791+r and r+1 < y < r+10:
            return True
    if img == '32x18':
        if 753-r < x < 794+r and r+2 < y < r+10:
            return True
    return False


def check_destination(x, y, r, img=None):
    if img == '8x6':
        if 812-r < x < 987+r and 832-r < y < 840:
            return True
    if img == '16x9':
        if 807-r < x < 894+r and 832-r < y < 840:
            return True
    if img == '32x18':
        if 802-r < x < 845+r and 834-r < y < 840:
            return True
    return False

