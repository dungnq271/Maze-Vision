import numpy as np
# import time
import cv2


def check_wall(x, y, image):
    if np.linalg.norm(image[y, x, :] - np.array([255, 255, 255])) < 1e-9:
        return True
    return False


def check_destination(x, y, img=None):
    if img == '8x6':
        if 613 < x < 789 and 0 < y < 10:
            return True
    if img == '16x9':
        if 709 < x < 789 and 0 < y < 10:
            return True
    if img == '32x18':
        if 755 < x < 791 and 0 < y < 6:
            return True
    return False


def check_entrance(x, y, img=None):
    if img == '8x6':
        if 809 < x < 985 and 834 < y < 840:
            return True
    if img == '16x9':
        if 809 < x < 883 and 782 < y < 798:
            return True
    if img == '32x18':
        if 807 < x < 841 and 791 < y < 798:
            return True
    return False

# x,y = 525,318
# path = '16x9_wall.txt'
# wall = np.loadtxt(path,delimiter=',')
# #start_time = time.time()
# print(check_wall(x,y,wall))
# end_time = time.time()
# elapsed_time = end_time - start_time
# print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

# 16x9
# entrance x:710-788, y:0-10
# out x:810-882, y:783-797 

# 32x18
# entrance x:756-790, y:0-5
# out x:808-840, y:792-797

# 738,286
