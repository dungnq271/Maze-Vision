import numpy as np
#import time 
import cv2
def check_wall(x,y,image):
    if np.linalg.norm(image[y,x,:]-np.array([255,255,255])) <1e-9:
        return True
    return False
def check_entrance(x,y,img = None):
    if img == '16x9':
        if x>709 and x<789 and y>0 and y<10:
            return True
        else:
            return False
    if img == '32x18':
        if x>755 and x<791 and y>0 and y<6:
            return True
        else:
            return False
def check_out(x,y,img = None):
    if img == '16x9':
        if x>809 and x<883 and y>782 and y<798:
            return True
        else:
            return False
    if img == '32x18':
        if x>807 and x<841 and y>791 and y<798:
            return True
        else:
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
