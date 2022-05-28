import cv2
import numpy as np


def visualize(path_image ='16x9_resized.png',object = '.',size_obj = None,x = 0,y=0):
    
    image = cv2.imread(path_image)

    if object == '.':
        cv2.circle(image, (x, y),radius=10, color=(139, 0, 0), thickness=-1)
    else:
        image_overlay = cv2.imread(object)
        image_overlay = cv2.resize(image_overlay, (size_obj, size_obj), interpolation=cv2.INTER_CUBIC)
        image_overlay = cv2.cvtColor(image_overlay, cv2.COLOR_BGR2RGB)
        if x - size_obj/2 > 0 and x + size_obj/2 <image.shape[1]:
            start_x = x - int(size_obj/2)
        elif x - size_obj/2 < 0:
            start_x = 0
        else:
            start_x = image.shape[1] - size_obj
        if y - size_obj/2 > 0 and y + size_obj/2 <image.shape[0]:
            start_y = y - int(size_obj/2)
        elif y - size_obj/2 < 0:
            start_y = 0
        else:
            start_y = image.shape[0] - size_obj
        image[start_y:start_y+size_obj,start_x:start_x+size_obj,:] = image_overlay
    return image   
    

# path_image = '16x9_resized.png'
# x,y = 34,78
# object = 'robot.png'
# size_obj = 30
# wait = True
# visualize(path_image,object,size_obj,x,y,wait)