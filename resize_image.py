import cv2
from PIL import Image
import numpy as np

def resize_image(path_img = '32x18.png',path_to_save = '32x18_resized.png'):
    resolution = (1600, 800)
    img = np.array(Image.open(path_img).convert('RGB'))

    img = cv2.resize(img, (1600, 800), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cv2.imwrite(path_to_save,img)