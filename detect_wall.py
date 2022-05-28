from matplotlib import image
import numpy as np


# import matplotlib.pyplot as plt

def detect_wall(path_image='32x18_resized.png', path_to_save='32x18_wall.txt'):
    img = image.imread(path_image)
    wall = []

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if np.linalg.norm(img[i, j, :] - np.array([0, 0, 0])) < 1.7:
                wall.append([j, i])
    wall = np.array(wall)
    print(wall.shape, wall[45256])
    np.savetxt(path_to_save, wall, delimiter=',')
