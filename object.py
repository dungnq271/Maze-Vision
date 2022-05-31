from utils.utils import *
from random import randint
import math


def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


class Object:
    def __init__(self, img_path, w, h):
        self.img = load_image(img_path, w, h)
        image = np.array(Image.open(img_path))
        self.image = image
        self.img = cv2.resize(image, (w, h), cv2.INTER_CUBIC)
        self.img_path = img_path
        self.w = w
        self.h = h
        self.w_range = None
        self.h_range = None
        self.x = None
        self.y = None
        self.new_w = None
        self.new_h = None

    def display(self, env, x, y):
        self.x = x
        self.y = y
        # w_range = [self.x-self.w//2, self.x+self.w//2]
        # h_range = [self.y-self.h//2, self.y+self.h//2]
        env_w = env.shape[1]
        env_h = env.shape[0]
        img = self.img

        if self.x < self.w // 2:
            self.x = self.w // 2
        if self.y < self.h // 2:
            self.y = self.h // 2
        if self.x > env_w - self.w // 2:
            self.x = env_w - self.w // 2
        if self.y > env_h - self.h // 2:
            self.y = env_h - self.h // 2

        # if self.x < self.w//2:
        #     self.new_w = self.x + self.w // 2
        #     img = img[:, self.w // 2 - self.x:]
        #     w_range = [0, self.new_w]
        # print(img.shape)
        #
        # if self.y < self.h//2:
        #     self.new_h = self.y + self.h // 2
        #     img = img[self.h // 2 - self.y:, :]
        #     h_range = [0, self.new_h]
        # print(img.shape)
        #
        # if self.x + self.w // 2 > env_w:
        #     img = img[:, :self.w // 2 + env_w - self.x]
        #     w_range = [self.x - self.w//2, env_w]
        # print(img.shape)
        #
        # if self.y + self.h // 2 > env_h:
        #     img = img[:self.h // 2 + env_h - self.y, :]
        #     h_range = [self.y - self.h//2, env_h]
        # print(img.shape)
        #
        # env[h_range[0]: h_range[1], w_range[0]: w_range[1]] = img
        y1, y2 = self.y - self.h // 2, self.y + self.h // 2
        x1, x2 = self.x - self.w // 2, self.x + self.w // 2

        # overlay transparent image
        # env[y1:y2, x1:x2] = env[y1:y2, x1:x2] * (1 - img[:, :, 3:] / 255) + \
        #                     img[:, :, :3] * (img[:, :, 3:] / 255)
        env = overlay(img, env, y1, y2, x1, x2)
        # env[y1:y2, x1:x2] = img
        return env


def overlay(img, env, y1, y2, x1, x2):
    env[y1:y2, x1:x2] = env[y1:y2, x1:x2] * (1 - img[:, :, 3:] / 255) + \
                        img[:, :, :3] * (img[:, :, 3:] / 255)
    return env


class Enemy(Object):
    def __init__(self, img_path, x, y, w, h, difficulty, speed=2):
        super().__init__(img_path, w, h)
        self.x = x
        self.y = y
        self.speed = speed
        self.mode = difficulty

    def move_towards_player(self, player):
        boost_x = 0 if self.mode[0] not in ['HARD', 'EXTREME'] else randint(0, 10)
        boost_y = 0 if self.mode[0] not in ['HARD', 'EXTREME'] else randint(0, 10)

        if self.x > player.x:
            self.x = self.x - self.speed - boost_x
        else:
            self.x += self.speed + boost_x
        if self.y > player.y:
            self.y -= (self.speed + boost_y)
        else:
            self.y += self.speed + boost_y

    def collide(self, player):
        if distance(self.x, self.y, player.x, player.y) < self.w:
            return True
