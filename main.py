import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from check import *

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
quit = False


def display_agent(img, agent, x, y, w, h):
    img_w = img.shape[0]
    img_h = img.shape[1]
    # left = x - w // 2 if x - w // 2 >= 0 else w // 2 + x
    # right = x + w // 2 if x + w // 2 <= img_w else img_w
    # upper = y - h // 2 if y - h // 2 >= 0 else 0
    # below = y + h // 2 if y + h // 2 <= img_h else img_h
    img[y, w] = agent
    return img


def play(path, level, show_camera=False):
    dead, win = False, False
    new_w, new_h = 1600, 840
    # new_w, new_h = 1000, 800
    r = 10
    x1, y1 = 0, 0
    x2, y2 = 0, 0
    x3, y3 = 0, 0

    maze = np.array(Image.open(f'maze/{path}.png').convert('RGB'))
    maze = cv2.resize(maze, (new_w, new_h), cv2.INTER_CUBIC)
    # agent_img = np.array(Image.open('robot.png').convert('RGB'))
    # agent_img = cv2.resize(agent_img, (10, 10), interpolation=cv2.INTER_AREA)

    # For webcam input:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.8) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            gray = cv2.cvtColor(maze, cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image1 = cv2.resize(image, (image.shape[1] * 10, image.shape[0] * 10), cv2.INTER_CUBIC)
            results = hands.process(image)

            # image_flip = cv2.flip(image, 1)
            # results_flip = hands.process(image_flip)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            maze_copy = gray.copy()

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    x1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * new_w)
                    y1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * new_h)

                    x2 = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * new_w)
                    y2 = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * new_h)

                    x3 = int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * new_w)
                    y3 = int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * new_h)

                    # image = display_agent(image, agent_img, x1, y1, new_w, new_h)
                    if show_camera:
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
            # if (maze_copy[y1 - r - 5: y1 + r + 5, x1 - r - 5: x1 + r + 5] == 0).any():
            #     dead = True
            #     print('dead')

            maze_copy = cv2.circle(maze_copy, (x1, y1),
                                   radius=r, color=(139, 0, 0), thickness=-1)
            if level >= 2:
                maze_copy = cv2.circle(maze_copy, (x2, y2),
                                       radius=r, color=(139, 0, 0), thickness=-1)
            if level >= 3:
                maze_copy = cv2.circle(maze_copy, (x3, y3),
                                       radius=r, color=(139, 0, 0), thickness=-1)

            # Display image full screen
            cv2.namedWindow('Resized Image', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('Resized Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            show_img = image if show_camera else maze_copy
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('Resized Image', cv2.flip(show_img, 1))

            if check_out(x1, y1, path):
                print('You win!!!')
                win = True
                break

            if cv2.waitKey(5) & 0xFF == ord('q'):
                global quit
                quit = True
                break

            if dead:
                break

    cap.release()
    return win


def live():
    level = 1
    for maze in ['8x6', '16x9', '32x18']:
        while True:
            win = play(maze, level, show_camera=False)
            if win:
                level += 1
                if level >= 4:
                    level = 1
                    break
            if quit:
                cv2.destroyAllWindows()
                return


if __name__ == '__main__':
    live()
    cv2.destroyAllWindows()

