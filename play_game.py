import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from check import *
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
font = cv2.FONT_HERSHEY_SIMPLEX
total = 0
org = (30, 50)
fontScale = 1
color = (139, 0, 0)
thickness = 2
quit = False


def play(idx, path, level, show_camera=False):
    s = []
    start, win = [False] * 2
    x1, y1, x2, y2, x3, y3 = [0] * 6
    new_w, new_h = 1600, 840
    # new_w, new_h = 1000, 800
    r = 10

    maze = np.array(Image.open(f'maze/{path}.png').convert('RGB'))
    maze = cv2.resize(maze, (new_w, new_h), cv2.INTER_CUBIC)
    # agent_img = np.array(Image.open('robot.png').convert('RGB'))
    # agent_img = cv2.resize(agent_img, (10, 10), interpolation=cv2.INTER_AREA)

    # For webcam input:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    with mp_hands.Hands(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5) as hands:
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
            show_img = cv2.flip(show_img, 1)

            if not start:
                show_img = cv2.putText(show_img, 'Put index finger at the entrance',
                                       org, font, fontScale, color, thickness, cv2.LINE_AA)

            if check_entrance(x1, y1, path):
                start = True

            if start:
                s.append(time.time())
                elapsed_time = round((time.time() - s[0]), 2)
                show_img = cv2.putText(show_img, f'Time: {elapsed_time}',
                                       org, font, fontScale, color, thickness, cv2.LINE_AA)
            cv2.imshow('Resized Image', show_img)

            if check_destination(x1, y1, path) and start:
                total_time = round((time.time() - s[0]), 2)
                show_img = cv2.putText(show_img, f'YOU PASS MAP {idx} LEVEL {level}!',
                                       (200, 400), font, 3, color, 4, cv2.LINE_AA)
                show_img = cv2.putText(show_img, f'Elapsed time: {int(total_time // 60)}m{total_time % 60}s',
                                       (350, 500), font, 2, color, 3, cv2.LINE_AA)
                cv2.imshow('Resized Image', show_img)
                cv2.waitKey(500)

                if idx == 3 and level == 3:
                    show_img = cv2.putText(gray.copy(), f'CONGRATULATION! YOU WIN!!!',
                                           (100, 400), font, 3, color, 4, cv2.LINE_AA)
                    show_img = cv2.putText(show_img, f'Total time: {int(total // 60)}m{total_time % 60}s',
                                           (350, 500), font, 2, color, 3, cv2.LINE_AA)
                    cv2.imshow('Resized Image', show_img)

                cv2.waitKey(600)
                win = True
                # quit = True
                del s
                break

            # check wall collision
            if (maze[y1 - r - 5: y1 + r + 5, x1 - r - 5: x1 + r + 5] == 0).any() and start:
                break
            if level >= 2 and (maze[y2 - r - 5: y2 + r + 5, x2 - r - 5: x2 + r + 5] == 0).any() and start:
                break
            if level >= 3 and (maze[y3 - r - 5: y3 + r + 5, x3 - r - 5: x3 + r + 5] == 0).any() and start:
                break

            if cv2.waitKey(5) & 0xFF == ord('q'):
                global quit
                quit = True
                cv2.destroyAllWindows()
                break

    cap.release()
    return win, 0


def live():
    level = 1
    for i, maze in enumerate(['8x6', '16x9', '32x18'], 1):
        while True:
            win, total_time = play(i, maze, level, show_camera=False)
            global total
            total += total_time
            if win:
                level += 1
                if level >= 4:
                    level = 1
                    break
            if quit:
                cv2.destroyAllWindows()
                return
    cv2.destroyAllWindows()


if __name__ == '__main__':
    live()
