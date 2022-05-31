import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from check import *
import time
from utils.effects import *
from utils.utils import *
from object import *

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
window = 'MAZE'
font = cv2.FONT_HERSHEY_SIMPLEX
total = 0
fontScale = 1
color = (51, 153, 255)
thickness = 2
max_level = 3
num_map = 3
test = False


def play(idx, path, level, difficulty, show_camera=False, test=False):
    s = []
    start, win = [False] * 2
    x1, y1, x2, y2, x3, y3 = [0] * 6
    new_w, new_h = 1216, 830
    # new_w, new_h = 1000, 800
    if path == '32x18':
        r = 20
    else:
        r = 40

    margin = r - 20
    hpt = 3
    total_time = 0
    quit = False

    stime = 0
    ltime = stime

    maze = load_image(f'maze/{path}.png', new_w, new_h)
    heart = load_image('symbols/heart.png', 40, 40)
    heart = cv2.cvtColor(heart, cv2.COLOR_RGB2BGR)
    agent = Object('symbols/robot.png', r, r)
    enemy = Enemy('symbols/ghost.png', randint(0, new_h), randint(0, new_w), r, r, difficulty)
    congrats = load_image('symbols/congrats.png', new_w, new_h)
    congrats = cv2.cvtColor(congrats, cv2.COLOR_BGR2RGB)

    # For webcam input:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
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
            results = hands.process(image)

            # Draw the hand annotations on the image.
            # image.flags.writeable = True
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            maze_copy = maze.copy()

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                x1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * new_w * 1.5)
                y1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * new_h * 1.5)

                x2 = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * new_w * 1.5)
                y2 = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * new_h * 1.5)

                x3 = int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * new_w * 1.5)
                y3 = int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * new_h * 1.5)

                # image = display_agent(image, agent_img, x1, y1, new_w, new_h)
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            image = cv2.resize(image, (320, 180), cv2.INTER_CUBIC)
            image = cv2.flip(image, 1)
            maze_copy = agent.display(maze_copy, x1, y1)
            if start:
                if difficulty[0] != 'EASY':
                    maze_copy = enemy.display(maze_copy, enemy.x, enemy.y)

            if level >= 2:
                maze_copy = agent.display(maze_copy, x2, y2)
                # maze_copy = cv2.circle(maze_copy, (x2, y2),
                #                        radius=r, color=(139, 0, 0), thickness=-1)
            if level >= 3:
                maze_copy = cv2.circle(maze_copy, (x3, y3),
                                       radius=r, color=(139, 0, 0), thickness=-1)

            # Display image full screen
            # cv2.namedWindow(window, cv2.WND_PROP_FULLSCREEN)
            # cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            show_img = image if show_camera else maze_copy
            # Flip the image horizontally for a selfie-view display.
            show_img = cv2.flip(show_img, 1)
            stat_img = np.zeros((650, 320, 3))

            if not start:
                show_img = cv2.putText(show_img, 'Put your fingertip(s) at the entrance',
                                       (20, 40), font, fontScale, color, thickness, cv2.LINE_AA)

            # if touching entrance then start the game
            if check_entrance(x1, y1, r, path):
                start = True

            ltime = time.time()
            fps = 1 / (ltime - stime)
            stime = ltime

            if start:
                enemy.move_towards_player(agent)
                s.append(time.time())
                elapsed_time = round((time.time() - s[0]), 2)
                stat_img = cv2.putText(stat_img, str(hpt),
                                       (130, 270), font, fontScale, color, thickness, cv2.LINE_AA)
                stat_img[240:280, 160:200] = heart
                stat_img = cv2.putText(stat_img, f'Time {elapsed_time}',
                                       (80, 320), font, fontScale, color, thickness, cv2.LINE_AA)
                stat_img = cv2.putText(stat_img, f'FPS  {fps:.1f}',
                                       (80, 370), font, fontScale, color, thickness, cv2.LINE_AA)

            # display image windows
            cv2.imshow(window, show_img)
            cv2.moveWindow(window, 0, 0)

            cv2.imshow('Camera', image)
            cv2.moveWindow('Camera', 1216, 0)

            cv2.imshow('Stats', stat_img)
            cv2.moveWindow('Stats', 1216, 180)

            # check if reach the destination
            if check_destination(x1, y1, r, path) and start:
                total_time = round((time.time() - s[0]), 2)
                show_img = cv2.putText(show_img, f'YOU PASS MAP {idx} LEVEL {level}!',
                                       (160, 400), font, 2, color, 4, cv2.LINE_AA)
                show_img = cv2.putText(show_img, f'Time: {int(total_time // 60)}m {total_time % 60}s',
                                       (360, 500), font, 2, color, 3, cv2.LINE_AA)
                cv2.imshow(window, show_img)
                cv2.waitKey(2000)

                if idx == num_map and level == max_level-1:
                    # show_img = cv2.putText(show_img, f'Total time: {int(total // 60)}m{total_time % 60}s',
                    #                        (350, 500), font, 2, color, 3, cv2.LINE_AA)
                    cv2.imshow(window, congrats)

                cv2.waitKey(600)
                win = True
                # quit = True
                del s
                break

            # check collision
            if not test:
                if start and\
                        ((maze[y1 - r + margin: y1 + r - margin, x1 - r + margin: x1 + r - margin] == 0).any()
                         or (level >= 2 and
                             (maze[y2 - r + margin: y2 + r - margin, x2 - r + margin: x2 + r - margin] == 0).any())
                         or (level >= 3
                             and (maze[y3 - r + margin: y3 + r - margin, x3 - r + margin: x3 + r - margin] == 0).any())):
                    hpt -= 1
                    if hpt == 0:
                        # reset enemy position
                        enemy = Enemy('symbols/ghost.png', randint(0, new_h), randint(0, new_w), r, r, difficulty)
                        break

                if start and difficulty[0] != 'EASY' and enemy.collide(agent):
                    hpt -= 1
                    if hpt == 0:
                        image, audio = add_effect(new_w, new_h)
                        cv2.imshow(window, image)
                        play_audio(audio)
                        cv2.waitKey(50)
                        enemy = Enemy('symbols/ghost.png', randint(0, new_h), randint(0, new_w), r, r, difficulty)
                        break

            if cv2.waitKey(5) & 0xFF == ord('q'):
                quit = True
                cv2.destroyAllWindows()
                break

    # cap.release()
    return win, total_time, quit


def live(difficulty=None):
    if difficulty is None:
        difficulty = ['MEDIUM']

    level = 1
    for i, maze in enumerate(['8x6', '16x9', '32x18'], 1):
        while True:
            win, total_time, q = play(i, maze, level, difficulty, show_camera=False, test=test)
            global total
            total += total_time
            if win:
                level += 1
                if level >= max_level:
                    level = 1
                    break
            if q:
                cv2.destroyAllWindows()
                return
    cv2.destroyAllWindows()


if __name__ == '__main__':
    live()
