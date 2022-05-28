import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from utils.utils import show_maze

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def display_agent(img, agent, x, y, w, h):
    img_w = img.shape[0]
    img_h = img.shape[1]
    left = x - w // 2 if x - w // 2 >= 0 else w // 2 + x
    right = x + w // 2 if x + w // 2 <= img_w else img_w
    upper = y - h // 2 if y - h // 2 >= 0 else 0
    below = y + h // 2 if y + h // 2 <= img_h else img_h
    # cropped_agent = agent[]
    # img[(y-h//2):(y+h//2), (x-w//2):(x+w//2)] = agent_img
    return img


def play(paths):
    new_w, new_h = 50, 50

    maze = np.array(Image.open(paths).convert('RGB'))
    maze = cv2.resize(maze, (1600, 800), cv2.INTER_CUBIC)
    # agent_img = np.array(Image.open('robot.png').convert('RGB'))
    # agent_img = cv2.resize(agent_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # agent_img = cv2.cvtColor(agent_img, cv2.COLOR_BGR2RGB)

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
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            # image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            maze_copy = maze.copy()

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    x1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * 1600)
                    y1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * 800)

                    x2 = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * 1600)
                    y2 = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * 800)

                    x3 = int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * 1600)
                    y3 = int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * 800)
                    maze_copy = cv2.circle(maze_copy, (x1, y1),
                                           radius=10, color=(139, 0, 0), thickness=-1)
                    # maze_copy = cv2.circle(maze_copy, (x2, y2),
                    #                        radius=10, color=(139, 0, 0), thickness=-1)
                    # maze_copy = cv2.circle(maze_copy, (x3, y3),
                    #                        radius=10, color=(139, 0, 0), thickness=-1)
                    # image = display_agent(image, agent_img, x1, y1, new_w, new_h)
                    # print(x1, y1)
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

            # Display image full screen
            cv2.namedWindow('Resized Image', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('Resized Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('Resized Image', cv2.flip(maze_copy, 1))
            if cv2.waitKey(5) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
    cap.release()


if __name__ == '__main__':
    mazes = 'maze/16x9.png'
    play(mazes)
