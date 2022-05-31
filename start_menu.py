import pygame
import pygame_menu
from play_game import live

# initializing the constructor
pygame.init()

# screen resolution
res = (720, 720)

# opens up a window
screen = pygame.display.set_mode(res)

DIFFICULTY = ['EASY']


def set_difficulty(value, difficulty):
    selected, index = value
    print(f'Selected difficulty: "{selected}" ({difficulty}) at index {index}')
    DIFFICULTY[0] = difficulty


def start_the_game(difficulty):
    # Do the job here !
    live(difficulty)


menu = pygame_menu.Menu('Welcome', 720, 720,
                        theme=pygame_menu.themes.THEME_BLUE)

menu.add.text_input('Name: ', default='Nguyen Dung')
menu.add.selector('Select difficulty: ',
                  [('Easy', 'EASY'),
                   ('Medium', 'MEDIUM'),
                   ('Hard', 'HARD'),
                   # ('Extreme', 'EXTREME')],
                   ],
                  onchange=set_difficulty,
                  selector_id='select_difficulty'
                  )

menu.add.button('Play', start_the_game, DIFFICULTY)
menu.add.button('Quit', pygame_menu.events.EXIT)

menu.mainloop(screen)
