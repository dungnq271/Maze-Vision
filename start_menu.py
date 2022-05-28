import pygame
import pygame_menu
from play_game import live

# initializing the constructor
pygame.init()

# screen resolution
res = (720, 720)

# opens up a window
screen = pygame.display.set_mode(res)


def set_difficulty(value, difficulty):
    # Do the job here !
    pass


def start_the_game():
    # Do the job here !
    live()


menu = pygame_menu.Menu('Welcome', 400, 300,
                        theme=pygame_menu.themes.THEME_BLUE)

menu.add.text_input('Name: ', default='Nguyen Dung')
# menu.add.selector('Difficulty: ', [('Hard', 1), ('Easy', 2)], onchange=set_difficulty)
menu.add.button('Play', start_the_game)
menu.add.button('Quit', pygame_menu.events.EXIT)

menu.mainloop(screen)

