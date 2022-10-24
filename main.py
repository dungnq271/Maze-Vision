import pygame
import pygame_menu
from play_game import live

# initializing the constructor
pygame.init()

# screen resolution
res = (1920, 1080)

# opens up a window
screen = pygame.display.set_mode(res)

# set background image
background_path = 'images/pngtree-halloween-3d-orange-spider-web-jack-o-lantern-image_1462956.jpg'


# set button colour
button_colour = (255, 140, 0)

# predefined difficulty
DIFFICULTY = ['EASY']

# scale to full screen
scale = 0.95


def set_difficulty(value, difficulty):
    selected, index = value
    print(f'Selected difficulty: "{selected}" ({difficulty}) at index {index}')
    DIFFICULTY[0] = difficulty


def start_the_game(difficulty):
    # Do the job here !
    live(difficulty)


mytheme = pygame_menu.themes.THEME_ORANGE.copy()
myimage = pygame_menu.baseimage.BaseImage(
    image_path=background_path,
)
mytheme.background_color = myimage

menu = pygame_menu.Menu('Maze Vision', 
                        width=res[0]*scale, 
                        height=res[1]*scale,
                        theme=mytheme)

# menu.add.text_input('Name: ', default='Nguyen Dung')
menu.add.selector('Select difficulty: ',
                [('Easy', 'EASY'),
                ('Medium', 'MEDIUM'),
                ('Hard', 'HARD'),
                # ('Extreme', 'EXTREME')],
                ],
                background_color=button_colour,
                onchange=set_difficulty,
                selector_id='select_difficulty'
                )


menu.add.button('Play', 
                start_the_game, DIFFICULTY,
                background_color=button_colour,
                )
menu.add.button('Quit', 
                pygame_menu.events.EXIT,
                background_color=button_colour,
                )

menu.mainloop(screen)