import pygame as pg
import time
from math import sin, cos, radians, sqrt
import random


def Update(delta_time):
    """Retrieve the controller input and update physics"""
    # Input
    if keys[pg.K_UP]:
        parameters['thrust_level'] = min(100, parameters['thrust_level'] + throttle_step)
    if keys[pg.K_DOWN]:
        parameters['thrust_level'] = max(0, parameters['thrust_level'] - throttle_step)
    if keys[pg.K_LEFT]:
        parameters['pitch_angle'] += pitch_step
        if parameters['pitch_angle'] > 180.0: parameters['pitch_angle'] -= 360
    if keys[pg.K_RIGHT]:
        parameters['pitch_angle'] -= pitch_step
        if parameters['pitch_angle'] < -180.0: parameters['pitch_angle'] += 360

    # Physics
    Fx = thrust_power * parameters['thrust_level'] * (-sin(radians(parameters['pitch_angle'])))
    Fy = thrust_power * parameters['thrust_level'] * (cos(radians(parameters['pitch_angle']))) - gravity * parameters['mass']

    parameters['vx'] += (Fx / parameters['mass']) * delta_time
    parameters['vy'] += (Fy / parameters['mass']) * delta_time

    parameters['x'] += parameters['vx'] * delta_time
    parameters['y'] += parameters['vy'] * delta_time

    parameters['altitude'] = (parameters['y'] - ground_height - 32) / 10.0

    # When the lander touches the ground
    if parameters['altitude'] <= 0.0:  
        if abs(parameters['pitch_angle']) < 10.0 and sqrt(parameters['vx']**2 + parameters['vy']**2) < 15.0:
            parameters['condition'] = "has_landed"
        else:
            parameters['condition'] = "has_crashed"
        parameters['vx'] = 0
        parameters['vy'] = 0
        parameters['y'] = ground_height + 32
        parameters['altitude'] = 0.0

def DrawText():
    """Show information on screen"""
    font = pg.font.SysFont("consolas", 20, True)
    text_colour = (237, 125, 58)

    text1 = f"Thrust setting: {parameters['thrust_level']:0.01f}%"
    text1_img = font.render(text1, True, text_colour)
    text1_rect = text1_img.get_rect()
    text1_rect.left = 10
    text1_rect.top = 10

    text2 = f"Pitch angle: {parameters['pitch_angle']:0.01f}" + u"\N{DEGREE SIGN}"
    text2_img = font.render(text2, True, text_colour)
    text2_rect = text2_img.get_rect()
    text2_rect.left = 10
    text2_rect.top = text1_rect.bottom

    text3 = f"Altitude: {parameters['altitude']:0.01f}"
    text3_img = font.render(text3, True, text_colour)
    text3_rect = text3_img.get_rect()
    text3_rect.left = 10
    text3_rect.top = text2_rect.bottom

    text4 = f"Velocity: {sqrt(parameters['vx']**2 + parameters['vy']**2) / 10.0:0.01f}"
    text4_img = font.render(text4, True, text_colour)
    text4_rect = text4_img.get_rect()
    text4_rect.left = 10
    text4_rect.top = text3_rect.bottom

    screen.blit(text1_img, text1_rect)
    screen.blit(text2_img, text2_rect)
    screen.blit(text3_img, text3_rect)
    screen.blit(text4_img, text4_rect)

    if parameters['condition'] == "has_landed":
        font = pg.font.SysFont('consolas', 40, True)
        text5 = "The Eagle has landed!"
        text5_img = font.render(text5, True, text_colour)
        text5_rect = text5_img.get_rect()
        text5_rect.centerx = int(width_screen_pixels/2)
        text5_rect.centery = int(height_screen_pixels/2 - height_screen_pixels/5)

        text6 = f"Distance from target: {abs(parameters['x'] - landing_target)/10:0.01f}"
        text6_img = font.render(text6, True, text_colour)
        text6_rect = text6_img.get_rect()
        text6_rect.centerx = text5_rect.centerx
        text6_rect.top = text5_rect.bottom

        
        screen.blit(text5_img, text5_rect)
        screen.blit(text6_img, text6_rect)
    elif parameters['condition'] == "has_crashed":
        font = pg.font.SysFont('consolas', 40, True)
        text5 = "The Eagle has crashed!"
        text5_img = font.render(text5, True, text_colour)
        text5_rect = text5_img.get_rect()
        text5_rect.centerx = int(width_screen_pixels/2)
        text5_rect.centery = int(height_screen_pixels/2 - height_screen_pixels/5)
        screen.blit(text5_img, text5_rect)

def DrawBackground():
    """Draw the background"""
    screen.fill(background_colour)

def DrawGround():
    """Draw the ground"""
    screen.fill(ground_colour, rect=(0, height_screen_pixels - ground_height, width_screen_pixels, ground_height))

def DrawTarget():
    """Draw the target landing zone"""
    screen.fill(target_colour, rect=(landing_target - target_range/2, height_screen_pixels - ground_height, target_range, ground_height / 4))

def DrawLander():
    """Draw the lander"""
    if parameters['thrust_level'] == 0:
        lander_image = engine_off_image
    elif 0 < parameters['thrust_level'] < 50:
        lander_image = engine_med_image
    else:
        lander_image = engine_high_image

    rotated_image = pg.transform.rotate(lander_image, parameters['pitch_angle'])
    new_rect = rotated_image.get_rect(center = lander_image.get_rect(topleft = (parameters['x'], height_screen_pixels - parameters['y'])).center)
    screen.blit(rotated_image, new_rect.topleft)

# Constants
width_screen_pixels = 1080
height_screen_pixels = 600

gravity = 10
ground_height = 100
throttle_step = 2
thrust_power = 100
pitch_step = 1

background_colour = (57, 70, 72)
ground_colour = (211, 212, 217)
target_colour = (199, 62, 29)

# Start PyGame
pg.init()
reso = (int(width_screen_pixels), int(height_screen_pixels))
screen = pg.display.set_mode(reso, pg.NOFRAME)
pg.display.set_caption('Lunar Lander')
window_icon = pg.image.load(r'Assets\lander-engine_high.png')
pg.display.set_icon(window_icon)
engine_off_image = pg.image.load(r'Assets\lander-engine_off.png')
engine_med_image = pg.image.load(r'Assets\lander-engine_med.png')
engine_high_image = pg.image.load(r'Assets\lander-engine_high.png')

parameters = {
     "thrust_level": 0.0,
     "pitch_angle": 0.0,
     "x": 500.0,
     "y": 500.0,
     "vx": 0.0,
     "vy": 0.0,
     "mass": 500.0,
     "altitude": 0.0,
     "condition": "flying"
     }

has_landed = False
has_crashed = False
landing_target = random.uniform(width_screen_pixels/5, 4*width_screen_pixels/5)
target_range = 70

# Main game loop
last_time = 0.0
running = True
while running:

    pg.event.pump()
    keys = pg.key.get_pressed()

    if not (parameters['condition'] == "has_landed" or parameters['condition'] == "has_crashed"):
        Update(pg.time.get_ticks()/1000 - last_time)

    last_time = pg.time.get_ticks()/1000

    DrawBackground()
    DrawGround()
    DrawTarget()
    DrawLander()
    DrawText()

    if keys[pg.K_ESCAPE]:                       # QUIT the game
        running = False

    pg.display.flip()                           # Update the screen

    time.sleep(0.02)

pg.quit()