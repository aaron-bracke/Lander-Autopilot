import pygame as pg
import time
from math import sin, cos, radians, sqrt
import random

class Lander:
    def __init__(self, player_controlled):
        self.parameters = {
            "thrust_level": 0.0,
            "pitch_angle": 0.0,
            "x": random.uniform(width_screen_pixels/2 - 150.0, width_screen_pixels/2 + 150.0),
            "y": random.uniform(height_screen_pixels+50.0, height_screen_pixels+150.0),
            "vx": random.uniform(-10.0, 10.0),
            "vy": random.uniform(-60.0, -20.0),
            "mass": total_mass,
            "fuel": total_fuel,
            "altitude": 0.0,
            "condition": "flying"
            }

        self.player_controlled = player_controlled

        if self.player_controlled:
            self.Update = self.PlayerUpdate
        else:
            self.Update = self.AIUpdate

    def PlayerUpdate(self, dt):
        """Retrieve player input and update physics"""
        # Input
        if keys[pg.K_UP]:
            self.parameters['thrust_level'] = min(100, self.parameters['thrust_level'] + throttle_step)
        if keys[pg.K_DOWN]:
            self.parameters['thrust_level'] = max(0, self.parameters['thrust_level'] - throttle_step)
        if keys[pg.K_LEFT]:
            self.parameters['pitch_angle'] += pitch_step
            if self.parameters['pitch_angle'] > 180.0: self.parameters['pitch_angle'] -= 360
        if keys[pg.K_RIGHT]:
            self.parameters['pitch_angle'] -= pitch_step
            if self.parameters['pitch_angle'] < -180.0: self.parameters['pitch_angle'] += 360

        # Physics
        self.parameters['fuel'] -= self.parameters['thrust_level'] * dt / 6
        if self.parameters['fuel'] <= 0.0:
            thrust_multiplier = 0.0
            self.parameters['fuel'] = 0.0
        else:
            thrust_multiplier = thrust_power

        self.parameters['mass'] = total_mass - total_fuel + self.parameters['fuel']


        Fx = thrust_multiplier * self.parameters['thrust_level'] * (-sin(radians(self.parameters['pitch_angle'])))
        Fy = thrust_multiplier * self.parameters['thrust_level'] * (cos(radians(self.parameters['pitch_angle']))) - gravity * self.parameters['mass']

        self.parameters['vx'] += (Fx / self.parameters['mass']) * dt
        self.parameters['vy'] += (Fy / self.parameters['mass']) * dt

        self.parameters['x'] += self.parameters['vx'] * dt
        self.parameters['y'] += self.parameters['vy'] * dt

        self.parameters['altitude'] = (self.parameters['y'] - ground_height - 32) / 10.0

        # When the lander touches the ground
        if self.parameters['altitude'] <= 0.0:  
            if abs(self.parameters['pitch_angle']) < 10.0 and sqrt(self.parameters['vx']**2 + self.parameters['vy']**2) < 15.0:
                self.parameters['condition'] = "has_landed"
            else:
                self.parameters['condition'] = "has_crashed"
            self.parameters['vx'] = 0
            self.parameters['vy'] = 0
            self.parameters['y'] = ground_height + 32
            self.parameters['altitude'] = 0.0

    def AIUpdate(self, dt):
        """Retrieve AI input and update physics"""
        # Input
        if random.choice([0, 1]):
            self.parameters['thrust_level'] = min(100, self.parameters['thrust_level'] + throttle_step)
        if random.choice([0, 1]):
            self.parameters['thrust_level'] = max(0, self.parameters['thrust_level'] - throttle_step)
        if random.choice([0, 1]):
            self.parameters['pitch_angle'] += pitch_step
            if self.parameters['pitch_angle'] > 180.0: self.parameters['pitch_angle'] -= 360
        if random.choice([0, 1]):
            self.parameters['pitch_angle'] -= pitch_step
            if self.parameters['pitch_angle'] < -180.0: self.parameters['pitch_angle'] += 360

        # Physics
        self.parameters['fuel'] -= self.parameters['thrust_level'] * dt / 6
        if self.parameters['fuel'] <= 0.0:
            thrust_multiplier = 0.0
            self.parameters['fuel'] = 0.0
        else:
            thrust_multiplier = thrust_power

        self.parameters['mass'] = total_mass - total_fuel + self.parameters['fuel']


        Fx = thrust_multiplier * self.parameters['thrust_level'] * (-sin(radians(self.parameters['pitch_angle'])))
        Fy = thrust_multiplier * self.parameters['thrust_level'] * (cos(radians(self.parameters['pitch_angle']))) - gravity * self.parameters['mass']

        self.parameters['vx'] += (Fx / self.parameters['mass']) * dt
        self.parameters['vy'] += (Fy / self.parameters['mass']) * dt

        self.parameters['x'] += self.parameters['vx'] * dt
        self.parameters['y'] += self.parameters['vy'] * dt

        self.parameters['altitude'] = (self.parameters['y'] - ground_height - 32) / 10.0

        # When the lander touches the ground
        if self.parameters['altitude'] <= 0.0:  
            if abs(self.parameters['pitch_angle']) < 10.0 and sqrt(self.parameters['vx']**2 + self.parameters['vy']**2) < 15.0:
                self.parameters['condition'] = "has_landed"
            else:
                self.parameters['condition'] = "has_crashed"
            self.parameters['vx'] = 0
            self.parameters['vy'] = 0
            self.parameters['y'] = ground_height + 32
            self.parameters['altitude'] = 0.0

    def DrawText(self):
        """Show information on screen"""
        font = pg.font.SysFont("consolas", 20, True)
        text_colour = (237, 125, 58)

        text1 = f"Thrust setting: {self.parameters['thrust_level'] if self.parameters['fuel'] != 0.0 else 0:0.01f}%"
        text1_img = font.render(text1, True, text_colour)
        text1_rect = text1_img.get_rect()
        text1_rect.left = 10
        text1_rect.top = 10

        text2 = f"Pitch angle: {self.parameters['pitch_angle']:0.01f}" + u"\N{DEGREE SIGN}"
        text2_img = font.render(text2, True, text_colour)
        text2_rect = text2_img.get_rect()
        text2_rect.left = 10
        text2_rect.top = text1_rect.bottom

        text3 = f"Altitude: {self.parameters['altitude']:0.01f}"
        text3_img = font.render(text3, True, text_colour)
        text3_rect = text3_img.get_rect()
        text3_rect.left = 10
        text3_rect.top = text2_rect.bottom

        text4 = f"Velocity: {sqrt(self.parameters['vx']**2 + self.parameters['vy']**2) / 10.0:0.01f}"
        text4_img = font.render(text4, True, text_colour)
        text4_rect = text4_img.get_rect()
        text4_rect.left = 10
        text4_rect.top = text3_rect.bottom

        text5 = f"Fuel left: {self.parameters['fuel']:0.01f}"
        text5_img = font.render(text5, True, text_colour)
        text5_rect = text5_img.get_rect()
        text5_rect.left = 10
        text5_rect.top = text4_rect.bottom

        screen.blit(text1_img, text1_rect)
        screen.blit(text2_img, text2_rect)
        screen.blit(text3_img, text3_rect)
        screen.blit(text4_img, text4_rect)
        screen.blit(text5_img, text5_rect)

        if self.parameters['condition'] == "has_landed":
            font = pg.font.SysFont('consolas', 40, True)
            text8 = "The Eagle has landed!"
            text8_img = font.render(text8, True, text_colour)
            text8_rect = text8_img.get_rect()
            text8_rect.centerx = int(width_screen_pixels/2)
            text8_rect.centery = int(height_screen_pixels/2 - height_screen_pixels/5)

            text9 = f"Distance from target: {abs(self.parameters['x'] - landing_target)/10:0.01f}"
            text9_img = font.render(text9, True, text_colour)
            text9_rect = text9_img.get_rect()
            text9_rect.centerx = text8_rect.centerx
            text9_rect.top = text8_rect.bottom

            
            screen.blit(text8_img, text8_rect)
            screen.blit(text9_img, text9_rect)
        elif self.parameters['condition'] == "has_crashed":
            font = pg.font.SysFont('consolas', 40, True)
            text8 = "The Eagle has crashed!"
            text8_img = font.render(text8, True, text_colour)
            text8_rect = text8_img.get_rect()
            text8_rect.centerx = int(width_screen_pixels/2)
            text8_rect.centery = int(height_screen_pixels/2 - height_screen_pixels/5)
            screen.blit(text8_img, text8_rect)

    def DrawLander(self):
        """Draw the lander"""
        if self.parameters['thrust_level'] == 0.0 or self.parameters['fuel'] == 0.0:
            lander_image = engine_off_image
        elif 0 < self.parameters['thrust_level'] < 50:
            lander_image = engine_med_image
        else:
            lander_image = engine_high_image

        rotated_image = pg.transform.rotate(lander_image, self.parameters['pitch_angle'])
        new_rect = rotated_image.get_rect(center = lander_image.get_rect(topleft = (self.parameters['x'] - 15, height_screen_pixels - self.parameters['y'])).center)
        screen.blit(rotated_image, new_rect.topleft)

def DrawBackground():
    """Draw the background"""
    screen.fill(background_colour)

def DrawGround():
    """Draw the ground"""
    screen.fill(ground_colour, rect=(0, height_screen_pixels - ground_height, width_screen_pixels, ground_height))

def DrawTarget():
    """Draw the target landing zone"""
    screen.fill(target_colour, rect=((landing_target - target_range/2, height_screen_pixels - ground_height), (target_range, ground_height / 4)))

# Constants
width_screen_pixels = 1080
height_screen_pixels = 600

gravity = 10
ground_height = 100
throttle_step = 2
thrust_power = 150
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

# Parameters
has_landed = False
has_crashed = False
landing_target = random.uniform(width_screen_pixels/5, 4*width_screen_pixels/5)
target_range = 70
total_fuel = 250.0
total_mass = 600.0

# Main game loop
last_time = 0.0
running = True

lander_list = []
for i in range(4):
    lander_list.append(Lander(False))
# lander_list.append(Lander(True))

while running:
    pg.event.pump()
    keys = pg.key.get_pressed()

    DrawBackground()
    DrawGround()
    DrawTarget()

    for lander in lander_list:
        if not (lander.parameters['condition'] == "has_landed" or lander.parameters['condition'] == "has_crashed"):
            lander.Update(pg.time.get_ticks()/1000 - last_time)
        lander.DrawLander()
        if lander.player_controlled:
            lander.DrawText()

    last_time = pg.time.get_ticks()/1000

    if keys[pg.K_ESCAPE]:                       # QUIT the game
        running = False

    pg.display.flip()                           # Update the screen

    time.sleep(0.02)

pg.quit()