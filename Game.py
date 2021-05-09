import time
import random
from math import sin, cos, radians, sqrt
import pygame as pg
import numpy as np
from NeuralNetwork import NeuralNetwork

class Lander:
    def __init__(self, player_controlled, read_only=False, parent1=None, parent2=None):
        self.player_controlled = player_controlled
        if self.player_controlled:
            self.Update = self.PlayerUpdate
        else:
            self.Update = self.AIUpdate
            self.neural_network = NeuralNetwork(read_only, parent1, parent2)

        self.SetInitialConditions()

    def SetInitialConditions(self):
        self.parameters = {
            "thrust_level": 0.0,
            "pitch_angle": random.uniform(-10.0, 10.0),
            "x": random.uniform(width_screen_pixels/2 - 150.0, width_screen_pixels/2 + 150.0),
            "altitude": 0.0,
            "y": random.uniform(height_screen_pixels+50.0, height_screen_pixels+150.0),
            "vx": random.uniform(-10.0, 10.0),
            "vy": random.uniform(-60.0, -20.0),
            "mass": total_mass,
            "fuel": total_fuel,
            "condition": "flying",
            "cost": 10000.0
            }
        self.parameters['altitude'] = (self.parameters['y'] - ground_height - 32) / 10.0

    def PlayerUpdate(self, dt, keys):
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
            if abs(self.parameters['pitch_angle']) < 10.0 and sqrt(self.parameters['vx']**2 + self.parameters['vy']**2) < 20.0:
                self.parameters['condition'] = "has_landed"
            else:
                self.parameters['condition'] = "has_crashed"
            self.parameters['vx'] = 0
            self.parameters['vy'] = 0
            self.parameters['y'] = ground_height + 32
            self.parameters['altitude'] = 0.0

    def AIUpdate(self, dt, landing_target):
        """Retrieve AI input and update physics"""
        # Retrieve input from neural network
        AI_input = self.neural_network.Predict(np.array([[self.parameters['thrust_level']/100], [sin(radians(self.parameters['pitch_angle']))], \
            [cos(radians(self.parameters['pitch_angle']))], [self.parameters['x']], [self.parameters['altitude']], [self.parameters['vx']], \
            [self.parameters['vy']], [self.parameters['fuel']], [landing_target]]))
        
        # Input
        if AI_input[0]:
            self.parameters['thrust_level'] = min(100, self.parameters['thrust_level'] + throttle_step)
        if AI_input[1]:
            self.parameters['thrust_level'] = max(0, self.parameters['thrust_level'] - throttle_step)
        if AI_input[2]:
            self.parameters['pitch_angle'] += pitch_step
            if self.parameters['pitch_angle'] > 180.0: self.parameters['pitch_angle'] -= 360
        if AI_input[3]:
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
            self.parameters['cost'] = sqrt(self.parameters['vx']**2 + self.parameters['vy']**2) + \
                                      abs(self.parameters['pitch_angle']) + \
                                      abs(self.parameters['x'] - landing_target) / 3 + \
                                      2*((total_fuel) - self.parameters['fuel'])

            if abs(self.parameters['pitch_angle']) < 10.0 and sqrt(self.parameters['vx']**2 + self.parameters['vy']**2) < 20.0:
                self.parameters['condition'] = "has_landed"
            else:
                self.parameters['condition'] = "has_crashed"
            self.parameters['vx'] = 0
            self.parameters['vy'] = 0
            self.parameters['y'] = ground_height + 32
            self.parameters['altitude'] = 0.0

    def DrawText(self, display_surface, landing_target):
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

        display_surface.blit(text1_img, text1_rect)
        display_surface.blit(text2_img, text2_rect)
        display_surface.blit(text3_img, text3_rect)
        display_surface.blit(text4_img, text4_rect)
        display_surface.blit(text5_img, text5_rect)

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

            
            display_surface.blit(text8_img, text8_rect)
            display_surface.blit(text9_img, text9_rect)
        elif self.parameters['condition'] == "has_crashed":
            font = pg.font.SysFont('consolas', 40, True)
            text8 = "The Eagle has crashed!"
            text8_img = font.render(text8, True, text_colour)
            text8_rect = text8_img.get_rect()
            text8_rect.centerx = int(width_screen_pixels/2)
            text8_rect.centery = int(height_screen_pixels/2 - height_screen_pixels/5)
            display_surface.blit(text8_img, text8_rect)

    def DrawLander(self, display_surface, sprite_list):
        """Draw the lander"""
        if self.parameters['thrust_level'] == 0.0 or self.parameters['fuel'] == 0.0:
            lander_image = sprite_list[0]
        elif 0 < self.parameters['thrust_level'] < 50:
            lander_image = sprite_list[1]
        else:
            lander_image = sprite_list[2]

        rotated_image = pg.transform.rotate(lander_image, self.parameters['pitch_angle'])
        new_rect = rotated_image.get_rect(center = lander_image.get_rect(topleft = (self.parameters['x'] - 15, height_screen_pixels - self.parameters['y'])).center)
        display_surface.blit(rotated_image, new_rect.topleft)

    def DetermineCost(self):
        return self.parameters['cost']

def DrawBackground(display_surface):
    """Draw the background"""
    display_surface.fill(background_colour)

def DrawGround(display_surface):
    """Draw the ground"""
    display_surface.fill(ground_colour, rect=(0, height_screen_pixels - ground_height, width_screen_pixels, ground_height))

def DrawTarget(display_surface, landing_target):
    """Draw the target landing zone"""
    display_surface.fill(target_colour, rect=((landing_target - target_range/2, height_screen_pixels - ground_height), (target_range, ground_height / 4)))

def WriteToFile(content):
    with open("all-time-best.txt", 'w') as f:
        f.write(str(content))

def ReadFromFile():
    with open("all-time-best.txt", 'r') as f:
        content = float(f.read())
    return content

def CreateLandingTarget():
    return random.uniform(width_screen_pixels/5, 4*width_screen_pixels/5)

def TrainController(landers_per_gen=40, num_of_gens = 100, time_before_skip = 25.0):
    dt = 0.02

    lander_list = []
    for j in range(int(landers_per_gen/4)):
            lander_list.append(Lander(False, True))

    for i in range(num_of_gens):
        landing_target = CreateLandingTarget()
        for j in range(int(3*landers_per_gen/4)):
            lander_list.append(Lander(False, False, np.random.choice(lander_list[:int(landers_per_gen/4)]).neural_network, np.random.choice(lander_list[:int(landers_per_gen/4)]).neural_network)) 

        # Main game loop
        t = 0
        running = True
        while running:
            t += dt
            still_flying = False
            for lander in lander_list:
                if not (lander.parameters['condition'] == "has_landed" or lander.parameters['condition'] == "has_crashed"):
                    still_flying = True
                    lander.Update(dt, landing_target)

            # End of this generation
            if not still_flying or t > time_before_skip:
                running = False
                lander_list.sort(key=Lander.DetermineCost)
                print(lander_list[0].DetermineCost())
                lander_list = lander_list[:int(landers_per_gen/4)]
                for lander in lander_list:
                    lander.SetInitialConditions()

    NeuralNetwork.SaveNeuralNetwork(lander_list[0].neural_network.parameters)

def PlayGame(display_ai=False):
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
    sprite_list = [engine_off_image, engine_med_image, engine_high_image]

    # Main game loop
    last_time = 0.0
    running = True

    landing_target = CreateLandingTarget()
    player_lander = Lander(True)
    if display_ai:
        ai_lander = Lander(False, True)

    while running:
        pg.event.pump()
        keys = pg.key.get_pressed()
        if keys[pg.K_ESCAPE]:                       # QUIT the game
            running = False

        DrawBackground(screen)
        DrawGround(screen)
        DrawTarget(screen, landing_target)

        if display_ai:
            if not (ai_lander.parameters['condition'] == "has_landed" or ai_lander.parameters['condition'] == "has_crashed"):
                ai_lander.Update(pg.time.get_ticks()/1000 - last_time, landing_target)
            ai_lander.DrawLander(screen, sprite_list)

        if not (player_lander.parameters['condition'] == "has_landed" or player_lander.parameters['condition'] == "has_crashed"):
            player_lander.Update(pg.time.get_ticks()/1000 - last_time, keys)
        player_lander.DrawLander(screen, sprite_list)
        player_lander.DrawText(screen, landing_target)

        last_time = pg.time.get_ticks()/1000
        pg.display.update()                           # Update the screen
        time.sleep(0.02)

    pg.quit()

# Constants
width_screen_pixels = 1080
height_screen_pixels = 600
target_range = 70

gravity = 10
ground_height = 100
throttle_step = 2
thrust_power = 150
pitch_step = 1
total_fuel = 250.0
total_mass = 600.0

background_colour = (57, 70, 72)
ground_colour = (211, 212, 217)
target_colour = (199, 62, 29)

# PlayGame(True)
TrainController()