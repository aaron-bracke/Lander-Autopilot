import time
import random
from math import sin, cos, radians, degrees, sqrt, asin
import pygame as pg
import numpy as np
from NeuralNetwork import NeuralNetwork

class Lander:
    def __init__(self, control_mode, display_simulation, display_stats=False):

        self.control_mode = control_mode
        self.display_simulation = display_simulation
        self.display_stats = display_stats
        self.SetInitialConditions()

        if self.control_mode == "player":
            self.Update = self.PlayerUpdate
        elif self.control_mode == "neural_net":
            self.Update = self.NNUpdate
            self.neural_network = NeuralNetwork()
        elif self.control_mode == "autopilot":
            self.Update = self.AutopilotUpdate

    def SetInitialConditions(self):
        self.parameters = {
            "thrust_level": 0.0,
            "pitch_angle": random.randint(-10.0, 10.0),
            "x": random.uniform(width_screen_pixels/2 - 150.0, width_screen_pixels/2 + 150.0),
            "altitude": 0.0,
            "y": random.uniform(height_screen_pixels+50.0, height_screen_pixels+150.0),
            "vx": random.uniform(-10.0, 10.0),
            "vy": random.uniform(-60.0, -20.0),
            "mass": total_mass,
            "fuel": total_fuel,
            "condition": "flying",
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

    def NNUpdate(self, dt, landing_target):
        """Retrieve NN output and update physics"""
        # Fuel assessment
        self.parameters['fuel'] -= self.parameters['thrust_level'] * dt / 6
        if self.parameters['fuel'] <= 0.0:
            thrust_multiplier = 0.0
            self.parameters['fuel'] = 0.0
        else:
            thrust_multiplier = thrust_power
        self.parameters['mass'] = total_mass - total_fuel + self.parameters['fuel']

        # Retrieve input from neural network
        NN_target_values = self.neural_network.Predict(np.array([[self.parameters['thrust_level']/100], [sin(radians(self.parameters['pitch_angle']))], \
            [cos(radians(self.parameters['pitch_angle']))], [self.parameters['x']/width_screen_pixels], [self.parameters['y']/height_screen_pixels], [self.parameters['vx']/100.0], \
            [self.parameters['vy']/100.0], [self.parameters['fuel']/250], [landing_target/width_screen_pixels]]))

        # Determine the target values the autopilot would put forth in this situation
        target_alt = 20.0
        kx = 50.0 # 60.0    # Spring behavious in horizontal direction
        ky = 110.0  # 110.0 # Spring behavious in vertical direction

        if abs(self.parameters['x'] - landing_target) < 8.0 and abs(self.parameters['altitude'] - target_alt) < 30.0:
            self.parameters['condition'] = "initiated_landing"

        # During the final landing phase
        if self.parameters['condition'] == "initiated_landing":
            target_alt = -5.0
            kx = 20.0 # 60.0    # Spring behavious in horizontal direction
            ky = 225.0  # 110.0 # Spring behavious in vertical direction

        # Determine target values for pitch angle and thrust level
        cx = 2 * sqrt(kx * self.parameters['mass'])
        if thrust_multiplier * self.parameters['thrust_level'] != 0.0:
            autopilot_target_sin_pitch = (kx*(self.parameters['x'] - landing_target) + cx*self.parameters['vx']) / (thrust_multiplier * self.parameters['thrust_level'])
        else:
            autopilot_target_sin_pitch = sin(radians(self.parameters['pitch_angle']))

        cy = 2 * sqrt(ky * self.parameters['mass'])
        if thrust_multiplier != 0.0:
            autopilot_target_thrust_level = (self.parameters['mass'] * gravity - cy * self.parameters['vy'] - ky * (self.parameters['altitude'] - target_alt))\
                                    / (thrust_multiplier * cos(radians(self.parameters['pitch_angle'])))
        else:
            autopilot_target_thrust_level = 0.0

        # Create default neural network input
        NN_input = np.zeros((4, 1))

        print(f"Thrust level: {NN_target_values[0]:0.01f} \t Pitch angle: {NN_target_values[1]:0.01f}")
        
        # Determine input from target values
        if self.parameters['thrust_level'] < NN_target_values[0]*1.1*100.0:
            NN_input[0] = 1
        elif self.parameters['thrust_level'] > NN_target_values[0]*0.9*100.0:
            NN_input[1] = 1

        if sin(radians(self.parameters['pitch_angle'])) < NN_target_values[1]:
            NN_input[2] = 1
        elif sin(radians(self.parameters['pitch_angle'])) > NN_target_values[1]:
            NN_input[3] = 1

        # Input
        if NN_input[0]:
            self.parameters['thrust_level'] = min(100, self.parameters['thrust_level'] + throttle_step)
        if NN_input[1]:
            self.parameters['thrust_level'] = max(0, self.parameters['thrust_level'] - throttle_step)
        if NN_input[2]:
            self.parameters['pitch_angle'] += pitch_step
            if self.parameters['pitch_angle'] > 180.0: self.parameters['pitch_angle'] -= 360
        if NN_input[3]:
            self.parameters['pitch_angle'] -= pitch_step
            if self.parameters['pitch_angle'] < -180.0: self.parameters['pitch_angle'] += 360

        # Physics
        Fx = thrust_multiplier * self.parameters['thrust_level'] * (-sin(radians(self.parameters['pitch_angle'])))
        Fy = thrust_multiplier * self.parameters['thrust_level'] * (cos(radians(self.parameters['pitch_angle']))) - gravity * self.parameters['mass']

        self.parameters['vx'] += (Fx / self.parameters['mass']) * dt
        self.parameters['vy'] += (Fy / self.parameters['mass']) * dt

        self.parameters['x'] += self.parameters['vx'] * dt
        self.parameters['y'] += self.parameters['vy'] * dt

        self.parameters['altitude'] = (self.parameters['y'] - ground_height - 32)

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
        
        return 0.001 * self.neural_network.DetermineGradient(np.array([[autopilot_target_thrust_level/100.0, autopilot_target_sin_pitch]]).T)

    def AutopilotUpdate(self, dt, landing_target):
        """Retrieve autopilot input and update physics"""

        # Fuel assessment
        self.parameters['fuel'] -= self.parameters['thrust_level'] * dt / 6
        if self.parameters['fuel'] <= 0.0:
            thrust_multiplier = 0.0
            self.parameters['fuel'] = 0.0
        else:
            thrust_multiplier = thrust_power
        self.parameters['mass'] = total_mass - total_fuel + self.parameters['fuel']


        # Determine the condition (phase) the lander is in
        autopilot_input = np.zeros((4, 1))

        target_alt = 20.0
        kx = 50.0 # 60.0    # Spring behavious in horizontal direction
        ky = 110.0  # 110.0 # Spring behavious in vertical direction

        if abs(self.parameters['x'] - landing_target) < 8.0 and abs(self.parameters['altitude'] - target_alt) < 30.0:
            self.parameters['condition'] = "initiated_landing"

        # During the final landing phase
        if self.parameters['condition'] == "initiated_landing":
            target_alt = -5.0
            kx = 20.0 # 60.0    # Spring behavious in horizontal direction
            ky = 225.0  # 110.0 # Spring behavious in vertical direction

        # Determine target values for pitch angle and thrust level
        cx = 2 * sqrt(kx * self.parameters['mass'])
        if thrust_multiplier * self.parameters['thrust_level'] != 0.0:
            target_sin_pitch = (kx*(self.parameters['x'] - landing_target) + cx*self.parameters['vx']) / (thrust_multiplier * self.parameters['thrust_level'])
        else:
            target_sin_pitch = sin(radians(self.parameters['pitch_angle']))

        cy = 2 * sqrt(ky * self.parameters['mass'])
        if thrust_multiplier != 0.0:
            target_thrust_level = (self.parameters['mass'] * gravity - cy * self.parameters['vy'] - ky * (self.parameters['altitude'] - target_alt))\
                                    / (thrust_multiplier * cos(radians(self.parameters['pitch_angle'])))
        else:
            target_thrust_level = 0.0

        # Determine input from target values
        if self.parameters['thrust_level'] < target_thrust_level*1.1:
            autopilot_input[0] = 1
        elif self.parameters['thrust_level'] > target_thrust_level*0.9:
            autopilot_input[1] = 1

        if sin(radians(self.parameters['pitch_angle'])) < target_sin_pitch:
            autopilot_input[2] = 1
        elif sin(radians(self.parameters['pitch_angle'])) > target_sin_pitch:
            autopilot_input[3] = 1
        
        # Input
        if autopilot_input[0]:
            self.parameters['thrust_level'] = min(100, self.parameters['thrust_level'] + throttle_step)
        if autopilot_input[1]:
            self.parameters['thrust_level'] = max(0, self.parameters['thrust_level'] - throttle_step)
        if autopilot_input[2]:
            self.parameters['pitch_angle'] += pitch_step
            if self.parameters['pitch_angle'] > 180.0: self.parameters['pitch_angle'] -= 360
        if autopilot_input[3]:
            self.parameters['pitch_angle'] -= pitch_step
            if self.parameters['pitch_angle'] < -180.0: self.parameters['pitch_angle'] += 360

        # Physics
        Fx = thrust_multiplier * self.parameters['thrust_level'] * (-sin(radians(self.parameters['pitch_angle'])))
        Fy = thrust_multiplier * self.parameters['thrust_level'] * (cos(radians(self.parameters['pitch_angle']))) - gravity * self.parameters['mass']

        self.parameters['vx'] += (Fx / self.parameters['mass']) * dt
        self.parameters['vy'] += (Fy / self.parameters['mass']) * dt

        self.parameters['x'] += self.parameters['vx'] * dt
        self.parameters['y'] += self.parameters['vy'] * dt

        self.parameters['altitude'] = (self.parameters['y'] - ground_height - 32)

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

    def DrawText(self, display_surface, landing_target):
        """Show information on screen"""
        font = pg.font.SysFont("consolas", 20, True)
        text_colour = (237, 125, 58)

        text1 = f"Thrust setting: {self.parameters['thrust_level'] if self.parameters['fuel'] != 0.0 else 0:1.0f}%"
        text1_img = font.render(text1, True, text_colour)
        text1_rect = text1_img.get_rect()
        text1_rect.left = 10
        text1_rect.top = 10

        text2 = f"Pitch angle: {self.parameters['pitch_angle']:1.0f}" + u"\N{DEGREE SIGN}"
        text2_img = font.render(text2, True, text_colour)
        text2_rect = text2_img.get_rect()
        text2_rect.left = 10
        text2_rect.top = text1_rect.bottom

        text3 = f"Altitude: {self.parameters['altitude']:0.01f}"
        text3_img = font.render(text3, True, text_colour)
        text3_rect = text3_img.get_rect()
        text3_rect.left = 10
        text3_rect.top = text2_rect.bottom

        text4 = f"Velocity: {sqrt(self.parameters['vx']**2 + self.parameters['vy']**2):0.01f}"
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

    def DetermineTotalCost(self):
        return self.total_cost

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

def TrainController(landers_per_gen=250, num_of_gens=1000, time_before_skip=25.0):
    dt = 0.05

    old_lander_list = []
    for _ in range(landers_per_gen):
            old_lander_list.append(Lander(False, False))

    for _ in range(num_of_gens):
        lander_list = []
        landing_target = CreateLandingTarget()
        for j in range(landers_per_gen):
            lander_list.append(Lander(False, False, np.random.choice(old_lander_list, p=np.array([1/lander.DetermineTotalCost() for lander in old_lander_list])/sum(1/lander.DetermineTotalCost() for lander in old_lander_list)).neural_network, \
                                                    np.random.choice(old_lander_list, p=np.array([1/lander.DetermineTotalCost() for lander in old_lander_list])/sum(1/lander.DetermineTotalCost() for lander in old_lander_list)).neural_network)) 
        for lander in lander_list:
            lander.SetInitialConditions()

        for k in range(10):
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
                    for lander in lander_list:
                        lander.total_cost += lander.parameters['cost']

        lander_list.sort(key=Lander.DetermineTotalCost)
        print(f"{lander_list[0].DetermineTotalCost():0.01f}")
        old_lander_list = lander_list
        

    NeuralNetwork.SaveNeuralNetwork(lander_list[0].neural_network.parameters)

def RunSimulation(lander_list):
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

    i = 0
    total_dc = np.zeros_like(lander_list[-1].neural_network.parameters)
    while running:
        i += 1
        pg.event.pump()
        keys = pg.key.get_pressed()
        if keys[pg.K_ESCAPE]:                       # QUIT the game
            running = False

        DrawBackground(screen)
        DrawGround(screen)
        DrawTarget(screen, landing_target)

        for lander in lander_list:
            if lander.display_simulation:
                if not (lander.parameters['condition'] == "has_landed" or lander.parameters['condition'] == "has_crashed"):
                    if lander.control_mode == "autopilot":
                        lander.Update(pg.time.get_ticks()/1000 - last_time, landing_target)
                    elif lander.control_mode == "neural_net":
                        total_dc += lander.Update(pg.time.get_ticks()/1000 - last_time, landing_target)
                    elif lander.control_mode == "player":
                        lander.Update(pg.time.get_ticks()/1000 - last_time, keys)
                lander.DrawLander(screen, sprite_list)
                if lander.display_stats:
                    lander.DrawText(screen, landing_target)

        last_time = pg.time.get_ticks()/1000
        pg.display.update()                           # Update the screen
        time.sleep(0.02)

    pg.quit()

    lander_list[-1].neural_network.parameters -= total_dc / i

    NeuralNetwork.SaveNeuralNetwork(lander_list[-1].neural_network.parameters)

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

for i in range(10):
    RunSimulation([Lander("neural_net", True, True)])
