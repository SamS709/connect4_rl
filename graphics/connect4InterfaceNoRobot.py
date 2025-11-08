import os
import logging

# Force TensorFlow to use CPU (RTX 5070 Ti compute capability 12.0 not yet fully supported)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=INFO, 2=WARNING, 3=ERROR
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from kivy.app import App
from kivy.graphics import Line, Color, Rectangle, Ellipse
from kivy.properties import ListProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.core.window import Window
from kivy.uix.widget import Widget
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.switch import Switch
from kivy.uix.screenmanager import Screen

import random as rd
from kivy.lang import Builder
import numpy as np
import sys
import os
Builder.load_file(os.path.join(os.path.dirname(__file__), 'connect4InterfaceNoRobot.kv'))

# Add parent directory and scripts directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'scripts'))

from scripts.Connect4 import Connect4
from scripts.rl_algorithms.DDQN import DDQN
from scripts.env import Env
from graphics.ai_models_interface import MyButton
from global_vars import *



class Grille(BoxLayout):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        #self.add_widget(Button(size_hint=(1,1)))
        self.Lbuttons = [] # list of buttons distributed in columns to capture clicks
        for i in range(7): # button creation
            self.Lbuttons.append(Button(text=str(i),background_color=(0,0,0,0),color=(1,1,1,0)))
            self.add_widget(self.Lbuttons[i])
        #self.add_widget(Button(size_hint=(1, 1)))
        self.LR = []
        self.LC = [[]for j in range(7)]
        with self.canvas.before:
            Color(0, 0, 1, 0.9)
            for i in range(7): # creating a blue background for each button => big blue rectangle that covers the whole page
                self.LR.append(Rectangle(pos=self.Lbuttons[2].pos, size=self.Lbuttons[0].size))
            Color(LIGHT_BLUE[0], LIGHT_BLUE[1], LIGHT_BLUE[2], LIGHT_BLUE[3])
            for i in range(6): # creating black circles on top of the blue rectangle to make 'holes' in the grid
                for j in range(7):
                    self.LC[i].append(Ellipse(pos=(100,100),size=(50,50)))
        
    

class Connect4GameNoRobot(Screen,FloatLayout,Connect4): # Main class for Connect4 game without robot

    def __init__(self, P1='1P', **kwargs):
        super().__init__(**kwargs)
        Window.bind(mouse_pos=self.mouse_pos)
        self.env = Env()
        self.P1 = P1 # game mode choice (made from Mode: takes the value given to Mode)
        self.reset() # initialize the game
        

        
        # This runs before any transition animation
        # Good for: Setting up data, initializing widgets
    

    def on_press_reset(self, instance):
        instance.button_color = DARK_BLUE
    
    def on_release_reset(self, instance):
        instance.button_color = BLUE
        self.reset()

    def reset(self):
        self.model_name = var1.model_name
        self.grid = self.env.reset()
        self.terminated = False
        self.clear_widgets() # remove all widgets from the window
        self.P1 = '2' # by default, the AI plays after
        self.j = 1 # yellow pieces represented by ones
        self.r = 2 # red pieces represented by twos
        self.dqn = DDQN(model_name= self.model_name , softmax_=False,P1=str(self.r),n_neurons=128,n_layers=3) #initializing dqn
        self.player = 'J' # by default, the user starts (user = yellow player)
        self.grille = Grille() # instantiate a new game grid
        self.add_widget(self.grille) # Display the grid
        self.init_C() # Initialize the pieces that will be displayed in the grid
        self.button() # initialize the bound buttons
        self.wpionJ = Widget() # we create the piece of user
        with self.canvas.before:
            Color(0.68, 0.85, 0.90, 1)  # light blue (RGB: 173, 216, 230)
            self.bg_rect = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._update_bg, size=self._update_bg)
        with self.wpionJ.canvas:
            Color(1,1,0,1)
            self.pionJ = Ellipse(pos=(100, 100), size=(50, 50)) # Associate a canvas to the Yellow piece
        self.add_widget(self.wpionJ) 
        self.player_one_button = MyButton(
            text='Play 2nd',
            size_hint=(0.2, 0.08),
            #size=(0.1, 0.1),
            pos_hint={'right': 0.22, 'top': 0.98},
            button_color = RED,
            on_press = self.on_press_player_one,
            on_release = self.on_release_player_one
        )
        self.reset_button = MyButton(
            text='Play again',
            size_hint=(0.2, 0.08),
            #size=(0.1, 0.1),
            pos_hint={'right': 0.98, 'top': 0.98},
            button_color = BLUE,
            on_press = self.on_press_reset,
            on_release = self.on_release_reset
        )
        self.add_widget(self.player_one_button) # Add the button to let the opponent start
        self.add_widget(self.reset_button)        
        self.on_size() # updateing the size of the elements
        self.model_name = var1.model_name
        print('var1 model name = ', var1.model_name)
    
    def on_press_player_one(self, instance):
        if instance.button_color == RED :
            instance.button_color = DARK_RED

    def on_release_player_one(self, instance):
        if instance.button_color == DARK_RED :
            instance.button_color = LIGHT_RED
            self.first_shot = True
            self.player = 'R' #the red player starts
            self.P1 = '1'
            self.j = 2 
            self.r = 1 
            print(self.model_name)
            self.play() #we play the first move of the red player


    def _update_bg(self, *args):
        self.bg_rect.pos = self.pos
        self.bg_rect.size = self.size
        
    def button(self): # creation of 7 vertical buttons capturing clicks
        for i in range(7):
            self.grille.Lbuttons[i].bind(on_release = self.release) # bind what happens when button is pressed
            self.grille.Lbuttons[i].bind(on_press=self.press) # bind what happens when button is released


    def det_action(self): # during a game vs a human, it is expected from the ai not to make forbidden moves.
        state = self.grid
        sorted_actions = self.dqn.get_sorted_actions(state)
        L_free_pos = np.array(super().free_pos_table(self.grid))
        print(sorted_actions)
        for action in sorted_actions: # we take the first valid action sorted by their probability of being taken by the ai
            if action in L_free_pos[:,1:]: 
                return action
    
    def play(self):
        self.player_one_button.button_color = LIGHT_RED
        action = self.det_action()
        next_state, reward, self.terminated = self.env.step(action,self.r)
        self.grid = next_state.copy()
        self.table = self.grid_to_table(self.grid)
        for i in range(self.table.shape[0]):
            for j in range(self.table.shape[1]):
                self.remove_widget(self.LwCR[i][j])
                self.remove_widget(self.LwCJ[i][j])
        for i in range(self.table.shape[0]):
            for j in range(self.table.shape[1]):
                if self.table[i,j] == self.r:
                    self.add_widget(self.LwCR[i][j])
                if self.table[i,j] == self.j:
                    self.add_widget(self.LwCJ[i][j])
        if not self.terminated:
            self.player = 'J'

    def press(self,instance):
        PP = self.free_pos_table(self.grid)
        if self.player == 'J' and not self.terminated:
            for i in range(6):
                for j in range(7):
                    if instance.text == str(j) and [i,j] in PP:
                        self.add_widget(self.LwCJ[i][j])
                        self.remove_widget(self.wpionJ)
    

    def release(self,instance):
        PP = self.free_pos_table(self.grid)
        self.table = self.grid_to_table(self.grid)
        if self.player == 'J' :
            for i in range(6):
                for j in range(7):
                    if instance.text == str(j) and [i,j] in PP:
                        next_state, reward, self.terminated = self.env.step(j,self.j)
                        self.grid = next_state.copy()
                        self.table = self.grid_to_table(self.grid)
                        self.add_widget(self.wpionJ)
                        self.player = 'R'
                        if not self.terminated:
                            self.play()
                        



    def init_C(self,*args): # init the circles
        self.LwCJ = [[] for j in range(7)] # yellow widgets
        self.LwCR = [[] for j in range(7)] # red widgets
        self.LCJ = [[] for j in range(7)] # yellow canvas
        self.LCR = [[] for j in range(7)] # red canvas
        for i in range(6):
            for j in range(7):
                self.LwCJ[i].append(Widget())
                self.LwCR[i].append(Widget())
        for i in range(6):
            for j in range(7):
                with self.LwCJ[i][j].canvas:
                    Color(1, 1, 0,1)
                    self.LCJ[i].append(Ellipse(pos=(100,100),size=(50,50)))
                with self.LwCR[i][j].canvas:
                    Color(1, 0, 0, 1)
                    self.LCR[i].append(Ellipse(pos=(100, 100), size=(50, 50)))



    def on_size(self, *args): # keeps proportions

        W,H= self.width, self.height
        w,h = W/7, 6*H/8
        hh = h/6
        R = min(2 * w / 3, h / 7)

        self.grille.width, self.grille.height =W,H
        self.pionJ.size = R,R

        for i in range(7):
            self.grille.LR[i].pos = i * w,0
            self.grille.LR[i].size = w,h

        for i in range(6):
            for j in range(7):
                self.grille.LC[i][j].size = R,R
                self.LCJ[i][j].size = R, R
                self.LCR[i][j].size = R, R
                self.grille.LC[i][j].pos = j * w + w / 2 - R / 2, i * hh + hh / 2 - R / 2
                self.LCJ[5-i][j].pos = j * w + w / 2 - R / 2, i * hh + hh / 2 - R / 2
                self.LCR[5-i][j].pos = j * w + w / 2 - R / 2, i * hh + hh / 2 - R / 2

    def mouse_pos(self, window, pos): # detects mouse pos, and changes the yellow piece position thanks to it
        self.mousepos = pos
        self.pionJ.pos = pos[0] - self.pionJ.size[0] / 2, self.height - self.pionJ.size[1]


class Game(Screen):
    
    def on_pre_enter(self, *args):
        global MODEL_NAME
        self.action_bar = self.ids.action_bar
        self.game = self.ids.game_game
        self.action_bar.title = "Playing against : " + var1.model_name
        self.model_name = var1.model_name
        self.game.reset()

class graphicsApp(App):
    title = "Connect 4 DQN"

    def build(self):
        return Connect4GameNoRobot()


if __name__ == "__main__":
    os.environ['KIVY_GL_BACKEND'] = 'sdl2'  # Use pure SDL2 backend for better compatibility
    try:
        graphicsApp().run()
    except Exception as e:
        import traceback
        print("Kivy app crashed with error:")
        traceback.print_exc()