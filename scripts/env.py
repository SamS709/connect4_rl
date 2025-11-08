import numpy as np
from scripts.Connect4 import Connect4

class Env(Connect4):
    
    def __init__(self):
        super().__init__()
        self.grid = np.array([0 for i in range(42)])
        self.state = self.grid
        self.action_space = [6,7]

    def step(self,action,player):
          L_free_pos = np.array(super().free_pos_table(self.grid))
          if action in L_free_pos[:,1:]: # if the action is valid (not out of the board)
               for pos in L_free_pos:
                    if pos[1]==action:
                         p,q = pos
               terminated = False
               reward = 0
               table = super().grid_to_table(self.grid)
               table[p,q] = player
               next_grid = super().table_to_grid(table)
               self.grid = next_grid.copy()
               #reward = super().score(table)/10
               if super().win(next_grid,player):
                    reward = 10
                    terminated = True
               elif super().lose(next_grid,player):
                    reward = -10
                    terminated = True
               elif super().tie(next_grid):
                    reward = 0
                    terminated = True
          else: # if the action is invalid: taking an invalid action is worst than losing from the DQN POV
               terminated = True 
               reward = -15
               next_grid = self.grid.copy()
          return next_grid, reward, terminated
    
    def reset(self):
         self.grid = np.array([0 for i in range(42)])
         return np.array([0 for i in range(42)])
    
    def render(self):
         print(self.grid_to_table(self.grid))
    
if __name__=="__main__":

     L_free_pos = np.array([[0,1],[2,3]])
     print(L_free_pos[:,1:])
     print(1 in L_free_pos[:,1:])