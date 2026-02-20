import torch
from scripts.Connect4 import Connect4

class Env(Connect4):
    
    def __init__(self):
        super().__init__()
        self.reset()
        self.state = self.table
        self.action_space = [6,7]

    def step(self,action,player):
          L_free_pos = torch.tensor(super().free_pos_table(self.table))
          if action in L_free_pos[:,1:]: # if the action is valid (not out of the board)
               for pos in L_free_pos:
                    if pos[1]==action:
                         p,q = pos
               terminated = False
               reward = 0
               self.table[p,q] = player
               #reward = super().score(table)/10
               if super().win(self.table,player):
                    reward = 10
                    terminated = True
               elif super().lose(self.table,player):
                    reward = -10
                    terminated = True
               elif super().tie(self.table):
                    reward = 0
                    terminated = True
          else: # if the action is invalid: taking an invalid action is worst than losing from the DQN POV
               terminated = True 
               reward = -15
          self.state = self.table
          return self.table, reward, terminated
    
    def reset(self):
         self.table = torch.tensor([[0 for j in range(7)] for i in range(6)])
         return torch.tensor([[0 for j in range(7)] for i in range(6)])
    
    def render(self):
         print(self.state)
    
if __name__=="__main__":

     L_free_pos = torch.tensor([[0,1],[2,3]])
     print(L_free_pos[:,1:])
     print(1 in L_free_pos[:,1:])