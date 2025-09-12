import random

from Connect4 import Connect4
import numpy as np

# Minimax algorithm is a tree searching algorithm that determines the best move to play

# In the case of Connect4, this algorith is boosted with alpha-beta pruning that deletes decrease the number of nodes that are evaluated by the minimax algorithm in its search tree

class MinMax(Connect4):

    def __init__(self,player):
        self.player = player # the one who wants the best move
        self.opponent = 1 if player == 2 else 2 # the opponent to the player (minimax algorithm miinimize the opponent's max reward)
        super().__init__()

    def minmax(self,table,depth,alpha,beta,n): # returns the score of a board
        # either an end is detected
        # either the exploration's depth is reached and the table is evaluated according to the score function
        grid = super().table_to_grid(table)
        if super().win(grid,self.player):
            return 1000
        if super().lose(grid,self.player):
            return -1000
        if super().tie(grid):
            return 0
        if depth == 0:
            return super().score(table)

        if n == self.player:
            a = -10000
            Lpos = super().avaible_pos_graphics(table)
            for pos in Lpos:
                TABLE = table.copy()
                TABLE[pos[0],pos[1]]=self.player
                val = self.minmax(TABLE,depth-1,alpha,beta,self.opponent)
                if val>=a:
                    a=val
                if a > beta:
                    return a
                alpha = max(alpha,val)
            return a

        else:
            b = 10000
            Lpos = super().avaible_pos_graphics(table)
            for pos in Lpos:
                TABLE = table.copy()
                TABLE[pos[0], pos[1]] = self.opponent
                val = self.minmax(TABLE, depth - 1, alpha, beta, self.player)
                if val <= b:
                    b = val
                if b < alpha:
                    return b
            return b

    def best_pos(self,table,depth): # returns the best move to play among all possible moves
        Lpos = self.avaible_pos_graphics(table)
        a = -10000
        Leqpos = []
        for pos in Lpos:
            TABLE = table.copy()
            TABLE[pos[0],pos[1]]=self.player
            val= self.minmax(TABLE,depth,-10000,10000,self.opponent)
            if val>a:
                a=val
                Leqpos = [pos]
            elif val == a:
                Leqpos.append(pos)
        POS = random.choice(Leqpos)
        return POS[1]

if __name__ == "__main__":
    table = np.array([
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,1,2,0,0,0],
        [1,0,1,2,0,2,2],
    ])
    minimax = MinMax(player=2)
    print(minimax.best_pos(table,3))

