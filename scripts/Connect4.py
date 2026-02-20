import torch 

class Connect4:

    def __init__(self):
        pass

    def win(self,table,player): # tells if the state of the table a win for the player (1 = player 1, 2 = player 2)
        # Horizontal positions
        for i in range(table.shape[0]):
            for j in range(4):
                h = table[i,j] == player and table[i,j + 1] == player and table[i,j + 2] == player and table[i,j + 3] == player
                if h:
                    return True

        # Vertical positions
        for j in range(table.shape[1]):
            for i in range(3):
                v = table[i,j] == player and table[i + 1,j] == player and table[i + 2,j] == player and table[i + 3,j] == player
                if v:
                    return True

        # Diagonal positions
        for i in range(3):
            for j in range(4):
                d1 = table[i,j] == player and table[i + 1,j + 1] == player and table[i + 2,j + 2] == player and table[i + 3,j + 3] == player
                d2 = table[i,j + 3] == player and table[i + 1,j + 2] == player and table[i + 2,j + 1] == player and table[i + 3,j] == player
                if d1 or d2:
                    return True
        #Autre
        return False


    def lose(self,table,player): # tells if the state of the table a lose for the player (1 = player 1, 2 = player 2)
        if player == 1:
            player = 2
        else:
            player = 1
        # Horizontal positions
        for i in range(table.shape[0]):
            for j in range(4):
                h = table[i,j] == player and table[i,j + 1] == player and table[i,j + 2] == player and table[i,j + 3] == player
                if h:
                    return True

        # Vertical positions
        for j in range(table.shape[1]):
            for i in range(3):
                v = table[i,j] == player and table[i + 1,j] == player and table[i + 2,j] == player and table[i + 3,j] == player
                if v:
                    return True

        # Diagonal positions
        for i in range(3):
            for j in range(4):
                d1 = table[i,j] == player and table[i + 1,j + 1] == player and table[i + 2,j + 2] == player and table[i + 3,j + 3] == player
                d2 = table[i,j + 3] == player and table[i + 1,j + 2] == player and table[i + 2,j + 1] == player and table[i + 3,j] == player
                if d1 or d2:
                    return True
        #Autre
        return False

    def tie(self,table): # tells if the state of the table a tie
        a = True
        if self.win(table,1) or self.lose(table,1):
            a = False
        else:
            a = table.all()
        return a

    def end(self,table): # tells if the state of the table an end
        if self.win(table):
            return True
        if self.lose(table):
            return True
        if self.tie(table):
            return True
        return False

    def grid_to_table(self,grid): # convert the grid representation (1d array) to the table representation (2d array) of the game
        return grid.reshape((7,6)).T

    def table_to_grid(self,table): # reverses the operation made by grid_t_table so that table_to_grid(grid_to_table(grid))=grid
        return table.T.ravel()
    
    def free_pos_table(self,table): # tells authorized moves in a given table
        L = []
        for j in range(table.shape[1]):
            i = 5
            while table[i, j] != 0 and i >= 0:
                i = i - 1
            if table[i, j] == 0:
                L.append([i, j])
        return L

    def free_pos(self, grid): # tells authorized moves in a given grid
        table = self.grid_to_table(grid)
        L = []
        Lpos = []
        for j in range(len(table[0])):
            i = 5
            while table[i,j] != 0 and i >= 0:
                i = i - 1
            if table[i,j] == 0:
                L.append([i, j])
        for i in range(len(L)):
            Lpos.append(L[i][0]+6*L[i][1])

        return torch.tensor(Lpos)
    
    def free_not_losing_pos_grid(self, grid):
        Not_losing_pos_grid = []
        Not_losing_pos = self.free_not_losing_pos(grid)
        for i in range(len(Not_losing_pos)):
            Not_losing_pos_grid.append(Not_losing_pos[i][0]+6*Not_losing_pos[i][1]) 
        return torch.tensor(Not_losing_pos_grid)
    
    def free_not_losing_pos(self, grid):
        Not_losing_pos = []
        table0 = self.grid_to_table(grid)
        L1 = self.free_pos_table(table0)
        if len(L1)!=0:
            for pos1 in L1:
                table1 = table0.clone()
                p1, q1 = pos1[0],pos1[1]
                table1[p1, q1] = 1
                L2 = self.free_pos_table(table1)
                lose = False
                if len(L2) != 0:
                    for pos2 in L2:
                        p2, q2 = pos2[0], pos2[1]
                        table2 = table1.clone()
                        table2[p2, q2] = 2
                        grid2 = self.table_to_grid(table2)
                        if self.lose(grid2) :
                            lose = True
                    if lose == False:
                        Not_losing_pos.append([p1, q1])
        return Not_losing_pos

    def avaible_pos_graphics(self,grid):
        table = self.grid_to_table(grid)
        if grid.shape[0]==6:
            table = grid.clone()
        L = []
        for j in range(table.shape[1]):
            i = 5
            while table[i, j] != 0 and i >= 0:
                i = i - 1
            if table[i, j] == 0:
                L.append([i, j])
        return L

    def count_lines(self,table,Nlines,n): # returns the number of lines having N consecutive pieces for player n

        if Nlines == 2:

            S2 = 0 #nombre de lignes à 2 éléms

            # Horizontal positions
            for i in range(table.shape[0]):
                for j in range(6):
                    h = table[i,j] == n and table[i,j+1] == n
                    if h:
                        S2 = S2 + 1

            # Vertical positions
            for j in range(table.shape[1]):
                for i in range(5):
                    v = table[i,j] == n and table[i+1,j] == n
                    if v:
                        S2 = S2 + 1

            #Diagonal positions
            for i in range(table.shape[0]-1):
                for j in range(table.shape[1]-1):
                    d1 = table[i,j] == n and table[i+1,j+1] == n
                    d2 = table[i+1,j] == n and table[i,j+1] == n
                    if d1:
                        S2 = S2+1
                    if d2:
                        S2 = S2+1
            return S2

        if Nlines == 3:

            S3 = 0  # nombre de lignes à 3 éléms

            # Horizontal positions
            for i in range(table.shape[0]):
                for j in range(table.shape[1]-2):
                    h = table[i,j] == n and table[i,j+1] == n and table[i,j+2] == n
                    if h:
                        S3 = S3 + 1

            # Vertical positions
            for j in range(table.shape[1]):
                for i in range(table.shape[0]-2):
                    v = table[i,j] == n and table[i+1,j] == n and table[i+2,j] == n
                    if v:
                        S3 = S3 + 1

            # Diagonal positions
            for i in range(table.shape[0] - 2):
                for j in range(table.shape[1] - 2):
                    d1 = table[i,j] == n and table[i+1,j+1] == n and table[i+2,j+2]==n
                    d2 = table[i+2,j] == n and table[i+1,j+1] == n and table[i,j+2]==n
                    if d1:
                        S3 = S3 + 1
                    if d2:
                        S3 = S3 + 1
            return S3

    def score(self,table): # returns the score of a given board (this is just a proposal to evaluate the board, a better one can surely be found)
        S1N3 = self.count_lines(table,Nlines=3,n=1)
        S1N2 = self.count_lines(table,Nlines=2,n=1) - 2*S1N3
        S2N3 = self.count_lines(table,Nlines=3,n=2)
        S2N2 = self.count_lines(table,Nlines=2,n=2) - 2*S2N3
        S = S1N3 + 0.3*S1N2 - (S2N3 + 0.3*S2N2)
        return S
    
    def posgrid_to_postable(self,pos):
        j = pos//6
        i = pos-6*j
        return[i,j]

if __name__=='__main__':
    connect4 = Connect4()
    table = torch.tensor([
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,2,0,0,0,0,0],
        [0,2,0,0,0,0,0],
        [0,2,0,0,0,0,0],
        [0,2,0,0,0,0,0]
        ])
    
    grid = connect4.table_to_grid(table)
    print(grid)
    win = connect4.win(table,2)
    print(win)