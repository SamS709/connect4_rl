import logging

# Simple one-line logging setup
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s - %(message)s')

from DQN2 import DQN
import numpy as np
from Connect4 import Connect4
from env import Env
import tensorflow as tf
from epsilon import epsilon
from MinMax import MinMax
import multiprocessing


N_LEARNING = 0

class Train(Connect4):

    def __init__(self,model_name="model_name",reset=False,learning_rate=0.5e-3,discount_factor=0.99,softmax_=False,eps = 0.5):
        super().__init__()
        self.dqnP1 = DQN(reset = reset, eps = eps, P1='1',learning_rate=learning_rate,gamma=discount_factor,model_name=model_name,softmax_=softmax_,n_neurons=128,n_layers=3)
        self.dqnP2 = DQN(reset = reset, eps = eps, P1='2',learning_rate=learning_rate,gamma=discount_factor,model_name=model_name,softmax_=softmax_,n_neurons=128,n_layers=3)
        self.env = Env()
        # self.minimax = MinMax()
 
    #replay_buffer.append(state,action,reward,next_state,run)
    def play_one_game(self,eps):

        t1 = False
        t2 = False
        A = self.env.reset()
        first_shot = True
        game_length = 0

        while not t1 and not t2:

            game_length += 1

            a1 = self.dqnP1.epsilon_greedy(A,eps)
            B, r1, t1 = self.env.step(a1,1)
            
            if t1: # if t1 made a move that led to a termination
                t2 = True 
                C = A
                if r1 == 1: # if P1 won
                    r2 = 0
                elif r1 == 0: # if P1 made a forbidden move
                    r2 = 1/7
                elif r1 == 1/7: # if it is a tie
                    r2 = 1/7

            if not first_shot: # replay_beuffer2 starts to append after the first loop
                self.dqnP2.replay_buffer.append((D,a2,r2-r1,B,(1-t1)*(1-t2)))

            if not t1: 
                a2 = self.dqnP2.epsilon_greedy(B,eps)
                C, r2, t2 = self.env.step(a2,2)
                if t2:
                    self.dqnP2.replay_buffer.append((B,a2,r2,B,1-t2)) # accroding to (E0), as t2 = True, the value of the next_state is not important
            
            if t2 and not t1 and r2==0: # if p2 made a forbidden move
                r1 = 1/7 # because we can't tell if the move made by P1 was good or not: we set the target proba for that move to 1/7 (uniform)
            self.dqnP1.replay_buffer.append((A,a1,r1-r2,C,(1-t1)*(1-t2)))                
            first_shot = False

            A = C
            D = B

        # for info in self.dqnP1.replay_buffer:
        #     L_info = list(info)
        #     print(
        #         f"state = \n{self.grid_to_table(L_info[0])}\n action = {L_info[1]}\n reward = {L_info[2]}\n next_state = \n{self.grid_to_table(L_info[3])}\n runs = {L_info[4]}\n\n"
        #     )
        # for info in self.dqnP2.replay_buffer:
        #     L_info = list(info)
        #     print(
        #         f"state = \n{self.grid_to_table(L_info[0])}\n action = {L_info[1]}\n reward = {L_info[2]}\n next_state = \n{self.grid_to_table(L_info[3])}\n runs = {L_info[4]}\n\n"
        #     )

        return game_length


    def train_n_games(self,n):
        game_lengths = []
        game_lengths_mean = []
        n_points = 50
        registration_rate = n/n_points # such that we get n_points points for the final game_length plot
        for episode in range(n):
            eps = epsilon(episode,n,2,eps_0=0.3)
            game_length = self.play_one_game(eps)
            game_lengths.append(game_length)
            print(f"\r[INFO] Episode: {episode + 1} / {n}, Game length: {game_length + 1}, eps: {eps:.3f}",end="")
            if episode >= 10 :
                self.dqnP1.training_step()
                self.dqnP2.training_step()
                if episode % registration_rate == 0:
                    game_lengths_mean.append(np.mean(game_lengths[int(episode-registration_rate):]))
                    self.dqnP1.target.set_weights(self.dqnP1.model.get_weights())
                    self.dqnP2.target.set_weights(self.dqnP2.model.get_weights())
                    self.dqnP1.model.save(self.dqnP1.dir_path,overwrite=True)
                    self.dqnP2.model.save(self.dqnP2.dir_path,overwrite=True)
        return game_lengths_mean
    
    def play_one_game_evaluation(self,j,depth):
        terminated = False
        state = self.env.reset()
        game_length = 0
        win = 0
        tie = 0 
        forbiden_move = 0
        if j == 1 :
            minimax = MinMax(player=2)
            while not terminated:
                game_length += 1
                action = self.dqnP1.epsilon_greedy(state,eps = 0) # best pos according to the DQN
                state, reward_ai, terminated = self.env.step(action,1)
                
                if terminated:
                    if reward_ai == 0:
                        forbiden_move = 1
                    elif reward_ai == 1:
                        win = 1
                    else:
                        tie = 1
                    break
                
                game_length += 1
                state = self.grid_to_table(state)
                action = minimax.best_pos(state, depth)
                state = self.table_to_grid(state)                
                state, reward_minimax, terminated = self.env.step(action,2)
        else:
            minimax = MinMax(player=1)
            while not terminated:
                game_length += 1
                action = self.dqnP2.epsilon_greedy(state,eps = 0) # best pos according to the DQN
                state, reward_ai, terminated = self.env.step(action,2)

                if terminated:
                    if reward_ai == 0:
                        forbiden_move = 1
                    elif reward_ai == 1:
                        win = 1
                    else:
                        tie = 1
                    break

                game_length += 1
                state = self.grid_to_table(state)
                action = minimax.best_pos(state, depth)
                state = self.table_to_grid(state)
                state, reward_minimax, terminated = self.env.step(action,1)
        return game_length, win, tie, forbiden_move





    
    def evaluate_model(self,n_games = 100,depth = 3):
        game_lengths = [[],[]]
        wins = [[],[]]
        ties = [[],[]]
        forbiden_moves = [[],[]]
        for j in range(2):
            print(f"[INFO] Evaluating the model number {j+1} over {n_games} games against minimax algorith (exploration's depth : {depth}).\n")
            for i in range(n_games):
                print(f"\r[{j*n_games+i} / {n_games} games played.",end="")
                game_length, win, tie, forbiden_move = self.play_one_game_evaluation(j+1,depth)
                game_lengths[j].append(game_length)
                wins[j].append(win)
                ties[j].append(tie)
                forbiden_moves[j].append(forbiden_move)
        mean_lengths_1 = np.mean(game_lengths[0])
        mean_lengths_2 = np.mean(game_lengths[1])
        total_wins_1 = np.sum(wins[0])
        total_wins_2 = np.sum(wins[1])
        total_forbiden_moves_1 = np.sum(forbiden_moves[0])
        total_forbiden_moves_2 = np.sum(forbiden_moves[1])

        str_ = f"Model performances when training {n_games} games against Minimax algorith explorating with depth = {depth}\n\n"
        
        str_ += f"   Model1 playing first:\n"
        str_ += f"       Mean length = {mean_lengths_1}\n"
        str_ += f"       Total wins = {total_wins_1} / {n_games} games\n"
        str_ += f"       Total losses = {n_games - (total_wins_1 + total_forbiden_moves_1)} / {n_games} games\n"
        str_ += f"       Total forbiden moves = {total_forbiden_moves_1} / {n_games} games\n\n"

        str_ += f"   Model2 playing second:\n"
        str_ += f"       Mean length = {mean_lengths_2}\n"
        str_ += f"       Total wins = {total_wins_2} / {n_games} games\n"
        str_ += f"       Total losses = {n_games - (total_wins_2 + total_forbiden_moves_2)} / {n_games} games\n"
        str_ += f"       Total forbiden moves = {total_forbiden_moves_2} / {n_games} games\n\n"

        print(str_)
        


if __name__=="__main__":

    train = True
    evaluate = True
    trainer = Train(model_name= "model_128_neurons_3_layers")

    if train :
        import matplotlib.pyplot as plt
        n = 10000
        game_lengths = trainer.train_n_games(n)
        plt.title('Lengths of the game during training')
        plt.plot(game_lengths)
        plt.savefig('plots/training_plot.png', dpi=300, bbox_inches='tight')
        plt.show()

    if evaluate:
        trainer.evaluate_model(n_games = 100, depth=3)


    

