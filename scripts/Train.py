import os
import sys
import logging

# Add project root to path to enable proper imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force TensorFlow to use CPU (RTX 5070 Ti compute capability 12.0 not yet fully supported)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

# Simple one-line logging setup
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s - %(message)s')
import importlib
import torch
from kivy.clock import mainthread
from scripts.rl_algorithms.DDQN import DDQN
import numpy as np
from scripts.Connect4 import Connect4
from scripts.env import Env
from scripts.logger import Logger
import tensorflow as tf
from scripts.epsilon import epsilon
from scripts.MinMax import MinMax
import multiprocessing
import matplotlib.pyplot as plt


N_LEARNING = 0

class Train(Connect4):

    def __init__(self,model_name,info_label = None,scrollable_lablel = None,box = None,pb = None,reset=False,learning_rate=5e-3, discount_factor=0.98,softmax_=False,eps = 0.5):
        super().__init__()
        self.load_models(reset, eps, learning_rate, discount_factor,model_name, softmax_)
        self.log_algo_name()    
        self.env = Env()
        self.model_name = model_name
        # self.minimax = MinMax()
        self.kivy = True
        
        self.info_label = info_label        
        self.scrollable_lablel = scrollable_lablel
        self.box = box
        self.pb = pb

        if self.info_label == None:
            self.kivy = False
        
        
    def load_models(self, reset, eps, learning_rate, discount_factor,model_name, softmax_):
        self.logger_ = Logger(model_name, "1")
        model_algo = self.logger_.get_model_algo()
        module = importlib.import_module(f"scripts.rl_algorithms.{model_algo}")
        AlgorithmClass = getattr(module, model_algo)
        self.agentP1 = AlgorithmClass(reset = reset, use_prioritized= True, eps = eps, P1='1',learning_rate=learning_rate,gamma=discount_factor,model_name=model_name,softmax_=softmax_,n_neurons=128,n_layers=3)
        self.agentP2 = AlgorithmClass(reset = reset, use_prioritized= True, eps = eps, P1='2',learning_rate=learning_rate,gamma=discount_factor,model_name=model_name,softmax_=softmax_,n_neurons=128,n_layers=3)
        self.logger1 = self.agentP1.logger
        self.logger2 = self.agentP2.logger 
        
    def log_algo_name(self):
        algo_name = "unknown"
        algo_name = self.agentP1.get_algo_name()
        self.logger1.set_algo_name(algo_name)
        self.logger2.set_algo_name(algo_name)
        
    @mainthread
    def modif_label(self,i,N=1):
        if N==1:
            self.info_label.text = "Nom du modèle: " + str(self.model_name) + "\nNombre d'epoques: " + str(i+1) + " / "+str(self.N)
            self.pb.value = (i+1)/self.N*self.pb.max
        if N==2:
            self.scrollable_lablel.layout.remove_widget(self.box)


 
    #replay_buffer.append(state,action,reward,next_state,run)
    def play_one_game(self,eps):

        t1 = False
        t2 = False
        A = self.env.reset()
        first_shot = True
        game_length = 0

        while not t1 and not t2:

            game_length += 1

            a1 = self.agentP1.select_action(A,eps)
            B, r1, t1 = self.env.step(a1,1)
            
                        
            if t1: # if t1 made a move that led to a termination
                t2 = True 
                C = A
                if r1 == 10: # if P1 won
                    r2 = -10
                    
            if not first_shot: # replay_beuffer2 starts to append after the first loop
                self.agentP2.replay_buffer.append((D,a2,r2,B,(1-t2)))

            if not t1: 
                a2 = self.agentP2.select_action(B,eps)
                C, r2, t2 = self.env.step(a2,2)
                if t2: # if p2 finished the game (wins or tie or forbidden move)
                    self.agentP2.replay_buffer.append((B,a2,r2,B,1-t2)) # accroding to (E0), as t2 = True, the value of the next_state is not important
            
            if t2 and not t1: 
                if r2 == 10: # if p2 won
                    r1 = -10
            self.agentP1.replay_buffer.append((A,a1,r1,C,(1-t1)*(1-t2)))                
            first_shot = False
            # print(B,r1, t1)
            # print(C, r2, t2)

            A = C
            D = B
        
        return game_length


    def train_n_games(self,n):
        starting_epoch = self.logger1.get_current_infos()[0]
        self.N = n
        game_lengths = []
        game_lengths_mean = []
        n_points = 50
        n_evals = 10
        registration_rate = 100 # such that we get n_points points for the final game_length plot
        n_backprop = 10
        evals_rate = 1000
        lossp1, lossp2 = 0.0, 0.0
        for episode in range(n):
            self.modif_label(episode,N=1) if self.kivy else None
            eps = epsilon(episode,n,2,eps_0=0.3)
            game_length = self.play_one_game(eps)
            game_lengths.append(game_length)
            if episode%100 == 0 and episode > 1:
                print(f"\r[INFO] Episode: {starting_epoch + episode + 1} / {n + starting_epoch}, Game length: {game_length + 1}, eps: {eps:.3f}, lossp1: {lossp1}", end="")
                self.logger1.write_loss(lossp1, episode)
                self.logger2.write_loss(lossp2, episode)
            if episode >= 10 :
                lossp1, lossp2 = 0.0, 0.0
                for i in range(n_backprop):
                    lossp1 += self.agentP1.training_step()
                    lossp2 += self.agentP2.training_step()
                lossp1 /= n_backprop
                lossp2 /= n_backprop
                if episode % registration_rate == 0:
                    print("[INFO] Model registered")
                    self.agentP1.save()
                    self.agentP2.save()
                    game_lengths_mean.append(np.mean(game_lengths[int(episode-registration_rate):]))
                    self.logger1.overwrite_epochs(starting_epoch + episode)
                    self.logger2.overwrite_epochs(starting_epoch + episode)
                    
                if episode % evals_rate == 0:
                    self.add_evals_infos(10, 3)
                    
        self.modif_label(i=n,N=2)
        return game_lengths_mean
    
    def play_one_game_evaluation(self,j,depth):
        self.agentP1.model.eval()
        self.agentP2.model.eval()
        with torch.no_grad():
            terminated = False
            state = self.env.reset()
            game_length = 0
            win = 0
            tie = 0 
            forbiden_move = 0
            if j == 1 :
                minimax = MinMax(player=2)
                while not terminated:
                    # print(state)                    
                    game_length += 1
                    action = self.agentP1.select_action(state,eps = 0)
                    state, reward_ai, terminated = self.env.step(action,1)
                    
                    if terminated:
                        if reward_ai == -15:
                            forbiden_move = 1
                        elif reward_ai == 10:
                            win = 1
                        else:  # reward_ai == 0, it's a tie
                            tie = 1
                        break
                    
                    game_length += 1
                    action = minimax.best_pos(state, depth)
                    state, reward_minimax, terminated = self.env.step(action,2)
                    if terminated:  # Minimax ended the game
                        if reward_minimax == 0:  # Tie
                            tie = 1
                        # else: reward_minimax == 10, minimax won (agent lost)
            else:
                minimax = MinMax(player=1)
                while not terminated:
                    # print(state)
                    game_length += 1
                    action = minimax.best_pos(state, depth)
                    state, reward_minimax, terminated = self.env.step(action,1)
                    
                    if terminated:  # Minimax ended the game
                        if reward_minimax == 0:  # Tie
                            tie = 1
                        # else: reward_minimax == 10, minimax won (agent lost)
                        break
                    
                    game_length += 1
                    action = self.agentP2.select_action(state,eps = 0)
                    state, reward_ai, terminated = self.env.step(action,2)
                    if terminated:
                        if reward_ai == -15:
                            forbiden_move = 1
                        elif reward_ai == 10:
                            win = 1
                        else:  # reward_ai == 0, it's a tie
                            tie = 1
                    
        return game_length, win, tie, forbiden_move


    
    def evaluate_model(self,n_games = 10,depth = 3):
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
        
        win_rate1, forbidden_rate1, lose_rate1 = total_wins_1/n_games, total_forbiden_moves_1/n_games, (n_games - (total_wins_1 + total_forbiden_moves_1)) / n_games
        win_rate2, forbidden_rate2, lose_rate2 = total_wins_2/n_games, total_forbiden_moves_2/n_games, (n_games - (total_wins_2 + total_forbiden_moves_2)) / n_games
        
        return win_rate1, forbidden_rate1, lose_rate1, win_rate2, forbidden_rate2, lose_rate2 

    def add_evals_infos(self,n_games, depth):
        
        win_rates1 = []
        forbidden_rates1 = []
        lose_rates1 = []
        
        win_rates2 = []
        forbidden_rates2 = []
        lose_rates2 = []
        
        for d in range(depth):
            win_rate1, forbidden_rate1, lose_rate1, win_rate2, forbidden_rate2, lose_rate2 = self.evaluate_model(n_games, d)
            win_rates1.append(win_rate1)
            forbidden_rates1.append(forbidden_rate1)
            lose_rates1.append(lose_rate1)
            win_rates2.append(win_rate2)
            forbidden_rates2.append(forbidden_rate2)
            lose_rates2.append(lose_rate2)
            
        self.logger1.add_new_evaluation(n_games, win_rates1, forbidden_rates1, lose_rates1)
        self.logger2.add_new_evaluation(n_games, win_rates2, forbidden_rates2, lose_rates2)

        
        
            

    def plot_current_perfs(self):
        total_epochs1, evaluation_epochs1, win_rates1, forbidden_rates1, lose_rates1 = self.logger1.get_current_infos()
        total_epochs2, evaluation_epochs2, win_rates2, forbidden_rates2, lose_rates2 = self.logger2.get_current_infos()
        
        # Check if we have data
        if len(win_rates1) == 0 or len(win_rates1[0]) == 0:
            print("[WARNING] No evaluation data available for plotting")
            return
        
        # Get number of depths
        n_depths = len(win_rates1[0])
        
        # Create one figure for Model 1 with subplots for each depth
        fig1, axes1 = plt.subplots(1, n_depths, figsize=(7 * n_depths, 6))
        if n_depths == 1:
            axes1 = [axes1]  # Make it iterable if only one subplot
        
        fig1.suptitle(f'{self.model_name}1 - Performance vs Minimax', fontsize=16)
        
        for depth_idx in range(n_depths):
            win_rate_depth1 = [wr[depth_idx] for wr in win_rates1]
            forbidden_rate_depth1 = [fr[depth_idx] for fr in forbidden_rates1]
            lose_rate_depth1 = [lr[depth_idx] for lr in lose_rates1]
            
            axes1[depth_idx].set_title(f'Depth {depth_idx}')
            axes1[depth_idx].set_xlabel('Training Epochs')
            axes1[depth_idx].set_ylabel('Rate')
            axes1[depth_idx].grid(True, alpha=0.3)
            
            axes1[depth_idx].plot(evaluation_epochs1, win_rate_depth1, marker='o', label='Win Rate', linewidth=2, color='green')
            axes1[depth_idx].plot(evaluation_epochs1, forbidden_rate_depth1, marker='s', label='Forbidden Move Rate', linewidth=2, color='orange')
            axes1[depth_idx].plot(evaluation_epochs1, lose_rate_depth1, marker='^', label='Loss Rate', linewidth=2, color='red')
            
            axes1[depth_idx].legend(loc='best')
            axes1[depth_idx].set_ylim([0, 1])
        
        plt.tight_layout()
        # plt.savefig(f'plots/{self.model_name}1_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create one figure for Model 2 with subplots for each depth
        fig2, axes2 = plt.subplots(1, n_depths, figsize=(7 * n_depths, 6))
        if n_depths == 1:
            axes2 = [axes2]  # Make it iterable if only one subplot
        
        fig2.suptitle(f'{self.model_name}2 - Performance vs Minimax', fontsize=16)
        
        for depth_idx in range(n_depths):
            win_rate_depth2 = [wr[depth_idx] for wr in win_rates2]
            forbidden_rate_depth2 = [fr[depth_idx] for fr in forbidden_rates2]
            lose_rate_depth2 = [lr[depth_idx] for lr in lose_rates2]
            
            axes2[depth_idx].set_title(f'Depth {depth_idx}')
            axes2[depth_idx].set_xlabel('Training Epochs')
            axes2[depth_idx].set_ylabel('Rate')
            axes2[depth_idx].grid(True, alpha=0.3)
            
            axes2[depth_idx].plot(evaluation_epochs2, win_rate_depth2, marker='o', label='Win Rate', linewidth=2, color='green')
            axes2[depth_idx].plot(evaluation_epochs2, forbidden_rate_depth2, marker='s', label='Forbidden Move Rate', linewidth=2, color='orange')
            axes2[depth_idx].plot(evaluation_epochs2, lose_rate_depth2, marker='^', label='Loss Rate', linewidth=2, color='red')
            
            axes2[depth_idx].legend(loc='best')
            axes2[depth_idx].set_ylim([0, 1])
        
        plt.tight_layout()
        # plt.savefig(f'plots/{self.model_name}2_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        


if __name__=="__main__":
    
    

    train = True
    plot = False
    trainer = Train(model_name= "torch_ddqn")

    if train :
        import matplotlib.pyplot as plt
        n = 100000
        trainer.train_n_games(n)
        # plt.plot(game_lengths)
        # plt.savefig('plots/training_plot.png', dpi=300, bbox_inches='tight')
        # plt.show()

    if plot:
        trainer.plot_current_perfs()

    

