import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Force TensorFlow to use CPU (RTX 5070 Ti compute capability 12.0 not yet fully supported)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=INFO, 2=WARNING, 3=ERROR
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import random
import keras
import tensorflow as tf
import numpy as np
from collections import deque
from scripts.env import Env
from scripts.Connect4 import *
from scripts.logger import Logger
from scripts.ReplayBuffer import ReplayBuffer, PrioritizedReplayBuffer

#model = dnn1
#target = dnn2

class DDQN(Connect4):

    def __init__(self, model_name, softmax_=False, n_layers=2, n_neurons=32, learning_rate=1e-2, gamma=1e-1, eps=0.9, P1="1", reset=False, use_prioritized=True, alpha=0.6, beta_start=0.4, beta_frames=100000):

        super().__init__()
        self.game_rewards = []
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        
        # Initialize replay buffer (prioritized or standard)
        if use_prioritized:
            self.replay_buffer = PrioritizedReplayBuffer(
                maxlen=2000,
                alpha=alpha,
                beta_start=beta_start,
                beta_frames=beta_frames
            )
            self.use_prioritized = True
        else:
            self.replay_buffer = ReplayBuffer(2000)
            self.use_prioritized = False
        
        self.epsilon = eps
        self.gamma = gamma
        self.P1 = P1
        self.batch_size = 32
        self.loss_fn = keras.losses.mean_squared_error
        self.optimizer = keras.optimizers.Nadam(learning_rate=learning_rate)
        self.model_name = model_name
        self.dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", self.model_name, self.model_name+self.P1+".h5")

        try:
            print("SUCCESSFULLY LOADED MODEL ", self.model_name)
            
            self.load_model()
        except:
            print("SUCCESSFULLY CREATED MODEL ", self.model_name)
            self.create_model()
            self.save()
        print(self.__str__())
        self.logger = Logger(model_name,P1)
        

    def __str__(self):
        return self.model.summary()
    
    def get_algo_name(self):
        return "DDQN"

    def save(self):
        self.target.set_weights(self.model.get_weights())
        self.model.save(self.dir_path,overwrite=True)
        

    def create_model(self):
        input_ = keras.layers.Input(shape=(42,))
        one_hot = keras.layers.Lambda(lambda x : __import__("tensorflow").one_hot(__import__("tensorflow").cast(x, 'int64'), 3),output_shape=(42,3))(input_)
        flatten = keras.layers.Flatten(input_shape=(42, 3))(one_hot)
        hidden1 = keras.layers.Dense(self.n_neurons,kernel_initializer="he_normal", use_bias=False)(flatten)
        BN1 = keras.layers.BatchNormalization()(hidden1)
        relu1 = keras.layers.Activation("relu")(BN1)
        dropout =keras.layers.Dropout(rate=0.2)(relu1)
        for i in range(self.n_layers-2):
            hidden = keras.layers.Dense(self.n_neurons,kernel_initializer="he_normal", use_bias=False)(dropout)
            BN = keras.layers.BatchNormalization()(hidden)
            relu = keras.layers.Activation("relu")(BN)
            dropout =keras.layers.Dropout(rate=0.2)(relu)
        hidden2 = keras.layers.Dense(self.n_neurons,kernel_initializer="he_normal", use_bias=False)(dropout)
        BN2 = keras.layers.BatchNormalization()(hidden2)
        relu2 = keras.layers.Activation("relu")(BN2)
        output_ = keras.layers.Dense(7, activation="linear")(relu2)
        self.model = keras.models.Model(inputs=[input_], outputs=[output_])
        self.model.save(self.dir_path,overwrite=True)
        self.load_model()
        
    def load_model(self):
        self.model = keras.models.load_model(self.dir_path)
        self.target = keras.models.load_model(self.dir_path)
        self.target.set_weights(self.model.get_weights())


    def epsilon_greedy(self, state, eps = None):
        epsilon = self.epsilon if eps==None else eps
        if np.random.rand()<epsilon: 
            return np.random.randint(0,7)
        else:
            Q_values = self.model.predict(state[np.newaxis],verbose=0)[0] 
            #np.newaxis augment la dimension de state
            #verbose = 0 => don't show the progress bar of evaluating
            return np.argmax(Q_values)
    
    def select_action(self, state, eps = None):
        return self.epsilon_greedy(state,eps)
    
    def get_sorted_actions(self, state):
        """Returns actions (indices) sorted by Q-values in descending order"""
        Q_values = self.model.predict(state[np.newaxis],verbose=0)[0]
        # argsort gives indices in ascending order, so we reverse with [::-1]
        return np.argsort(Q_values)[::-1]
    
    def get_action_probabilities(self, state):
        """Returns Q-values (probabilities) and corresponding sorted actions"""
        Q_values = self.model.predict(state[np.newaxis],verbose=0)[0]
        sorted_actions = np.argsort(Q_values)[::-1]
        sorted_probabilities = Q_values[sorted_actions]
        return sorted_actions, sorted_probabilities

    
    def sample_experiences(self):
        if self.use_prioritized:
            # Prioritized sampling returns: experiences, indices, weights
            experiences, indices, weights = self.replay_buffer.sample(self.batch_size)
            return experiences, indices, weights
        else:
            # Standard uniform sampling
            experiences = self.replay_buffer.sample(self.batch_size)
            return experiences, None, None  # No indices/weights for uniform
    
    def training_step(self):
        
        experiences, indices, weights = self.sample_experiences()
        states, actions, rewards, next_states, runs = experiences
        
        # Compute Q-values and targets
        next_actions = np.argmax(self.model.predict(next_states, verbose=0), axis=1)
        next_Q_values = self.target.predict(next_states, verbose=0)  # using target network
        max_next_Q_values = next_Q_values[np.arange(self.batch_size), next_actions]
        target_Q_values = rewards + runs * self.gamma * max_next_Q_values
        target_Q_values = target_Q_values.reshape(-1, 1)
        
        states_tensor = tf.constant(states, dtype=tf.float32)
        target_Q_values_tensor = tf.constant(target_Q_values, dtype=tf.float32)
        
        mask = tf.one_hot(actions, 7)
        
        with tf.GradientTape() as tape:
            all_Q_values = self.model(states_tensor)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            
            # Compute TD errors for priority updates
            td_errors = target_Q_values_tensor - Q_values
            
            # Apply importance sampling weights if using prioritized replay
            if self.use_prioritized:
                weights_tf = tf.constant(weights.reshape(-1, 1), dtype=tf.float32)
                weighted_loss = weights_tf * self.loss_fn(target_Q_values_tensor, Q_values)
                loss = tf.reduce_mean(weighted_loss)
            else:
                loss = tf.reduce_mean(self.loss_fn(target_Q_values_tensor, Q_values))
        
        # Update network
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        # Update priorities in replay buffer
        if self.use_prioritized:
            td_errors_np = np.abs(td_errors.numpy().flatten())
            self.replay_buffer.update_priorities(indices, td_errors_np)


if __name__=="__main__":
    dqn = DDQN("helloooooo",eps=0.5)
    state = np.array([0 for i in range(42)])
    obs = dqn.env.reset()
    dqn.select_action(state,0.3)



    """  actions = dqn.model.predict(state[np.newaxis],verbose=0)[0] 

    print(actions)
    print(np.sum(actions))
    print(np.argmax(actions))"""