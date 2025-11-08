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

class A2C(Connect4):

    def __init__(self, model_name, softmax_=False, n_layers=2, n_neurons=32, learning_rate=1e-3, gamma=0.99, eps=0.9, P1="1", reset=False, use_prioritized=True, alpha=0.6, beta_start=0.4, beta_frames=100000):

        super().__init__()
        self.game_rewards = []
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        
        
        self.replay_buffer = ReplayBuffer(1000)
        
        self.epsilon = eps
        self.gamma = gamma
        self.P1 = P1
        self.target_update_frequency = 10  # Update target network every 10 training steps
        self.training_step_counter = 0
        self.actor_optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate=learning_rate*2)  # Critic learns faster  
        self.model_name = model_name
        self.actor_dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", self.model_name, self.model_name+self.P1+".h5")
        self.critic_dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", self.model_name, self.model_name+self.P1+"_critic.h5")

        try:
            print("SUCCESSFULLY LOADED MODEL ", self.model_name)
            
            self.load_model()
        except:
            print("SUCCESSFULLY CREATED MODEL ", self.model_name)
            self.create_model()
        print(self.__str__())
        self.logger = Logger(model_name,P1)
        

    def __str__(self):
        return self.actor.summary()
    
    def get_algo_name(self):
        return "A2C"
    
    def save(self):
        self.actor.save(self.actor_dir_path,overwrite=True)
        self.critic_target.set_weights(self.critic.get_weights())
        self.critic.save(self.critic_dir_path, overwrite=True)

    def create_actor_model(self):
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
        output_ = keras.layers.Dense(7, activation="softmax")(relu2)
        self.actor = keras.models.Model(inputs=[input_], outputs=[output_])
        self.actor.save(self.actor_dir_path, overwrite=True)
        
    def create_critic_model(self):
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
        output_ = keras.layers.Dense(1, activation="linear")(relu2)
        self.critic = keras.models.Model(inputs=[input_], outputs=[output_])
        self.critic.save(self.critic_dir_path, overwrite=True)
        
    def create_model(self):
        self.create_actor_model()
        self.create_critic_model()
        self.load_model()
        

    def load_model(self):
        self.actor = keras.models.load_model(self.actor_dir_path)
        self.critic = keras.models.load_model(self.critic_dir_path)
        self.critic_target = keras.models.load_model(self.critic_dir_path)
        self.critic_target.set_weights(self.critic.get_weights())

    def select_action(self, state, eps = None):
        # Use the model in inference mode (training=False by default with predict)
        probabilities = self.actor(tf.constant(state[np.newaxis], dtype=tf.float32), training=False).numpy()[0]
        # Sample action according to probabilities (introduces randomness naturally)
        action = np.random.choice(7, p=probabilities)
        return action
    
    def get_sorted_actions(self, state):
        """Returns actions (indices) sorted by Q-values in descending order"""
        Q_values = self.actor.predict(state[np.newaxis],verbose=0)[0]
        # argsort gives indices in ascending order, so we reverse with [::-1]
        return np.argsort(Q_values)[::-1]
    
    def get_action_probabilities(self, state):
        """Returns Q-values (probabilities) and corresponding sorted actions"""
        Q_values = self.actor.predict(state[np.newaxis],verbose=0)[0]
        sorted_actions = np.argsort(Q_values)[::-1]
        sorted_probabilities = Q_values[sorted_actions]
        return sorted_actions, sorted_probabilities

    
    def sample_experiences(self):
        experiences = self.replay_buffer.get_exp()
        return experiences, None, None  # No indices/weights for uniform
    
    def training_step(self):
        
        # Check if we have enough experiences to train
        # A2C uses ALL experiences in buffer, so we need a substantial batch
        # 100 experiences ≈ 3-5 games, provides stable gradient estimates
        if len(self.replay_buffer) < 100:  
            return
        
        experiences, indices, weights = self.sample_experiences()
        states, actions, rewards, next_states, runs = experiences
        
        # Ensure actions are 1D (flatten if needed)
        actions = np.array(actions).flatten().astype(np.int32)
        
        # Reshape rewards and runs to be column vectors
        rewards = np.array(rewards).reshape(-1, 1).astype(np.float32)
        runs = np.array(runs).reshape(-1, 1).astype(np.float32)
        
        # Convert states to tensors
        states_tensor = tf.constant(states, dtype=tf.float32)
        next_states_tensor = tf.constant(next_states, dtype=tf.float32)
        
        # Compute V(s') using target network (in inference mode)
        V_s_next = self.critic_target(next_states_tensor, training=False).numpy()
        
        # Compute advantages and targets
        V_s_current = self.critic(states_tensor, training=False).numpy()
        advantages = rewards + runs * self.gamma * V_s_next - V_s_current
        V_targets = rewards + runs * self.gamma * V_s_next
        
        # Convert to tensors for gradient computation
        advantages_tensor = tf.constant(advantages, dtype=tf.float32)
        V_targets_tensor = tf.constant(V_targets, dtype=tf.float32)
        
        # Update Actor (Policy)
        mask = tf.one_hot(actions, 7)  # Shape: [batch_size, 7]
        with tf.GradientTape() as tape_actor:
            all_probas = self.actor(states_tensor, training=True)  # Shape: [batch_size, 7]
            probas = tf.reduce_sum(all_probas * mask, axis=1, keepdims=True)  # Shape: [batch_size, 1]
            # Policy gradient loss: maximize log probability weighted by advantage
            # We add negative sign because we're minimizing loss (gradient descent)
            actor_loss = tf.reduce_mean(-tf.math.log(probas + 1e-10) * advantages_tensor)
        # Apply actor gradients
        
        grads = tape_actor.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
        
        # Update Critic (Value function)
        with tf.GradientTape() as tape_critic:
            V_s = self.critic(states_tensor, training=True)  
            critic_loss = tf.reduce_mean(tf.square(V_targets_tensor - V_s))
        # Apply critic gradients
        grads = tape_critic.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))
        
        # Clear buffer AFTER training (on-policy: experiences are now outdated)
        self.replay_buffer.empty_buffer()


if __name__=="__main__":
    a2c = A2C("test_a2c", eps=0.5)
    state = np.array([0 for i in range(42)])
    obs = a2c.env.reset()
    action = a2c.select_action(state)