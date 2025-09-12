import os
import logging
# Suppress TensorFlow output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=INFO, 2=WARNING, 3=ERROR
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import random
import keras
import tensorflow as tf
import tensorflow
from collections import deque
from env import Env
from Connect4 import *

#model = dnn1
#target = dnn2

class DQN(Connect4):

    def __init__(self,model_name,softmax_=False,n_layers=2,n_neurons=32,learning_rate=1e-2,gamma=1e-1,eps = 0.9,P1="1",reset = False):

        super().__init__()
        self.env = Env()
        self.game_rewards = []
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.replay_buffer = deque(maxlen=2000)
        self.epsilon = eps
        self.gamma = gamma
        self.P1 = P1
        self.batch_size = 32
        self.loss_fn = keras.losses.mean_squared_error
        self.optimizer = keras.optimizers.Nadam(learning_rate=learning_rate)
        self.model_name = model_name
        self.boost = True # tells the ai if a given position will lead to a lose at next turn
        self.dir_path = "models/"+self.model_name+self.P1+".h5"

        try:
            self.load_model()
        except:
            self.create_model()
        print(self.__str__())

    def __str__(self):
        return self.model.summary()


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
        output_ = keras.layers.Dense(7, activation="softmax")(relu2)
        self.model = keras.models.Model(inputs=[input_], outputs=[output_])
        self.model.save(self.dir_path,overwrite=True)
        self.target = keras.models.load_model(self.dir_path)

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
        
    def play_one_step(self, state ):
        action = self.epsilon_greedy(state)
        next_state, reward, terminated = self.env.step(action)
        self.replay_buffer.append((state, action, reward, next_state,terminated))
        return next_state, reward, terminated
    


    
    
    def sample_experiences(self):
        indexes = np.random.randint(len(self.replay_buffer),size=self.batch_size)
        batch = [self.replay_buffer[index] for index in indexes]
        return [
                np.array([experience[field_index] for experience in batch])
                for field_index in range(5)
            ]    
    
    def training_step(self):
        experiences = self.sample_experiences()
        states, actions, rewards, next_states, runs = experiences
        rewards[rewards<=0]=0
        next_Q_values = self.target.predict(next_states, verbose=0) # using target network tell the target value
        max_next_Q_values = next_Q_values.max(axis=1)
        target_Q_values = rewards + runs * self.gamma * max_next_Q_values
        target_Q_values = target_Q_values.reshape(-1, 1)
        mask = tf.one_hot(actions, 7)
        with tf.GradientTape() as tape:
            all_Q_values = self.model(states)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))


if __name__=="__main__":
    dqn = DQN("model_name",eps=0.5)
    state = np.array([0 for i in range(42)])
    obs = dqn.env.reset()
    dqn.epsilon_greedy(state,0.3)



    """  actions = dqn.model.predict(state[np.newaxis],verbose=0)[0] 

    print(actions)
    print(np.sum(actions))
    print(np.argmax(actions))"""