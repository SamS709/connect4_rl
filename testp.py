import tensorflow as tf
import keras
import keras.backend as K
from Connect4 import Connect4
import numpy as np
from collections import deque

array1 = np.array([0 for i in range(42)])
array2 = np.array([0 for i in range(42)])
q_vals = np.array([0.2,0.1,0.3,0.4])
print(np.argsort(q_vals)[::-1])

