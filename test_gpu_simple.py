#!/usr/bin/env python3
"""
GPU-compatible test for RTX 5070 Ti
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

a = tf.constant(1.0)
