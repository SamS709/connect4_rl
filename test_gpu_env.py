#!/usr/bin/env python3
"""
Test script for TensorFlow GPU environment
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import tensorflow as tf
import numpy as np

def test_gpu_environment():
    print("=" * 50)
    print("TensorFlow GPU Environment Test")
    print("=" * 50)
    
    # Basic info
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Python version: {os.sys.version.split()[0]}")
    
    # GPU detection
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPUs available: {len(gpus)}")
    
    if gpus:
        print(f"GPU device: {gpus[0]}")
        
        # Test GPU computation
        print("\nTesting GPU computation...")
        with tf.device('/GPU:0'):
            # Create random matrices
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            
            # Matrix multiplication
            c = tf.matmul(a, b)
            
        print(f"Matrix multiplication successful!")
        print(f"Result shape: {c.shape}")
        print(f"Computation device: {c.device}")
        
        # Test neural network creation
        print("\nTesting neural network creation...")
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(42,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(7, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        print("Neural network created successfully!")
        print(f"Model parameters: {model.count_params()}")
        
    else:
        print("No GPU detected - will use CPU")
    
    print("\n" + "=" * 50)
    print("Environment test completed!")
    print("=" * 50)

if __name__ == "__main__":
    test_gpu_environment()
