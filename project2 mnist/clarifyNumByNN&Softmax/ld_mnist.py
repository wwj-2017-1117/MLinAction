import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

def load_digits():
    mnist = input_data.read_data_sets("C:/Users/marsggbo/Documents/Code/ML/TF Tutorial/data/MNIST_data", one_hot=True)
    return mnist