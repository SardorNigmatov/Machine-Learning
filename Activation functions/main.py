import numpy as np
import tensorflow as tf

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

a = np.array([-20,-1.0,0.0,1.0,20])
b = sigmoid(a)

print(b)


def softmax(x):
    e_x = np.exp(x-max(x))
    return e_x / e_x.sum()

a = np.array([-20,-1.0,0.0,1.0,20])
b = softmax(a)
print(b)


def relu(x):
    return max(0,x)

foo = tf.constant([-10,-5,0,5,10],dtype = tf.float32)
y = tf.keras.activations.relu(foo).numpy()
y