# import input_data
import tensorflow as tf
import matplotlib.pyplot as plt

minst = tf.keras.datasets.minst # dataset of hand-written digits from 0-9

# learnign parameters /hyper param
learn_rate = 0.01 # how fast the weights are being updated , 
                  # too fast and it might skip the optimal solution ,
                  # too slow might mean too many iterations,
                  # think stockfish and how the AI is capped to x speed at lower levels
training_iteration = 30
batch_size = 100
display_step = 2

(x_train , y_train) , (x_test, y_test) = minst.load_data()

print(x_train[0])

