import numpy as np
import preprocess

TRAINING_DATA_SIZE = preprocess.train_data.shape[0]
ROWS = preprocess.train_data.shape[1]
COLS = preprocess.train_data.shape[2]
LAYER_SIZE = 16

first_weights = np.random.rand(LAYER_SIZE, ROWS * COLS) * 2 - 1
output_weights = np.random.rand(10, LAYER_SIZE) * 2 - 1 # 10 x 16
#first_biases = np.random.rand(LAYER_SIZE,) # 1 x 16
first_biases = np.zeros(LAYER_SIZE,)
# output_biases = np.random.rand(10,) # there are 10 digits
output_biases = np.zeros(10,) # there are 10 digits

np.save("model_test.npy", {'first_weights':first_weights, 
                           'output_weights':output_weights, 
                           'first_biases':first_biases,
                            'output_biases':output_biases})