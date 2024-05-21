import numpy as np
from matplotlib import pyplot
import preprocess
 
def feed_forward(input_vector):
    hidden_activation = first_weights @ input_vector
    hidden_activation = np.add(first_biases, hidden_activation)
    for j in range(LAYER_SIZE):
        hidden_activation[j,] = sigmoid(hidden_activation[j,])
    output_vector = output_weights @ hidden_activation
    output_vector = np.add(output_biases,  output_vector)
    for j in range(len(output_vector)):
        output_vector[j,] = sigmoid(output_vector[j,])
    return hidden_activation, output_vector


def train_batch(start, batch_length, learning_rate):
    """ 
    test_batch tests batches and performs backpropogation.
    :param start: starting index of starting batch
    :param batch_length: length of batch length
    """
    i = start
    output_weights_delta = np.zeros(10, LAYER_SIZE)
    output_biases_delta = np.zeros(10,)
    first_weights_delta = np.zeros(LAYER_SIZE, ROWS * COLS)
    first_biases_delta = np.zeros(LAYER_SIZE,)
    while i < (start + batch_length):
        input_vector = (np.array(preprocess.train_data[i]).flatten()).T
        hidden_activation, output_vector = feed_forward(input_vector)
        activation_hat = np.zeros(10,) # vector with 10 digits
        activation_hat[preprocess.train_labels[i],] = 1 # set actual label to 1 (rest will be 0)
        # summation for all deltas in output weights/biases
        for j in range(len(output_weights_delta)):
            row_delta = get_output_delta(output_vector[j], activation_hat[j])
             # getting bias delta
            output_biases_delta[j] += row_delta
            # getting weights delta
            for k in range(len(output_weights_delta[j])):
                output_weights_delta[j,k] += row_delta * hidden_activation[k,] # changed
        # summation for all deltas in hidden layer
        for j in range(len(first_weights_delta)):
            #solve each delta per row
            row_delta = 0
            output_delta = get_output_delta(output_vector[j], activation_hat[j])
            # get delta for current layer using output layer
            for k in range(len(output_weights)):
                row_delta += output_delta * output_weights[k, j] * hidden_activation[j, ] * (1 - hidden_activation[j,]) # change
            first_biases_delta[j] += row_delta
            for k in range(len(first_weights_delta[j])):
                first_weights_delta[j,k] += row_delta * first_weights[j,k] # change
        i += 1
    output_weights_delta /= batch_length
    output_biases_delta /= batch_length
    first_weights_delta /= batch_length
    first_biases_delta /= batch_length
    first_weights = first_weights - learning_rate * first_weights_delta
    first_biases = first_biases - learning_rate * first_biases_delta
    output_weights = output_weights - learning_rate * output_weights_delta
    output_biases = output_biases - learning_rate * output_biases_delta

    # TODO feedforward once more (to be changed)
    hidden_activation, output_vector = feed_forward(input_vector)
    activation_hat = np.zeros(10,) # vector with 10 digits
    activation_hat[preprocess.train_labels[i],] = 1 # set actual label to 1 (rest will be 0)
    return get_error(output_vector, activation_hat)



def get_error(actual, expected):
    error = 0
    for i in range(len(actual)):
        error += (actual[i,] - expected[i,]) ** 2
    return error
    
def get_output_delta(activation, activation_hat):
    return 2 * (activation_hat - activation) * activation * (1 - activation)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sigmoid(x)(1 - sigmoid(x))

# CONSTANTS
TRAINING_DATA_SIZE = preprocess.train_data.shape[0]
ROWS = preprocess.train_data.shape[1]
COLS = preprocess.train_data.shape[2]
LAYER_SIZE = 16
BATCH_LENGTH = 1000
HIDDEN_LAYERS = 1
COST_THRESHOLD  = 0.5

# MAIN STARTS HERE
first_weights = np.random.rand(LAYER_SIZE, ROWS * COLS) * 2 - 1
output_weights = np.random.rand(10, LAYER_SIZE) * 2 - 1 # 10 x 16
#first_biases = np.random.rand(LAYER_SIZE,) # 1 x 16
first_biases = np.zeros(LAYER_SIZE,)
# output_biases = np.random.rand(10,) # there are 10 digits
output_biases = np.zeros(10,) # there are 10 digits
i = 0
learning_rate = 0.1
while i < TRAINING_DATA_SIZE:
    cost = train_batch(i, BATCH_LENGTH, learning_rate)
    learning_rate *= .5 

    foo = (cost - COST_THRESHOLD) / COST_THRESHOLD
    learning_rate = learning_rate * 
    



    i += 1000
i = 0
while True:
    cost = train_batch(i, BATCH_LENGTH, learning_rate)
    learning_rate *= .5
    if (cost < COST_THRESHOLD):
        break
    i += 1000
    if(i > TRAINING_DATA_SIZE):
        i = 0
