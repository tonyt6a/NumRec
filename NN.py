import numpy as np
from matplotlib import pyplot
import preprocess
 
def feed_forward(input_vector):
    hidden_activation = first_weights @ input_vector # hidden layer activations
    hidden_activation = np.add(hidden_activation, first_biases) # add hidden bias
    # apply sigmoid on each activation
    hidden_activation = sigmoid(hidden_activation)
    output_vector = output_weights @ hidden_activation # output layer activation
    output_vector = np.add(output_vector, output_biases) # add output bias
    # apply sigmoid on output
    output_vector = sigmoid(output_vector)
    # return activations
    return hidden_activation, output_vector


     
def train_batch(start, batch_length, learning_rate):
    """ 
    test_batch tests batches and performs backpropogation.
    :param start: starting index of starting batch
    :param batch_length: length of batch length
    """
    # use global variables
    global first_weights
    global first_biases
    global output_weights
    global output_biases
    i = start
    # make delta matrices
    output_weights_change = np.zeros((10, LAYER_SIZE))
    output_biases_change = np.zeros(10,)
    first_weights_change = np.zeros((LAYER_SIZE, ROWS * COLS))
    first_biases_change = np.zeros(LAYER_SIZE,)
    # start on batch
    while i < (start + batch_length) and i < TRAINING_DATA_SIZE:
        # flatten input value into column vector
        input_vector = ((np.array(preprocess.train_data[i]).flatten()).T) / 255
        norm = np.linalg.norm(input_vector)
        # feed forward and get activations for input vector
        hidden_activation, output_vector = feed_forward(input_vector)
        # Make expected vector
        activation_hat = np.zeros(10,) # vector with 10 digits
        activation_hat[preprocess.train_labels[i],] = 1 # set actual label to 1 (rest will be 0)
        # instantiate deltas for output neurons
        output_delta = []
        # summation for all deltas in output weights/biases
        # iterating through neurons of output layer 0 - 9
        for j in range(output_weights_change.shape[0]):
            # getting delta from neuron in output layer using activation and expected
            # add delta for hidden layer delta
            output_delta += [get_output_delta(output_vector[j], activation_hat[j])]
            # getting bias delta
            output_biases_change[j,] += output_delta[j]
            # iterating through columns of a row in output_weights_delta
            for k in range(output_weights_change.shape[1]):
                output_weights_change[j,k] += output_delta[j] * hidden_activation[k,] # changed
        # summation for all deltas in hidden layer 0 - LAYER_SIZE
        for j in range(first_weights_change.shape[0]):
            # solve each delta per row
            row_delta = 0
            # get delta for current layer using output layer
            for k in range(output_weights.shape[0]):
                row_delta += output_delta[k] * output_weights[k, j]
            row_delta *= hidden_activation[j, ] * (1 - hidden_activation[j,])
            # add delta to bias
            first_biases_change[j,] += row_delta
            # iterating through all weights affecting a hidden layer neuron 0 - ROWS * COLS (784)
            for k in range(first_weights_change.shape[1]):
                first_weights_change[j,k] += row_delta * input_vector[k,] # changed
        i += 1

    output_weights_change /= batch_length
    output_biases_change /= batch_length
    first_weights_change /= batch_length
    first_biases_change /= batch_length
    first_weights = first_weights - learning_rate * first_weights_change
    first_biases = first_biases + learning_rate * first_biases_change
    output_weights = output_weights - learning_rate * output_weights_change
    output_biases = output_biases + learning_rate * output_biases_change

    # TODO feedforward once more (to be changed)
    hidden_activation, output_vector = feed_forward(input_vector)
    activation_hat = np.zeros(10,) # vector with 10 digits
    activation_hat[preprocess.train_labels[i-1],] = 1 # set actual label to 1 (rest will be 0)
    return get_error(output_vector, activation_hat)



    
def get_output_delta(activation, activation_hat):
    return (activation - activation_hat) * activation * (1 - activation)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sigmoid(x)(1 - sigmoid(x))
    
def get_error(actual, expected):
    error = 0
    for i in range(len(actual)):
        error += (actual[i,] - expected[i,]) ** 2
    return error   

def test_batch(start, batch_length, error_list):
    i = 0
    error = 0
    while i < (start + batch_length) and i < TRAINING_DATA_SIZE:
        input_vector = (np.array(preprocess.train_data[i]).flatten()).T / 255
        hidden_activation, output_vector = feed_forward(input_vector)
        activation_hat = np.zeros(10,) # vector with 10 digits
        activation_hat[preprocess.train_labels[i-1],] = 1 # set actual label to 1 (rest will be 0)
        error += get_error(output_vector, activation_hat)
        i += 1
    print("Current average for index: ", i, "error:: ", error / i)
    error_list.append(error / i)
    # pyplot.plot(error_list, label='Training Error')
    # if start % 3200 == 0:
        # pyplot.xlabel('Batch Number')
        # pyplot.ylabel('Error')
        # pyplot.title('Training and Testing Error over Time')
        # pyplot.show()
    return error / i
        
# CONSTANTS
TRAINING_DATA_SIZE = preprocess.train_data.shape[0]
ROWS = preprocess.train_data.shape[1]
COLS = preprocess.train_data.shape[2]
LAYER_SIZE = 16
BATCH_LENGTH = 64
HIDDEN_LAYERS = 1
COST_THRESHOLD  = 0.000001
EPOCHS = 15

# MAIN STARTS HERE
first_weights = np.random.randn(LAYER_SIZE, ROWS * COLS) * np.sqrt(2. / (ROWS * COLS))
output_weights = np.random.randn(10, LAYER_SIZE) * np.sqrt(2. / (ROWS * COLS)) # 10 x 16
#first_biases = np.random.rand(LAYER_SIZE,) # 1 x 16
first_biases = np.zeros(LAYER_SIZE,)
# output_biases = np.random.rand(10,) # there are 10 digits
output_biases = np.zeros(10,) # there are 10 digits
i = 0
learning_rate_exp = 1
learning_rate = 10 ** -learning_rate_exp
j = 0
error = []
# ru
while i < TRAINING_DATA_SIZE:
    train_batch(i, BATCH_LENGTH, learning_rate)
    test_batch(i, BATCH_LENGTH, error)
    # if i % 3200 == 0:
    #     print("Current index:: ", i)
    # i += BATCH_LENGTH 

    # foo = (cost - COST_THRESHOLD) / COST_THRESHOLD
    # learning_rate = learning_rate * foo
indices = np.random.permutation(len(preprocess.train_data))

train_data = preprocess.train_data[indices]
train_labels = preprocess.train_labels[indices]
i = 0
while j < EPOCHS:
    while i < TRAINING_DATA_SIZE:
        train_batch(i, BATCH_LENGTH, learning_rate)  
        # print("Current index:: ", i)
        # if i % 3200 == 0:
        #      test_batch(i, BATCH_LENGTH, error)
        #      if error[int(i / BATCH_LENGTH)] * (10 ** learning_rate_exp) < 1:
        #          learning_rate_exp += 1
        #          learning_rate = 10 ** -learning_rate_exp
        i += BATCH_LENGTH
    indices = np.random.permutation(len(preprocess.train_data))
    train_data = preprocess.train_data[indices]
    train_labels = preprocess.train_labels[indices]
    i = 0
    j += 1
    print("current epoch", j)


