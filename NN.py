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
    output_weights_delta = np.zeros((10, LAYER_SIZE))
    output_biases_delta = np.zeros(10,)
    first_weights_delta = np.zeros((LAYER_SIZE, ROWS * COLS))
    first_biases_delta = np.zeros(LAYER_SIZE,)
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
        # delta for output neurons
        output_delta = get_output_delta(output_vector, activation_hat)
        # calculated all deltas in output weights/biases
        output_biases_delta += output_delta
        output_weights_delta += np.outer(output_delta, hidden_activation)
        # calculate hidden layer row delta
        row_delta = output_delta @ output_weights * hidden_activation * (1 - hidden_activation)
        first_biases_delta += row_delta
        first_weights_delta += np.outer(row_delta, input_vector)
        i += 1

    output_weights_delta /= batch_length
    output_biases_delta /= batch_length
    first_weights_delta /= batch_length
    first_biases_delta /= batch_length
    first_weights = first_weights + learning_rate * first_weights_delta
    first_biases = first_biases + learning_rate * first_biases_delta
    output_weights = output_weights + learning_rate * output_weights_delta
    output_biases = output_biases + learning_rate * output_biases_delta

    # TODO feedforward once more (to be changed)
    hidden_activation, output_vector = feed_forward(input_vector)
    activation_hat = np.zeros(10,) # vector with 10 digits
    activation_hat[preprocess.train_labels[i-1],] = 1 # set actual label to 1 (rest will be 0)
    test_batch(start, batch_length)
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

def test_batch(start, batch_length):
    i = 0
    error = 0
    while i < (start + batch_length) and i < TRAINING_DATA_SIZE:
        input_vector = (np.array(preprocess.train_data[i]).flatten()).T / 255
        hidden_activation, output_vector = feed_forward(input_vector)
        activation_hat = np.zeros(10,) # vector with 10 digits
        activation_hat[preprocess.train_labels[i],] = 1 # set actual label to 1 (rest will be 0)
        error += get_error(output_vector, activation_hat)
        i += 1
    if len(batches) == 0:
        batches.append(1)
    else:
        batches.append(batches[-1] + 1)
    costs.append(error / i)
    print("Current average for index: ", i, "error:: ", error / i)
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
EPOCHS = 30

# MAIN STARTS HERE
first_weights = np.random.randn(LAYER_SIZE, ROWS * COLS) * np.sqrt(2. / (ROWS * COLS))
output_weights = np.random.randn(10, LAYER_SIZE) * np.sqrt(2. / (ROWS * COLS)) # 10 x 16
#first_biases = np.random.rand(LAYER_SIZE,) # 1 x 16
first_biases = np.zeros(LAYER_SIZE,)
# output_biases = np.random.rand(10,) # there are 10 digits
output_biases = np.zeros(10,) # there are 10 digits
i = 0
j = 0
batches = []
costs = []
learning_rate_exp = 1
learning_rate = 10 ** -learning_rate_exp
while i < TRAINING_DATA_SIZE:
    train_batch(i, BATCH_LENGTH, learning_rate)
    test_batch(i, BATCH_LENGTH)
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
    if j == 15:
        learning_rate = .01
    if j == 25:
        learning_rate = .001
    i = 0
    while i < TRAINING_DATA_SIZE:
        train_batch(i, BATCH_LENGTH, learning_rate)
        # if learning_rate > .00001:
        #     learning_rate *= .1
        # if (cost < COST_THRESHOLD):
        #     break
        test_batch(i, BATCH_LENGTH)
        i += BATCH_LENGTH
    indices = np.random.permutation(len(preprocess.train_data))
    train_labels = preprocess.train_labels[indices]
    j += 1
    print("current epoch", j)

pyplot.plot(batches, costs)
pyplot.xlabel("Batch")
pyplot.ylabel("Cost")
pyplot.title("Average Cost per 1000 Images")
pyplot.show()

np.save("model.npy", {'first_weights': first_weights,
                      'output_weights': output_weights,
                      'first_biases': first_biases,
                      'output_biases': output_biases})