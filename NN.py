import numpy as np
from matplotlib import pyplot
import preprocess
 
def feed_forward(input_vector):
    hidden_activation = first_weights @ input_vector # hidden layer activations
    hidden_activation = np.add(first_biases, hidden_activation) # add hidden bias
    # apply sigmoid on each activation
    for j in range(LAYER_SIZE):
        hidden_activation[j,] = sigmoid(hidden_activation[j,])
    output_vector = output_weights @ hidden_activation # output layer activation
    output_vector = np.add(output_biases,  output_vector) # add output bias
    # apply sigmoid on output
    for j in range(len(output_vector)):
        output_vector[j,] = sigmoid(output_vector[j,])
    # return activations
    return hidden_activation, output_vector



def normalize_input(input):
    return input / 255.0
     
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
    while i < (start + batch_length):
        # flatten input value into column vector
        input_vector = (np.array(preprocess.train_data[i]).flatten()).T
        input_vector = normalize_input(input_vector)
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
    return 2 * (activation_hat - activation) * activation * (1 - activation)

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
    while i < (start + batch_length):
        input_vector = (np.array(preprocess.train_data[i]).flatten()).T
        hidden_activation, output_vector = feed_forward(input_vector)
        activation_hat = np.zeros(10,) # vector with 10 digits
        activation_hat[preprocess.train_labels[i],] = 1 # set actual label to 1 (rest will be 0)
        error += get_error(output_vector, activation_hat)
        i += 1
    print("Current average error:: ", error / i)
    if len(batches) == 0:
        batches.append(1)
    else:
        batches.append(batches[-1] + 1)
    costs.append(error / i)
        
# CONSTANTS
TRAINING_DATA_SIZE = preprocess.train_data.shape[0]
ROWS = preprocess.train_data.shape[1]
COLS = preprocess.train_data.shape[2]
LAYER_SIZE = 16
BATCH_LENGTH = 1000
HIDDEN_LAYERS = 1
COST_THRESHOLD  = 0.000001
EPOCHS = 30

# MAIN STARTS HERE
first_weights = np.random.rand(LAYER_SIZE, ROWS * COLS) * 2 - 1
output_weights = np.random.rand(10, LAYER_SIZE) * 2 - 1 # 10 x 16
#first_biases = np.random.rand(LAYER_SIZE,) # 1 x 16
first_biases = np.zeros(LAYER_SIZE,)
# output_biases = np.random.rand(10,) # there are 10 digits
output_biases = np.zeros(10,) # there are 10 digits
i = 0
learning_rate = 0.1
j = 0
batches = []
costs = []
while i < TRAINING_DATA_SIZE:
    cost = train_batch(i, BATCH_LENGTH, learning_rate)
    # print("Cost:: ", cost)
    # test_batch(i, BATCH_LENGTH)
    # print("Learning Rate:: ", learning_rate)    
    # if learning_rate > .00001 and i % 5000 == 0:
    #     learning_rate *= .1
    print("Current index:: ", i)
    i += 1000

    # foo = (cost - COST_THRESHOLD) / COST_THRESHOLD
    # learning_rate = learning_rate * foo

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
        i += 1000
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