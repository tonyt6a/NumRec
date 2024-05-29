import numpy as np
from matplotlib import pyplot
import preprocess

class Model:
    def __init__(self, first_weights, second_weights, output_weights, first_biases, second_biases, output_biases):
        self.first_weights = first_weights
        self.second_weights = second_weights
        self.output_weights = output_weights
        self.first_biases = first_biases
        self.second_biases = second_biases
        self.output_biases = output_biases

 
def feed_forward(input_vector, model):
    first_hidden_activation = model.first_weights @ input_vector # hidden layer activations
    first_hidden_activation = np.add(first_hidden_activation, model.first_biases) # add hidden bias
    # apply sigmoid on each activation
    first_hidden_activation = sigmoid(first_hidden_activation)
    second_hidden_activation = model.second_weights @ first_hidden_activation + model.second_biases
    second_hidden_activation = sigmoid(second_hidden_activation)
    output_vector = model.output_weights @ second_hidden_activation # output layer activation
    output_vector = np.add(output_vector, model.output_biases) # add output bias
    # apply sigmoid on output
    output_vector = sigmoid(output_vector)
    # return activations
    return first_hidden_activation, second_hidden_activation, output_vector


def train_batch(start, batch_length, learning_rate, model, train_data, train_labels):
    """ 
    test_batch tests batches and performs backpropogation.
    :param start: starting index of starting batch
    :param batch_length: length of batch length
    """
    # # use global variables
    # global first_weights
    # global first_biases
    # global output_weights
    # global output_biases
    i = start
    # make delta matrices
    output_weights_delta = np.zeros((10, LAYER_SIZE))
    output_biases_delta = np.zeros(10,)
    first_weights_delta = np.zeros((LAYER_SIZE, ROWS * COLS))
    first_biases_delta = np.zeros(LAYER_SIZE,)
    second_weights_delta = np.zeros((LAYER_SIZE, LAYER_SIZE))
    second_biases_delta = np.zeros(LAYER_SIZE,)
    # start on batch
    while i < (start + batch_length) and i < TRAINING_DATA_SIZE:
        # flatten input value into column vector
        input_vector = ((np.array(train_data[i]).flatten()).reshape(-1)) / 255
        # feed forward and get activations for input vector
        first_hidden_activation, second_hidden_activation, output_vector = feed_forward(input_vector, model)
        # Make expected vector
        activation_hat = np.zeros(10,) # vector with 10 digits
        activation_hat[train_labels[i],] = 1 # set actual label to 1 (rest will be 0)
        # delta for output neurons
        output_delta = get_output_delta(output_vector, activation_hat)
        # calculated all deltas in output weights/biases
        output_biases_delta += output_delta
        output_weights_delta += np.outer(output_delta, second_hidden_activation)
        # calculate hidden layer row delta
        row_delta = output_delta @ model.output_weights * second_hidden_activation * (1 - second_hidden_activation)
        second_biases_delta += row_delta
        second_weights_delta += np.outer(row_delta, first_hidden_activation)
        
        first_row_delta = row_delta @ model.second_weights * first_hidden_activation * (1 - first_hidden_activation)
        first_biases_delta += first_row_delta
        first_weights_delta += np.outer(first_row_delta, input_vector)
        i += 1

    output_weights_delta /= batch_length
    output_biases_delta /= batch_length
    first_weights_delta /= batch_length
    first_biases_delta /= batch_length
    second_weights_delta /= batch_length
    second_biases_delta /= batch_length
    model.first_weights = model.first_weights - learning_rate * first_weights_delta
    model.first_biases = model.first_biases - learning_rate * first_biases_delta
    model.output_weights = model.output_weights - learning_rate * output_weights_delta
    model.output_biases = model.output_biases - learning_rate * output_biases_delta
    model.second_weights = model.second_weights - learning_rate * second_weights_delta
    model.second_biases = model.second_biases - learning_rate * second_biases_delta



def get_softmax_function(z):
    beta = 1.0
    return np.exp(beta * z) / np.sum(np.exp(beta * z)) 
    
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

def test_batch(start, batch_length, model,train_data, train_labels):
    i = 0
    error = 0
    while i < (start + batch_length) and i < TRAINING_DATA_SIZE:
        input_vector = (np.array(train_data[i]).flatten().reshape(-1)) / 255
        first_hidden_activation, second_hidden_activation, output_vector = feed_forward(input_vector, model)
        activation_hat = np.zeros(10,) # vector with 10 digits
        activation_hat[train_labels[i],] = 1 # set actual label to 1 (rest will be 0)
        error += get_error(output_vector, activation_hat)
        i += 1
    # if len(batches) == 0:
    #     batches.append(1)
    # else:
    #     batches.append(batches[-1] + 1)
    # costs.append(error / i)
    print("Current average for index: ", i, "error:: ", error / i)
    # pyplot.plot(error_list, label='Training Error')
    # if start % 3200 == 0:
        # pyplot.xlabel('Batch Number')
        # pyplot.ylabel('Error')
        # pyplot.title('Training and Testing Error over Time')
        # pyplot.show()
    return error / i
        
# CONSTANTS
# ROWS = preprocess.train_data.shape[1]
# COLS = preprocess.train_data.shape[2]
ROWS = 28
COLS = 28
TRAINING_DATA_SIZE = preprocess.train_data.shape[0]
LAYER_SIZE = 16
BATCH_LENGTH = 64
HIDDEN_LAYERS = 2
COST_THRESHOLD  = 0.000001
EPOCHS = 30

# MAIN
if __name__ == "__main__":
    
    model = Model(first_weights=np.random.randn(LAYER_SIZE, ROWS * COLS) * np.sqrt(2. / (ROWS * COLS)), # 16 x 784
                  second_weights=np.random.randn(LAYER_SIZE, LAYER_SIZE) * np.sqrt(2. / (ROWS * COLS)),
                  output_weights=np.random.randn(10, LAYER_SIZE) * np.sqrt(2. / (ROWS * COLS)), # 10 x 16
                  first_biases=np.zeros(LAYER_SIZE,), # 16 x 1
                  second_biases=np.zeros(LAYER_SIZE), # 16 x 1
                  output_biases=np.zeros(10,)) # 10 x 1
    # there are 10 digits
    i = 0
    j = 0
    batches = []
    costs = []
    learning_rate = 1
    for k in range(int(TRAINING_DATA_SIZE / 1000)):
        print("Real index", k)
        while i < 1000:
            train_batch(i, BATCH_LENGTH, learning_rate, model, preprocess.train_data, preprocess.train_labels)
            test_batch(i, BATCH_LENGTH, model, preprocess.train_data, preprocess.train_labels)
            i += BATCH_LENGTH
        train_data = np.concatenate((preprocess.train_data[1000:], preprocess.train_data[:1000]))
        train_labels = np.concatenate((preprocess.train_labels[1000:], preprocess.train_labels[:1000]))
        i = 0
    indices = np.random.permutation(len(preprocess.train_data))
    train_data = preprocess.train_data[indices]
    train_labels = preprocess.train_labels[indices]
    train_data = np.ascontiguousarray(train_data)
    train_labels = np.ascontiguousarray(train_labels)
    i = 0
    while j < EPOCHS:
        i = 0
        for k in range(int(TRAINING_DATA_SIZE / 1000)):
            print("Real index", k)
            while i < 1000:
                train_batch(i, BATCH_LENGTH, learning_rate, model, train_data, train_labels)
                test_batch(i, BATCH_LENGTH, model, train_data, train_labels)
                i += BATCH_LENGTH
            train_data = np.concatenate((train_data[1000:], train_data[:1000]))
            train_labels = np.concatenate((train_labels[1000:], train_labels[:1000]))
            i = 0
        indices = np.random.permutation(len(preprocess.train_data))
        train_data = preprocess.train_data[indices]
        train_labels = preprocess.train_labels[indices]
        train_data = np.ascontiguousarray(train_data)
        train_labels = np.ascontiguousarray(train_labels)
        j += 1
        print("current epoch", j)

    pyplot.plot(batches, costs)
    pyplot.xlabel("Batch")
    pyplot.ylabel("Cost")
    pyplot.title("Average Cost per 1000 Images")
    pyplot.show()

    np.save("model.npy", {'first_weights': model.first_weights,
                          'second_weights': model.second_weights,
                        'output_weights': model.output_weights,
                        'first_biases': model.first_biases,
                        'second_biases': model.second_biases,
                        'output_biases': model.output_biases})