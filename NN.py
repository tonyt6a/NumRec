import numpy as np
from matplotlib import pyplot
import preprocess
import torch
import torch.autograd.profiler as profiler


class Model:
    def __init__(self, first_weights, second_weights, output_weights, first_biases, second_biases, output_biases):
        self.first_weights = first_weights.to("cuda")
        self.second_weights = second_weights.to("cuda")
        self.output_weights = output_weights.to("cuda")
        self.first_biases = first_biases.to("cuda")
        self.second_biases = second_biases.to("cuda")
        self.output_biases = output_biases.to("cuda")

 
def feed_forward(input_vector, model):
    first_hidden_activation = torch.matmul(model.first_weights, input_vector) # hidden layer activations
    first_hidden_activation = torch.add(first_hidden_activation, model.first_biases) # add hidden bias
    # apply sigmoid on each activation
    first_hidden_activation = sigmoid(first_hidden_activation)
    second_hidden_activation = torch.matmul(model.second_weights, first_hidden_activation) + model.second_biases
    second_hidden_activation = sigmoid(second_hidden_activation)
    output_vector = model.output_weights @ second_hidden_activation # output layer activation
    output_vector = torch.add(output_vector, model.output_biases) # add output bias
    # apply sigmoid on output
    output_vector = sigmoid(output_vector)
    # return activations
    return first_hidden_activation, second_hidden_activation, output_vector


def train_batch(start, batch_length, learning_rate, model, train_data, train_labels, device):
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
    output_weights_delta = torch.zeros((10, LAYER_SIZE), device=device)
    output_biases_delta = torch.zeros((10, 1), device=device)
    first_weights_delta = torch.zeros((LAYER_SIZE, ROWS * COLS), device=device)
    first_biases_delta = torch.zeros((LAYER_SIZE,1), device=device)
    second_weights_delta = torch.zeros((LAYER_SIZE, LAYER_SIZE), device=device)
    second_biases_delta = torch.zeros((LAYER_SIZE, 1), device=device)
    # start on batch
    while i < (start + batch_length) and i < TRAINING_DATA_SIZE:
        # flatten input value into column vector
        input_vector = ((torch.tensor(train_data[i], device=device).flatten().view(-1,1))) / 255
        # feed forward and get activations for input vector
        first_hidden_activation, second_hidden_activation, output_vector = feed_forward(input_vector, model)
        # Make expected vector
        activation_hat = torch.zeros((10, 1), device=device) # vector with 10 digits
        activation_hat[train_labels[i],] = 1 # set actual label to 1 (rest will be 0)
        # delta for output neurons
        output_delta = get_output_delta(output_vector, activation_hat)
        # calculated all deltas in output weights/biases
        output_biases_delta += output_delta
        output_weights_delta += torch.outer(output_delta.squeeze(), second_hidden_activation.squeeze())
        # calculate hidden layer row delta
        row_delta = torch.matmul(output_delta.view(1,10), model.output_weights).view(16,1) * second_hidden_activation * (1 - second_hidden_activation)
        second_biases_delta += row_delta
        second_weights_delta += torch.outer(row_delta.squeeze(), first_hidden_activation.squeeze())
        
        first_row_delta = torch.matmul(row_delta.view(1,16), model.second_weights).view(16,1) * first_hidden_activation * (1 - first_hidden_activation)
        first_biases_delta += first_row_delta
        first_weights_delta += torch.outer(first_row_delta.squeeze(), input_vector.squeeze())
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
    return torch.exp(beta * z) / torch.sum(torch.exp(beta * z)) 
    
def get_output_delta(activation, activation_hat):
    return (activation - activation_hat) * activation * (1 - activation)

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

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
        input_vector = (torch.tensor(train_data[i], device='cuda').flatten().view(-1,1)) / 255
        first_hidden_activation, second_hidden_activation, output_vector = feed_forward(input_vector, model)
        activation_hat = torch.zeros((10,1),device='cuda') # vector with 10 digits
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
BATCH_LENGTH = 1000
HIDDEN_LAYERS = 1
COST_THRESHOLD  = 0.000001
EPOCHS = 30

device = 'cuda'
# MAIN
if __name__ == "__main__":
    
    with profiler.profile(use_cuda=True) as prof:
        print("a")
        model = Model(first_weights=torch.randn(LAYER_SIZE, ROWS * COLS, device=device) * np.sqrt(2. / (ROWS * COLS)), # 16 x 784
                    second_weights=torch.randn(LAYER_SIZE, LAYER_SIZE, device=device) * np.sqrt(2. / (ROWS * COLS)),
                    output_weights=torch.randn(10, LAYER_SIZE, device=device) * np.sqrt(2. / (ROWS * COLS)), # 10 x 16
                    first_biases=torch.zeros((LAYER_SIZE,1), device=device), # 16 x 1
                    second_biases=torch.zeros((LAYER_SIZE,1), device=device), # 16 x 1
                    output_biases=torch.zeros((10,1), device=device)) # 10 x 1
        # there are 10 digits
        i = 0
        j = 0
        batches = []
        costs = []
        learning_rate = 0.42
        with profiler.record_function("train_batch"):
            train_batch(i, BATCH_LENGTH, learning_rate, model, preprocess.train_data, preprocess.train_labels, device)
    print(prof)
    while i < TRAINING_DATA_SIZE:
        train_batch(i, BATCH_LENGTH, learning_rate, model, preprocess.train_data, preprocess.train_labels, device)
        test_batch(i, BATCH_LENGTH, model, preprocess.train_data, preprocess.train_labels)
        i += BATCH_LENGTH

    train_data = preprocess.train_data
    train_labels = preprocess.train_labels
    i = 0
    while j < EPOCHS:
        if j == 15:
            learning_rate = .01
        if j == 25:
            learning_rate = .001
        i = 0
        while i < TRAINING_DATA_SIZE:
            train_batch(i, BATCH_LENGTH, learning_rate, model, train_data, train_labels, device=device)
            test_batch(i, BATCH_LENGTH, model, train_data, train_labels)
            i += BATCH_LENGTH
        train_data = preprocess.train_data
        train_labels = preprocess.train_labels
        j += 1
        print("current epoch", j)

    pyplot.plot(batches, costs)
    pyplot.xlabel("Batch")
    pyplot.ylabel("Cost")
    pyplot.title(f"Average Cost per {BATCH_LENGTH} Images")
    pyplot.show()

    # np.save("model.npy", {'first_weights': model.first_weights,
    #                       'second_weights': model.second_weights,
    #                     'output_weights': model.output_weights,
    #                     'first_biases': model.first_biases,
    #                     'second_biases': model.second_biases,
    #                     'output_biases': model.output_biases})