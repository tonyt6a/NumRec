import numpy as np
import preprocess

# will load model to test

LAYER_SIZE = 16


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def get_error(actual, expected):
    error = 0
    for i in range(len(actual)):
        error += (actual[i,] - expected[i,]) ** 2
    return error


data = np.load('model.npy', allow_pickle=True).item()
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

first_weights = data['first_weights']
output_weights = data['output_weights']
first_biases = data['first_biases']
output_biases = data['output_biases']
i = 1

input_vector = (np.array(preprocess.train_data[i]).flatten()).T
hidden_activation, output_vector = feed_forward(input_vector)
activation_hat = np.zeros(10,) # vector with 10 digits
activation_hat[preprocess.train_labels[i],] = 1 # set actual label to 1 (rest will be 0)
error = get_error(output_vector, activation_hat)
print("hi")