import numpy as np
import preprocess

# will load model to test

LAYER_SIZE = 16
TRAINING_DATA_SIZE = preprocess.train_data.shape[0]


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
amt_correct= 0
for i in range(TRAINING_DATA_SIZE):
    input_vector = (np.array(preprocess.train_data[i]).flatten()).T
    hidden_activation, output_vector = feed_forward(input_vector)
    index_max = 0
    for j in range(1, len(output_vector)):
        if output_vector[j] > output_vector[index_max]:
            index_max = j
    if index_max == preprocess.train_labels[i]:
        amt_correct += 1

print(f'Amount correct:: {amt_correct}/{TRAINING_DATA_SIZE}\nPercentage:: {amt_correct / TRAINING_DATA_SIZE}')