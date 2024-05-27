import numpy as np
import preprocess
import NN

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

first_weights = data['first_weights']
output_weights = data['output_weights']
first_biases = data['first_biases']
output_biases = data['output_biases']
training_amt_correct= 0
test_amt_correct = 0
model = NN.Model(first_weights, output_weights, first_biases, output_biases)
for i in range(TRAINING_DATA_SIZE):
    input_vector = (np.array(preprocess.train_data[i]).flatten()).T
    hidden_activation, output_vector = NN.feed_forward(input_vector, model)
    index_max = 0
    for j in range(1, len(output_vector)):
        if output_vector[j] > output_vector[index_max]:
            index_max = j
    if index_max == preprocess.train_labels[i]:
        training_amt_correct += 1
for i in range(preprocess.test_data.shape[0]):
    input_vector = (np.array(preprocess.test_data[i]).flatten()).T
    hidden_activation, output_vector = NN.feed_forward(input_vector, model)
    index_max = 0
    for j in range(1, len(output_vector)):
        if output_vector[j] > output_vector[index_max]:
            index_max = j
    if index_max == preprocess.test_labels[i]:
        test_amt_correct += 1
print("From training data - ")
print(f'Amount correct:: {training_amt_correct}/{TRAINING_DATA_SIZE}\nPercentage:: {training_amt_correct / TRAINING_DATA_SIZE}')
print("From test data - ")
print(f'Amount correct:: {test_amt_correct}/{preprocess.test_data.shape[0]}\nPercentage:: {test_amt_correct / preprocess.test_data.shape[0]}')