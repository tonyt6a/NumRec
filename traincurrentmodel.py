import numpy as np
import preprocess
import NN


EPOCHS = 5

data = np.load('model.npy', allow_pickle=True).item()

first_weights = data['first_weights']
output_weights = data['output_weights']
first_biases = data['first_biases']
output_biases = data['output_biases']

model = NN.Model(first_weights,
                 output_weights,
                 first_biases,
                 output_biases)

i = 0
j = 0
batches = []
costs = []
learning_rate_exp = 1
learning_rate = .01


while j < EPOCHS:
    indices = np.random.permutation(len(preprocess.train_data))

    train_data = preprocess.train_data[indices]
    train_labels = preprocess.train_labels[indices]
    if j == 15:
        learning_rate = .01
    if j == 25:
        learning_rate = .001
    i = 0
    while i < NN.TRAINING_DATA_SIZE:
        NN.train_batch(i, NN.BATCH_LENGTH, learning_rate, model, train_data, train_labels)
        # if learning_rate > .00001:
        #     learning_rate *= .1
        # if (cost < COST_THRESHOLD):
        #     break
        NN.test_batch(i, NN.BATCH_LENGTH, model, train_data, train_labels)
        i += NN.BATCH_LENGTH
    # indices = np.random.permutation(len(preprocess.train_data))
    # train_labels = preprocess.train_labels[indices]
    j += 1
    print("current epoch", j)

result = input("Do you want to update the current model? (Y/N)")
if result == "Y":
    np.save("model.npy", {'first_weights': model.first_weights,
                        'output_weights': model.output_weights,
                        'first_biases': model.first_biases,
                        'output_biases': model.output_biases})
    
print("Completed.")