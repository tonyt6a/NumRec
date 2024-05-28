import numpy as np
import preprocess
import NN


EPOCHS = 10

data = np.load('model.npy', allow_pickle=True).item()

first_weights = data['first_weights']
second_weights = data['second_weights']
output_weights = data['output_weights']
first_biases = data['first_biases']
second_biases = data['second_biases']
output_biases = data['output_biases']

model = NN.Model(first_weights,
                 second_weights,
                 output_weights,
                 first_biases,
                 second_biases,
                 output_biases)

i = 0
j = 0
batches = []
costs = []
learning_rate_exp = 1
learning_rate = 1


while j < EPOCHS:
    indices = np.random.permutation(len(preprocess.train_data))

    train_data = preprocess.train_data[indices]
    train_labels = preprocess.train_labels[indices]
    train_data = np.ascontiguousarray(train_data)
    train_labels = np.ascontiguousarray(train_labels)
    # if j == 15:
    #     learning_rate = .01
    # if j == 25:
    #     learning_rate = .001
    i = 0
    for k in range(int(NN.TRAINING_DATA_SIZE / 1000)):
        print("Real index", k)
        while i < 1000:
            NN.train_batch(i, NN.BATCH_LENGTH, learning_rate, model, train_data, train_labels)
            # if learning_rate > .00001:
            #     learning_rate *= .1
            # if (cost < COST_THRESHOLD):
            #     break
            NN.test_batch(i, NN.BATCH_LENGTH, model, train_data, train_labels)
            i += NN.BATCH_LENGTH
        train_data = np.concatenate((train_data[1000:], train_data[:1000]))
        train_labels = np.concatenate((train_labels[1000:], train_labels[:1000]))
        i = 0
    # indices = np.random.permutation(len(preprocess.train_data))
    # train_labels = preprocess.train_labels[indices]
    j += 1
    print("current epoch", j)

result = input("Do you want to update the current model? (Y/N)")
if result == "Y":
    np.save("model.npy", {'first_weights': model.first_weights,
                          'second_weights': model.second_weights,
                        'output_weights': model.output_weights,
                        'first_biases': model.first_biases,
                        'second_biases': model.second_biases,
                        'output_biases': model.output_biases})
    
print("Completed.")