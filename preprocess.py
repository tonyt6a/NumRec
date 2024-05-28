import numpy as np
from matplotlib import pyplot

from keras._tf_keras.keras.datasets import mnist


(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

indices = np.random.permutation(len(train_data))

train_data = train_data[indices]
train_labels = train_labels[indices]
train_data = np.ascontiguousarray(train_data)
train_labels = np.ascontiguousarray(train_labels)

print('X_train: ' + str(train_data.shape))
print('Y_train: ' + str(train_labels.shape))
print('X_test:  '  + str(test_data.shape))
print('Y_test:  '  + str(test_labels.shape))

# for i in range(9):  
#     pyplot.subplot(330 + 1 + i)
#     pyplot.imshow(train_data[i], cmap=pyplot.get_cmap('gray'))
# pyplot.show()

a = np.array(train_data[0])
# print(a.shape)
# return train_data, train_labels, test_data, test_labels