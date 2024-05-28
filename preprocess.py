import numpy as np
from matplotlib import pyplot
import torch

from keras._tf_keras.keras.datasets import mnist


(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
train_data = torch.tensor(train_data, dtype=torch.float32, device='cuda')
train_labels = torch.tensor(train_labels, dtype=torch.long, device='cuda')
test_data = torch.tensor(test_data, dtype=torch.float32, device='cuda')
test_labels = torch.tensor(test_labels, dtype=torch.long, device='cuda')


indices = torch.randperm(len(train_data), device='cuda')

train_data = train_data[indices]
train_labels = train_labels[indices]

print('X_train: ' + str(train_data.shape))
print('Y_train: ' + str(train_labels.shape))
print('X_test:  '  + str(test_data.shape))
print('Y_test:  '  + str(test_labels.shape))

# for i in range(9):  
#     pyplot.subplot(330 + 1 + i)
#     pyplot.imshow(train_data[i], cmap=pyplot.get_cmap('gray'))
# pyplot.show()

# print(a.shape)
# return train_data, train_labels, test_data, test_labels