import _pickle as pickle
import matplotlib.pyplot as plt
import os
import numpy as np


class LoadCifar:
    def __init__(self):
        self.CIFAR_BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cifar')
        print(self.CIFAR_BASE_PATH)

    def load_test_dataset(self):
        num_test = 10000
        test_data = np.zeros(shape=(num_test, 3, 32, 32))
        test_labels = np.zeros(shape=num_test)
        d = self.__unpickle(os.path.join(self.CIFAR_BASE_PATH, "test_batch"))
        batch_data = d[b'data']
        test_data[0: num_test, :, :, :] = batch_data.reshape(-1, 3, 32, 32) / 255 # Normalization
        batch_labels = d[b'labels']
        test_labels[0: num_test] = batch_labels

        return test_data, test_labels

    def load_train_dataset(self):
        num_train = 50000
        train_data = np.zeros(shape=(num_train, 3, 32, 32))
        train_labels = np.zeros(shape=num_train)
        for i in range(5):
            d=self.__unpickle(os.path.join(self.CIFAR_BASE_PATH, "data_batch_" + str(i + 1)))
            batch_data = d[b'data']
            batch_size = batch_data.shape[0]
            start_idx = i * batch_size
            end_idx= (i + 1) * batch_size
            train_data[start_idx : end_idx, :, :, :] = batch_data.reshape(-1,3,32,32) / 255 # Normalization
            batch_labels = d[b'labels']
            train_labels[start_idx: end_idx] = batch_labels

        return train_data, train_labels

    def __unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def display_random_image(self, data):
        pass

    def get_mini_batches(self, train_x, train_y, batch_size=512):
        num_train=train_x.shape[0]
        num_full_batches = int(num_train / batch_size)
        mini_batches= list()
        for i in range(num_full_batches):
            start_idx = i * batch_size
            end_idx = (i+1) * batch_size
            batch_x = train_x[start_idx:end_idx, :, :, :]
            batch_y = train_y[start_idx:end_idx]
            mini_batches.append((batch_x, batch_y))

        if num_train % batch_size != 0:
            batch_x = train_x[end_idx:, :, :, :]
            batch_y = train_y[end_idx:]
            mini_batches.append((batch_x,batch_y))

        return mini_batches


if __name__ == '__main__':
    cifar=LoadCifar()
    train_x, train_y = cifar.load_train_dataset()
    # cifar.load_test_dataset()

    mini_batches = cifar.get_mini_batches(train_x, train_y)
    print()
