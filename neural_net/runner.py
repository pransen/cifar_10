import torch
import torch.nn as nn
import torch.optim as optim

from dataset.dataset_util import LoadCifar
from neural_net.cnn_model import NeuralNet

cifar = LoadCifar()

def train(batch_size=512, learning_rate = 1e-3, num_epochs = 200, num_classes=10):
    net = NeuralNet(num_classes)
    train_x, train_y = cifar.load_train_dataset()
    min_batches = cifar.get_mini_batches(train_x, train_y, batch_size)
    # optimizer = optim.SGD(net.parameters(), learning_rate, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), learning_rate)
    criterion = nn.CrossEntropyLoss()
    num_batches = len(min_batches)

    for e in range(num_epochs):
        data_loss = 0.0
        for idx, batch in enumerate(min_batches):
            batch_x, batch_y = batch
            x_tensor = torch.from_numpy(batch_x)
            x_tensor = x_tensor.type('torch.FloatTensor')
            y_tensor = torch.from_numpy(batch_y)
            y_tensor = y_tensor.type('torch.LongTensor')

            # zero the gradients
            optimizer.zero_grad()

            # forward pass
            output = net(x_tensor)

            loss = criterion(output, y_tensor)

            loss.backward()
            optimizer.step()

            # print loss
            data_loss += loss.item()

            print('[%d, %5d] loss: %.5f' % (e + 1, idx + 1, loss.item()))
        print('Average loss after epoch %d is : %.5f' % (e + 1, data_loss / num_batches))

    return net


def test(model):
    test_x, test_y = cifar.load_test_dataset()
    x_tensor = torch.from_numpy(test_x)
    x_tensor = x_tensor.type('torch.FloatTensor')
    y_tensor = torch.from_numpy(test_y)
    y_tensor = y_tensor.type('torch.LongTensor')
    with torch.no_grad():
        predicted_outputs = net(x_tensor)
        _, predicted = torch.max(predicted_outputs.data, 1)
        correct = 0
        correct += (predicted == y_tensor).sum().item()
        total = y_tensor.size(0)
        print('Accuracy of the network on the %d test images: %d %%' % (total, 100 * correct / total))


if __name__ == '__main__':
    net = NeuralNet(10)
    # torch.save(net.state_dict(), 'cifar_ckpt.pth')
    net.load_state_dict(torch.load('cifar_ckpt.pth'))

    test(net)
