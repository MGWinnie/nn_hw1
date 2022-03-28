import numpy as np
from config import param
import gzip
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def dSigmoid(x):
    z = sigmoid(x)
    return z * (1 - z)

class Net:
    def __init__(self, nn_sequential, lr, reg_rate, batch_size):
        self.params = {}
        self.nn_sequential = nn_sequential
        for idx, layer in enumerate(self.nn_sequential):
            input_dim = layer[0]
            output_dim = layer[1]
            self.params['W' + str(idx + 1)] = np.random.randn(input_dim, output_dim)
            self.params['b' + str(idx + 1)] = np.random.randn(output_dim)

        self.lr = lr
        self.reg_rate = reg_rate
        self.batch_size = batch_size

    def softmax(self, z):
        z = np.exp(z)
        return z / np.sum(z, axis=1).reshape(-1, 1)

    def l2_regularize(self):
        l2_norm = 0
        for key, value in self.params.items():
            l2_norm += np.linalg.norm(value)
        return self.reg_rate * l2_norm

    def forward(self, x, y):
        self.z = []
        self.a = [x]
        for idx in range(len(self.nn_sequential)):
            W = self.params["W" + str(idx + 1)]
            b = self.params["b" + str(idx + 1)]
            z = np.dot(x, W) + b
            x = sigmoid(z)
            self.z.append(z)
            self.a.append(x)

        self.output = self.softmax(x)
        self.pred = self.output * y
        loss = -np.sum(np.log(np.max(self.pred, axis=1))) + self.l2_regularize()
        return loss, self.output

    def backward(self, y):
        self.grads = {}
        delta = self.output - y
        for idx in range(len(self.nn_sequential)-1, -1, -1):
            d_regW = self.reg_rate * self.params["W" + str(idx + 1)]
            d_regb = self.reg_rate * self.params["b" + str(idx + 1)]
            d_activation = dSigmoid(self.z[idx])
            self.grads["dW" + str(idx + 1)] = (np.dot(self.a[idx].T, delta * d_activation) + d_regW) / self.batch_size
            self.grads["db" + str(idx + 1)] = (np.sum(delta * d_activation, axis=0) + d_regb) / self.batch_size
            delta = np.dot(delta * d_activation, self.params["W" + str(idx + 1)].T)

    def lr_decay(self):
        self.lr *= 0.95

    def sgd_update(self):
        for idx, layer in enumerate(self.nn_sequential):
            self.params["W" + str(idx + 1)] -= self.lr * self.grads["dW" + str(idx + 1)]
            self.params["b" + str(idx + 1)] -= self.lr * self.grads["db" + str(idx + 1)]

    def evaluate(self, x_test, y_test):
        test_loss, test_pred = self.forward(x_test, y_test)
        return test_loss, accuracy(np.argmax(test_pred, axis=1), np.argmax(y_test, axis=1))

    def save_model(self, path):
        np.savez(path + 'classifier.npz', params=self.params, grads=self.grads)


def accuracy(pred, label):
    result = np.array(pred == label)
    return np.sum(result) / len(result)

def plot_results(train_loss, test_loss, train_acc, test_acc, path):
    plt.plot(train_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train loss')
    plt.savefig(path + 'train_loss.jpg', dpi=300)
    plt.cla()

    plt.plot(test_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Test loss')
    plt.savefig(path + 'test_loss.jpg', dpi=300)
    plt.cla()

    plt.plot(train_acc)
    plt.xlabel('Epoch')
    plt.title('Train accuracy')
    plt.savefig(path + 'train_acc.jpg', dpi=300)
    plt.cla()

    plt.plot(test_acc)
    plt.xlabel('Epoch')
    plt.title('Test accuracy')
    plt.savefig(path + 'test_acc.jpg', dpi=300)


def load_data():
    with gzip.open('data/train-labels-idx1-ubyte.gz', 'rb') as lbpath:
        train_labels = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    train_labels = np.identity(10)[train_labels] # To one-hot

    with gzip.open('data/train-images-idx3-ubyte.gz', 'rb') as imgpath:
        train_images = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(train_labels), 784)

    with gzip.open('data/t10k-labels-idx1-ubyte.gz', 'rb') as lbpath:
        test_labels = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    test_labels = np.identity(10)[test_labels] # To one-hot

    with gzip.open('data/t10k-images-idx3-ubyte.gz', 'rb') as imgpath:
        test_images = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(test_labels), 784)
    return train_images, train_labels, test_images, test_labels

def train():
    x_train, y_train, x_test, y_test = load_data()
    params = param()
    nn = Net(params.layer_dim, params.lr, params.reg_rate, params.batch_size)

    test_loss = []
    test_acc = []
    train_loss = []
    train_acc = []

    for epoch in range(params.epochs):
        losses = []
        accs = []
        for i in range(0, len(y_train), nn.batch_size):
            X = x_train[i: min(i + nn.batch_size, len(y_train))]
            Y = y_train[i: min(i + nn.batch_size, len(y_train))]
            loss, pred = nn.forward(X, Y)
            nn.backward(Y)
            nn.sgd_update()
            losses.append(loss)
            accs.append(accuracy(np.argmax(pred, axis=1), np.argmax(Y, axis=1)))

        loss, acc = nn.evaluate(x_test, y_test)
        test_loss.append(loss)
        test_acc.append(acc)
        train_loss.append(sum(losses) / len(losses))
        train_acc.append(sum(accs) / len(accs))
        nn.lr_decay()

        print('epoch:', epoch, 'loss:', "{:.3f}".format(sum(losses) / len(losses)),
              'test accuracy:', "{:.3f}".format(acc))

    nn.save_model(params.path)
    plot_results(train_loss, test_loss, train_acc, test_acc, params.path)