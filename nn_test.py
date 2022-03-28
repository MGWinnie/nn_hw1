import numpy as np
import gzip
from nn_train import Net
from config import param

def load_data():
    with gzip.open('data/t10k-labels-idx1-ubyte.gz', 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    labels = np.identity(10)[labels] # To one-hot

    with gzip.open('data/t10k-images-idx3-ubyte.gz', 'rb') as imgpath:
        images = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(labels), 784)
    return images, labels

def load_model(layer_dim, lr, reg_rate, batch_size, model_path):
    saved_model = np.load(model_path, allow_pickle=True)
    model = Net(layer_dim, lr, reg_rate, batch_size)
    model.params = saved_model['params'][()]
    return model

def test():
    x_test, y_test = load_data()
    params = param()
    model = load_model(params.layer_dim, params.lr, params.reg_rate, params.batch_size, params.path + 'classifier.npz')
    _, acc = model.evaluate(x_test, y_test)
    print('Test accuracy:', acc)


