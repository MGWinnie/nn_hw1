
class param:
    def __init__(self):
        self.lr = 0.2
        self.layer_dim = [[28*28, 200], [200, 10]]
        self.reg_rate = 0.4
        self.batch_size = 20
        self.path = ''
        self.epochs = 100