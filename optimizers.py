import numpy as np

class SGD:

    def __init__(self, sequentialprocess, lr, momentum, w_decay=0, lr_decay=1):
        self.sequentialprocess = sequentialprocess
        self.lr = lr
        self.v_dict = {}
        self.count = 0
        self.momentum = momentum
        self.w_decay = w_decay
        self.lr_decay = lr_decay
        for name, layer in self.sequentialprocess.layers.items():
            try:
                parameters = layer.parameters
                self.v_dict[name] = {}
                for parameter in parameters.keys():
                    self.v_dict[name][parameter] = 0
            except:
                continue

    def step(self):
        lr = self.lr * (self.lr_decay ** self.count)
        self.count += 1
        for name, layer in self.sequentialprocess.layers.items():
            try:
                for parameter in layer.parameters.keys():
                    self.v_dict[name][parameter] = self.momentum* layer.grads[parameter] \
                                                   + (1 - self.momentum)* layer.grads[parameter]
                    parameter_cor = self.v_dict[name][parameter] / (1 - self.momentum**self.count)
                    layer.parameters[parameter] = layer.parameters[parameter] * (1 - self.w_decay) - self.lr * parameter_cor
            except:
                continue