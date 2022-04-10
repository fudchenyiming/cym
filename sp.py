import numpy as np
class Modules:

    def __init__(self):
        raise NotImplementedError

    def forward(self, x):
        for layer in self.layers.values():
            x = layer(x)
        return x

    def backward(self, dx, fine_tuning = False):
        for name,layer in reversed(self.layers.items()):
            if fine_tuning and layer.__class__.__name__ == "Relu":
                dx = layer.backward(np.zeros(dx.shape))
            else:
                dx = layer.backward(dx)

    def __call__(self,x):
        return self.forward(x)

    def state_dict(self):
        return self.parameters

    def save_state_dict(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.parameters, f, pickle.HIGHEST_PROTOCOL)
        return

    def load_state_dict(self, state_dict=None, path=None):
        if not state_dict:
            import pickle
            with open(path, 'rb') as f:
                state_dict = pickle.load(f)
        for layer in state_dict.keys():
            for param, v in state_dict[layer].items():
                self.parameters[layer][param] = v
        return

class SequentialProcess(Modules):

    def __init__(self, architecture):
        cdict = {}
        self.layers = {}
        self.parameters = {}
        for layer in architecture:
            name = layer.__class__.__name__
            try:
                count = cdict[name] + 1
            except:
                count = 1
            cdict[name] = count
            self.layers[name.lower() + str(count)] = layer
            try:
                self.parameters[name.lower() + str(count)] = layer.parameters
            except:
                pass
            