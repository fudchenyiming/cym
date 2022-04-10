from train import *
model = sp.SequentialProcess([functions.Flatten(),
                                  functions.Linear(dims_in=784, dims_out=256),
                                  functions.Relu(),
                                  functions.Dropout(),
                                  functions.Linear(dims_in=256, dims_out=10),
        ])
model.load_state_dict(path='model.pkl')
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
def test(model, data_test):
    testdata = ValidData(data_test)
    testloader = Dataloader(testdata, batch_size=128)
    tn = 0
    for x,y in testloader:
        y_hat = model(x)
        tn += accuracy(y_hat, y, 10)
    print(tn/10000)
test(model,(x_test, y_test))