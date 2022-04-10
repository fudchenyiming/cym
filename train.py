import optimizers
import functions
import sp
import os
import time
import tensorflow as tf
from data import Dataset, Dataloader
from tqdm import tqdm
import numpy as np


class TrainData(Dataset):
    def __init__(self,data_train):
        (x,y)= data_train
        x = x / 255
        self.x = x.reshape(x.shape[0], 28, 28, 1).transpose(0, 2, 1, 3)
        self.y = np.eye(10)[y].reshape(y.shape[0],10)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        x = self.x[index].copy()
        return (x, self.y[index])

class ValidData(Dataset):
    def __init__(self,data_valid):
        (x,y) =data_valid
        x = x / 255
        self.x = x.reshape(x.shape[0], 28, 28, 1).transpose(0, 2, 1, 3)
        y = y.reshape(y.shape[0],1)
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

def accuracy(y_test_hat, y_test, nums, flag=False):
    test_hat = (y_test_hat == np.max(y_test_hat, axis = 1).reshape(y_test.shape[0], 1)).astype(int)
    labels = np.array(list(range(nums))).reshape(nums, 1)
    th = np.matmul(test_hat, labels)
    if flag:
        tmp = np.matmul(y_test, labels)
        return np.sum(th == tmp)
    else:
        return np.sum(th == y_test)
def crossvalid(dataset):
    x,y = dataset
    train_index = np.random.choice(len(x), round(len(x) * 0.8), replace=False)
    valid_index = np.array(list(set(range(len(x))) - set(train_index)))
    return (x[train_index], y[train_index]), (x[valid_index], y[valid_index])

def train(iter_num,model, data,units,epoch_num = 100, lr = 1e-3, train_batchsize = 32, seed = 147, plot = True, monitor=True,save = True ,lr_decay = 1):
    np.random.seed(seed)
    (x_train, y_train), (x_test, y_test) = data.load_data()
    (x_train,y_train),(x_valid, y_valid) = crossvalid((x_train, y_train))
    criterion = functions.EntropyLoss(nnModel = model)
    optimizer = optimizers.SGD(model, lr = lr, momentum = 0.9, lr_decay=lr_decay)
    costs = []
    traindata = TrainData((x_train, y_train))
    validdata = ValidData((x_valid, y_valid))
    mtrain = traindata.__len__()
    mvalid = validdata.__len__()
    trainloader = Dataloader(traindata, batch_size=train_batchsize)
    validloader = Dataloader(validdata, batch_size=128)
    train_accs = []
    valid_accs = []
    train_time = []
    test_time = []
    flag = False
    for epoch in range(epoch_num):
        tn = 0
        tic = time.time()
        cost = 0
        for x_train, y_train in tqdm(trainloader):
            y_hat = model(x_train)
            loss = criterion(y_hat, y_train)
            costs.append(loss.item)
            cost += loss.item
            tn += accuracy(y_test_hat=y_hat, y_test=y_train, nums=10, flag=True)
            loss.backward(False)
            optimizer.step()
        train_time.append(time.time() - tic)
        train_accs.append(tn / mtrain)
        tn = 0
        tic = time.time()
        for xv, yv in validloader:
            yv_hat = model(xv)
            tn += accuracy(yv_hat, yv, 10)
        test_time.append(time.time() - tic)
        valid_accs.append(tn / mvalid)
        if epoch > 5 and max(valid_accs) < 0.9:
            flag = True
            break
        if monitor:
            print('Epoch:{}\tcost:{}\ttrain acc:{}\tvalid acc:{}'.format(epoch + 1, cost, train_accs[-1], valid_accs[-1]))

    if save and not flag:
        path = os.path.join('models', 'model{}baseline{}units{}best_acc={}'.format(iter_num+1,lr, units,max(valid_accs)))
        if not os.path.exists(path):
            os.makedirs(path)
        model.save_state_dict(os.path.join(path,'model.pkl'))
    return max(valid_accs)
def excute(iter_num,lr,units,lr_decay):
    model = sp.SequentialProcess([functions.Flatten(),
                                  functions.Linear(dims_in=784, dims_out=units),
                                  functions.Relu(),
                                  functions.Dropout(),
                                  functions.Linear(dims_in=units, dims_out=10),
        ])
    mnist = tf.keras.datasets.mnist
    max_acc = train(iter_num,model=model, data=mnist,units = units,epoch_num=50, lr=lr, seed=123, train_batchsize=64, lr_decay=lr_decay)
    return max_acc