from train import *
import random
for iter in range(50):
    i = random.uniform(0,4)
    lr = 10 ** (-i)
    j = random.randint(5,8)
    units = 2 ** j
    k = random.uniform(4,6)
    lr_decay = 1 - 10 **(-k)
    max_acc = excute(iter, lr, units, lr_decay)