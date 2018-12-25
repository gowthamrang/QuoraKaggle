import numpy as np
from keras import backend as K

def search_lr(model, X,y, base_lr=1e-5, max_lr=100):
    step_size = min(len(X), 300) #every 500 iterations or so double the learning rate
    batch_count = int(X.shape[0]/64)
    loss = {}
    cur_ = base_lr
    for i in range(step_size*20):        
        if(i%step_size == 0):
            cur_ *=2
            print('setting new current value %.4f' %cur_)
            loss[cur_] = []
            K.set_value(model.optimizer.lr, cur_)
        if cur_ >= max_lr: break
        start = (i*64)%batch_count       
        loss[cur_].append(model.train_on_batch(X[start*64:start*64+64], y[start*64:start*64+64]))


    for each in loss:
        print(each, np.mean(loss[each]))
    return 