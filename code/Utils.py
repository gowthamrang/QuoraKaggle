import numpy as np
from keras import backend as K
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


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



GCOUNT = 0 

def ternary_search(left, right, pred, true):
    global GCOUNT
    GCOUNT+=1
    if(GCOUNT > 100): return -1, -1
    if (left-right) < 0.05: return get_fscore(pred, true, (left+right)/2), (left+right)/2
    
    left_third = get_fscore(pred, true, (left*2+right)/3)
    right_third = get_fscore(pred, true, (left+2*right)/3) 
    print("Fscore at %.3f is %.3f" %((left*2+right)/3, left_third))
    print("Fscore at %.3f is %.3f" %((left+2*right)/3, right_third))

    if left_third<right_third:
        return ternary_search(left_third, right, pred, true)
    return ternary_search(left, right_third, pred, true)
    
    
def get_best_threshold(pred, true):
    vals = [(get_fscore(pred, true, i*1./10.), i*1./10) for i in range(1,10,1)]
    for e1,e2 in vals:
        print("Fscore at %.3f is %.3f" %(e2, e1))
    fscore, val = ternary_search(0,1.0, pred, true)
    v_ = max(vals)
    print("Best from t search %.3f @ %.3f, normal %.3f @ %.3f" %(fscore , val, v_[1], v_[0]))
    return max(vals)[1]


    
def get_fscore(pred, true, thresh):
    return f1_score(true, (pred>thresh).astype(int))




class Metrics(Callback):

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
    
    def on_epoch_end(self, epoch, logs={}):
        val_predict = (self.model.predict(self.validation_data[0])>0.33).astype(int)
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(f'— val_f1: {_val_f1} — val_precision: {_val_precision} — val_recall {_val_recall}')
        return
 
