import argparse
import numpy as np 
import pandas as pd
import os
from collections import Counter
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from keras.preprocessing.sequence import pad_sequences
from collections import Counter , defaultdict
from keras.utils import Sequence
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
import Utils
FOLDS = 3 #donot change
DATA_SPLIT_SEED = 1403 #donot change



def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')    
def getxyfromdf(df, T): return pad_sequences(T.text_to_sequence(df.question_text.values), maxlen=config['seq_len']), df.target.values  if 'target' in df else None


class Tokenizer():  
    def __init__(self, max_vocab=400000):
        self.c = Counter()
        self.word2id, self.id2word = {"#pad#":0}  , ['#pad#']
        self.MAX_VOCAB = 400000
        return

    @staticmethod
    def tokenize(text):
        for each in "!\"#$%&()*+,-./:;<=>?@[\]^_`{|}~'’'":
            text = text.replace(each,'')        
        return text.split(' ')
        
    def fit_texts(self,texts, embeddings_index=None):
        for text in texts:
            self.c.update(self.tokenize(text))   
        if embeddings_index !=None: self.build_vocab(embeddings_index)
        
    def text_to_sequence(self,texts):
       return [[ self.word2id[x] if x in self.word2id else 0  for x in self.tokenize(text) ] for text in texts]
        
    def sequences_to_text(self,seqs):
        return [' '.join( self.id2word[i] for x in seq) for seq in seqs]

    def build_vocab(self, embedding_index):
        if embedding_index:
            all_embs = np.stack(embeddings_index.values())
            emb_mean,emb_std = all_embs.mean(), all_embs.std()
        else:
            emb_mean,emb_std = 0., 1.

        vocab_size = min(self.MAX_VOCAB, len(self.c))
        #self.embedding_matrix = np.random.normal( emb_mean, emb_std, (vocab_size, all_embs.shape[1]))
        self.embedding_matrix =  K.eval(tf.truncated_normal((vocab_size, 300), emb_mean, emb_std))
        self.unrepresented = []
        for ind,word in enumerate(sorted(self.c, key= lambda x: self.c[x], reverse=True)) :
            if ind > vocab_size: break

            self.word2id[word] = len(self.id2word)
            self.id2word.append(word)
            if word in embedding_index:
                self.embedding_matrix[ind] = embedding_index[word]
            elif word.lower() in embedding_index:
                self.embedding_matrix[ind] = embedding_index[word.lower()]
            else:
                self.unrepresented.append(word)
        return
################################
class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        self.b = self.add_weight((input_shape[1],),
                                    initializer='zero',
                                    name='{}_b'.format(self.name),
                                    regularizer=self.b_regularizer,
                                    constraint=self.b_constraint) if self.bias else None

        self.built = True

    def compute_mask(self, input, input_mask=None): return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias: eij += self.b
        eij = K.tanh(eij)
        a = K.exp(eij)
        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim
####################  
def lstm_gru_attn(embedding_matrix):
    global config
    inp = Input(shape=(config['seq_len'],)) 
    x = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights = [embedding_matrix], trainable=False)(inp)    
    x = SpatialDropout1D(config['dropout'])(x)
    x = Bidirectional(CuDNNLSTM(config['internal_dim'], return_sequences=True))(x) 
    y = Bidirectional(CuDNNGRU(config['internal_dim'], return_sequences=True))(x)
    A1 = Attention(config['seq_len'])(x)
    A2 = Attention(config['seq_len'])(y)

    pool1 = GlobalAveragePooling1D()(y)
    pool2 = GlobalMaxPooling1D()(y)
    concat = concatenate([pool1, pool2, A1, A2])
    x = Dense(64, activation="relu")(concat)
    x = Dropout(config['dropout'])(x)
    output = Dense(1, activation='sigmoid')(x)

    opt = Adam(lr=config['lr'], decay=1e-6)

    model = Model(inputs=inp, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=opt)  
    print(model.summary())
    return model
    
################################
##################### HELPER FUNC BEGIN ####################
def one_round(model, train_x, train_y, val_x, val_y, tx, epochs ):
    for e in range(epochs):
        model.fit(train_x,train_y,batch_size=config['bs'], epochs=1)
        val_pred_y = model.predict(val_x)
        print(f"Epoch {e} Aprox. f1 score {f1_score(val_y, (val_pred_y>0.33).astype(int))}")
    test_y = model.predict(tx)
    return val_pred_y, test_y

def get_fscore(pred, true, thresh):
    return f1_score(true, (pred>thresh).astype(int))

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

##################### HELPER FUNC ENDS ####################
def preprocess(train_df, test_df, embeddings_index):
    T = Tokenizer()
    T.fit_texts(train_df.question_text.values)
    T.fit_texts(test_df.question_text.values)
    T.build_vocab(embeddings_index)
    return T

def train_full(train_df, test_df, embeddings_index):
    global FOLDS, config
    T = preprocess(train_df, test_df, embeddings_index)
    train_X, train_Y = getxyfromdf(train_df, T)
    test_X, _ = getxyfromdf(test_df, T)    
    
    model = config['func'](T.embedding_matrix)
    #Utils.search_lr(model,train_X, train_Y)
    print('Splitting stratified k fold')
    splits = list(StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=DATA_SPLIT_SEED).split(train_X, train_Y))    
    train_meta = np.zeros_like(train_Y)
    test_meta = np.zeros((test_X.shape[0],1))
    
    for idx, (train_idx, valid_idx) in enumerate(splits):
        tr_x = train_X[train_idx]
        tr_y = train_Y[train_idx]
        val_x = train_X[valid_idx]
        val_y = train_Y[valid_idx]       
        print(f"Running fold {idx}") 
        val_pred_y, test_y = one_round(model,tr_x, tr_y, val_x, val_y, test_X, 1)
        train_meta[valid_idx] = val_pred_y.reshape(-1)
        test_meta += test_y/FOLDS
    print('Getting best threshold and saving the test tsv file')
    #th = get_best_threshold(val_pred_y, val_y)
    train_df["targets_%s" %config['func'].__name__] = train_meta
    train_df.to_csv("../output/train.csv", index_label ='qid')

    test_df["targets_%s" %config['func'].__name__] = test_meta    
    test_df.to_csv("../output/test.csv", index_label ='qid')    
    return
################ CONFIGURATIONS #################

config = {}
config['embedding'] = 'None'
config['seq_len'] = 100
config['bs'] = 64
config['dropout'] = 0.25
config['internal_dim'] = 100
config['lr'] = 0.01
config['func'] = lstm_gru_attn #function is used as a variable



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Map wikipedia categories to kngine entity types')
    for each in config:
        parser.add_argument(f'--{each}', default = config[each], type = type(config[each]), required=False)
    args = vars(parser.parse_args())
    print('Configuration....')
    for each in args:
        print(each, args[each])
    print('#'*20)
    EMBEDDING_FILE = None
    if config['embedding'] == 'glove':
        EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
    
    train_df = pd.read_csv("../output/train.csv")
    test_df = pd.read_csv("../output/test.csv")
    if EMBEDDING_FILE and os.path.exists(EMBEDDING_FILE):
        print('Loading %s' %EMBEDDING_FILE )
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf-8") if len(o)>100)
    else:
        embeddings_index = {}    
    train_full(train_df, test_df,embeddings_index )
    