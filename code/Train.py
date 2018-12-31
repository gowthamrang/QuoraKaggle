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
from Architecture import lstm_gru_attn
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import glob
import re
import nltk
lemma = nltk.wordnet.WordNetLemmatizer()

FOLDS = 3 #donot change
DATA_SPLIT_SEED = 1403 #donot change


def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')    
def getxyfromdf(df, T): return pad_sequences(T.text_to_sequence(df.question_text.values), maxlen=config['seq_len'], padding='post'), df.target.values  if 'target' in df else None

class Tokenizer():  
    def __init__(self, embeddings_index, max_vocab=400000):
        self.c = Counter()
        self.word2id, self.id2word = {"#pad#":0}  , ['#pad#']
        self.MAX_VOCAB = 400000
        self.vocab = set(embeddings_index.keys())
        return

    def inVocab(self, word):
        return word in self.vocab or word.lower() in self.vocab or word.upper() in self.vocab 

    def two_split_word(self, word):
        if len(word)<6 or self.inVocab(word): return [word]
        for i in range(3,len(word)-3):
            if self.inVocab(word[:i])  and self.inVocab(word[i:]):
                return [word[:i], word[i:]]
        return [word]

    def tokenize(self, text):
        if config['number_to_x'] == True:
            text = re.sub('\d+.?\d+', "x", text)
        if config['block_math'] == True:
            text = re.sub('\[math\].*\[/math\]', "x", text,flags=re.IGNORECASE)
        if config['swear_word_map'] == True:
            text = re.sub('f[\*|u][\*|c][\*|k]', 'fuck', text, flags=re.IGNORECASE)
            text = re.sub('d[\*|i][\*|c][\*|k]', 'dick', text, flags=re.IGNORECASE)
            text = re.sub('c[\*|o][\*|c][\*|k]', 'cock', text, flags=re.IGNORECASE)
            text = re.sub('b[\*]+[tch]?', 'bitch', text ,flags=re.IGNORECASE)
            text = re.sub('s[\*|h][\*|i][\*|t]', 'shit', text,flags=re.IGNORECASE)
            text = re.sub('a[\*|s][\*|s]', 'ass', text,flags=re.IGNORECASE)
            text = re.sub('p[\*|u]ssy', 'ass', text,flags=re.IGNORECASE)            
        if config['n_t_replace'] == True:
            text = text.replace("can't", ' can not ')
            text = text.replace("n't", ' not ')
            text = text.replace("'ve", ' have ')
            text = text.replace("'m", ' am ')
        for each in "!\"#$%&()*+,-./:;<=>?@[\]^_`{|}~'’'":
            text = text.replace(each,' ')        
        res = list(filter(lambda x: len(x)>0, text.split(' ')))
        if config['two_split'] == True:
            res1 = []
            for e in res: res1.extend(self.two_split_word(e))
            res = res1
        return res


        
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

# Set callback functions to early stop training and save the best model so far

def one_round(model, model_dir, train_x, train_y, val_x, val_y, tx, epochs):
    callbacks = [   ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=2, min_lr=0.001),
                    ModelCheckpoint(filepath='%s/weights.{epoch:02d}-{val_loss:.3f}.hdf5' %model_dir, monitor='val_loss'),
                    Utils.Metrics()]
    model.fit(train_x,train_y,validation_data = (val_x, val_y), batch_size=config['bs'], epochs=epochs, callbacks=callbacks)
    print('Averaging predictions...')        
    #averaging -outputs
    top_k = []
    for ck_file in os.listdir(model_dir):
        top_k.append((float(ck_file.split('-')[1][:5]), ck_file))
    top_k.sort()
    val_pred_y = np.zeros_like(val_y, dtype=float)
    test_y = np.zeros((tx.shape[0],1))
    print(top_k)
    CHKPOINTS = len(top_k) if len(top_k) <  config['top_k_checkpoints'] else config['top_k_checkpoints']
    for _,model_file in top_k[:config['top_k_checkpoints']]:
        model.load_weights(os.path.abspath(f'{model_dir}/{model_file}'))
        val_pred_y += model.predict(val_x).reshape(-1)
        test_y += np.expand_dims(model.predict(tx).reshape(-1), -1)


    files = glob.glob(os.path.abspath(model_dir)+'/*')
    for f in files:
        os.remove(f)    
    return val_pred_y/CHKPOINTS, test_y/CHKPOINTS


##################### HELPER FUNC ENDS ####################
def preprocess(train_df, test_df, embeddings_index):
    T = Tokenizer(embeddings_index)
    T.fit_texts(train_df.question_text.values)
    T.fit_texts(test_df.question_text.values)
    T.build_vocab()
    return T

def train_full(train_df, test_df, embeddings_index):
    global FOLDS, config
    T = preprocess(train_df, test_df, embeddings_index)
    train_X, train_Y = getxyfromdf(train_df, T)
    test_X, _ = getxyfromdf(test_df, T)      
    model = config['func'](T.embedding_matrix, **config)
    #Utils.search_lr(model,train_X, train_Y)
    print('Splitting stratified k fold')
    splits = list(StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=DATA_SPLIT_SEED).split(train_X, train_Y))    
    
    train_meta = np.zeros_like(train_Y, dtype=float)
    test_meta = np.zeros((test_X.shape[0],1))
    
    for idx, (train_idx, valid_idx) in enumerate(splits):
        tr_x = train_X[train_idx]
        tr_y = train_Y[train_idx]
        val_x = train_X[valid_idx]
        val_y = train_Y[valid_idx]       
        print(f"Running fold {idx}") 
        val_pred_y, test_y = one_round(model,config['model_dir'], tr_x, tr_y, val_x, val_y, test_X, 20)  
        train_meta[valid_idx] = val_pred_y
        test_meta += test_y/FOLDS
    
    columnname = f"targets_{config['func'].__name__}_avg_{config['top_k_checkpoints']}__{config['embedding']}"
    train_df[columnname] = train_meta
    train_df.to_csv("../output/train.csv", index_label ='qid')

    test_df[columnname] = test_meta    
    test_df.to_csv("../output/test.csv", index_label ='qid')    
    return
################ CONFIGURATIONS #################

config = {}
config['embedding'] = 'wiki'
config['seq_len'] = 100
config['bs'] = 512
config['dropout'] = 0.25
config['internal_dim'] = 100
config['lr'] = 0.01
config['func'] = lstm_gru_attn #function is used as a variable
config['top_k_checkpoints'] = 3 #for checkpoint ensemble-averaging (maximum of epochs)
config['number_to_x'] = False
config['block_math'] = False
config['swear_word_map'] = False
config['n_t_replace'] = False
config['lemmatize'] = False
config['two_split'] = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Map wikipedia categories to kngine entity types')
    for each in config:
        parser.add_argument(f'--{each}', default = config[each], type = type(config[each]), required=False)
    args = vars(parser.parse_args())
    print('Configuration....')
    for each in args:
        print(each, args[each])
        config[each] = args[each]
    print('#'*20)
    
    config['model_dir'] = '__'.join(f'{k}_{v}' if k != 'func' else f'{k}_{v.__name__}' for k,v in args.items())
    if not os.path.exists(config['model_dir']):
        os.mkdir(config['model_dir'])
    EMBEDDING_FILE = None
    if config['embedding'] == 'glove':
        EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
    elif config['embedding'] == 'paragram':
        EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
    elif config['embedding'] == 'wiki':
        EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
    

    train_df = pd.read_csv("../output/small_train.csv", index_col="qid")
    test_df = pd.read_csv("../output/small_test.csv", index_col="qid")
    # sample_train = train_df.sample(1000)
    # sample_test = test_df.sample(1000)
    # sample_train.to_csv("../output/small_train.csv")
    # sample_test.to_csv("../output/small_test.csv")
    # assert False

    if EMBEDDING_FILE and os.path.exists(EMBEDDING_FILE):
        print('Loading %s' %EMBEDDING_FILE )
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf-8", errors='ignore') if len(o)>100)
    else:
        embeddings_index = {}    
    print('Done loading embedding file')
    train_full(train_df, test_df,embeddings_index )
    