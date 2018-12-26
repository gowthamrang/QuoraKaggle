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
def lstm_gru_attn(embedding_matrix, **config):
    inp = Input(shape=(config['seq_len'],)) 
    x = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights = [embedding_matrix], trainable=True)(inp)    #fine-tuning makes a lot of difference
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