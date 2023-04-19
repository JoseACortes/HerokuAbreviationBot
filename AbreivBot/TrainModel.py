import numpy as np
import pandas as pd
import string
import tensorflow as tf
import AbreivBot.Functions as fnc

possible_chars = string.ascii_letters + string.digits + ' .@'

def __init__():
    pass
def train(model, train_dataframe, possible_chars = possible_chars, verbose=0):

    possible_chars = possible_chars

    charlens = train_dataframe.WordLength.unique()
    
    hist = {}
    
    for charlen in charlens:
        traincharcut = train_dataframe[train_dataframe['WordLength'] == charlen]

        train_X = traincharcut['Word']
        train_X = train_X.loc[:].to_numpy()
        train_X = np.array([fnc.one_hot_encode_string(x, possible_chars) for x in train_X])
        train_X = np.float64(train_X)

        train_y = traincharcut['parse_ylabel']
        train_y = train_y.loc[:].to_numpy()
        train_y = np.array([fnc.one_hot_encode_string(x, possible_chars) for x in train_y])
        train_y = np.float64(train_y)
        
        with tf.device('/device:GPU:0'):

            if len(train_X) == 1:
            
                hist[charlen] = model.fit(train_X, train_y, epochs=50, verbose=verbose)
            
            elif len(train_X) > 1:
            
                hist[charlen] = model.fit(train_X, train_y, epochs=50, verbose=verbose, validation_split=0.2)

    return hist