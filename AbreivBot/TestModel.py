import numpy as np
import pandas as pd
import string
import tensorflow as tf
import AbreivBot.Functions as fnc

possible_chars = string.ascii_letters + string.digits + ' .@'

def __init__(self):
    pass
def test(model, test_dataframe, possible_chars = possible_chars, verbose=0):

    possible_chars = possible_chars

    charlens = test_dataframe.WordLength.unique()
    
    results = {}
    
    for charlen in charlens:
        traincharcut = test_dataframe[test_dataframe['WordLength'] == charlen]

        train_X = traincharcut['Word']
        train_X = train_X.loc[:].to_numpy()
        train_X = np.array([fnc.one_hot_encode_string(x, possible_chars) for x in train_X])
        train_X = np.float64(train_X)

        train_y = traincharcut['parse_ylabel']
        train_y = train_y.loc[:].to_numpy()
        train_y = np.array([fnc.one_hot_encode_string(x, possible_chars) for x in train_y])
        train_y = np.float64(train_y)
        
        with tf.device('/device:CPU:0'):

            if len(train_X) == 1:
                
                results[charlen] = model.test_on_batch(train_X, train_y)
            
            elif len(train_X) > 1:
            
                results[charlen] = model.test_on_batch(train_X, train_y)

    return results