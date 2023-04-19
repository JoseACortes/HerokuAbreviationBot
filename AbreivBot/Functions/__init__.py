import numpy as np

import string

possible_chars = string.ascii_letters + string.digits + ' .@'


def encode_ylabel(abbreviation, word):
    abv = abbreviation.lower()
    wrd = word.lower()
    wrds = [*wrd]
    opt = ['@']*(len(word))
    curr = 0
    for i, a in enumerate(abv):
        remcharsword = wrds[curr:]
        remcharsabv = abv[i:]
        if np.isin(a, remcharsword):
            preind = remcharsword.index(a)
            if len(remcharsword[preind:])<(len(remcharsabv)):
                ind = 0
            else:
                ind = preind
            opt[ind+curr] = abbreviation[i]
            curr = ind+curr+1
        else:
            opt[curr] = abbreviation[i]
            curr = curr+1
    return opt

def decode_ylabel(abbreviation):
    return abbreviation.replace('@', '')

import tensorflow as tf

def one_hot_encode_string(s, alphabet):
    char_to_index = {char: i for i, char in enumerate(alphabet)}
    num_chars = len(alphabet)
    one_hot_matrix = tf.eye(num_chars, dtype=tf.int32)
    chars = list(s)
    indices = [char_to_index.get(char, num_chars-1) for char in chars]
    one_hot_vectors = tf.gather(one_hot_matrix, indices)
    return tf.squeeze(one_hot_vectors)


def one_hot_decode_string(one_hot, alphabet):
    return ''.join([alphabet[np.argmax(x)] for x in one_hot])



def vecotrize_list(words, possible_chars = possible_chars):
    
    X = words
    X = np.array([one_hot_encode_string(x, possible_chars) for x in X])
    X = np.float64(X)
    
    return X

def unvectorize_list(onehot, possible_chars = possible_chars):
    
    X = onehot
    X = np.array([one_hot_decode_string(x, possible_chars) for x in X])
    
    return X