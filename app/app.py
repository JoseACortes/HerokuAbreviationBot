

import numpy as np
import pandas as pd
import tensorflow as tf

import AbreivBot.Functions as fnc
from AbreivBot.Models.BaseModels import BaseModel

import string

possible_chars = string.ascii_letters + string.digits + ' .@'

model = BaseModel(dictionary=possible_chars)
model.load_weights('AbreivBot/Models/TrainedModels/BaseModel')


from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def predict():
    data = request.args.get('data')
    data = data.split(',')[:-1]
    data = fnc.vecotrize_list(data)
    
    output = []

    with tf.device('/device:CPU:0'):
        output = model.predict(data)
    
    output = fnc.unvectorize_list(output)
    
    return ','.join(output)