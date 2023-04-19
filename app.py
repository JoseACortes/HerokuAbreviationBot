import numpy as np
import pandas as pd
import tensorflow as tf
print('test1')
import AbreivBot.Functions as fnc
from AbreivBot.Models.BaseModels import BaseModel
print('test2')
import string

possible_chars = string.ascii_letters + string.digits + ' .@'

model = BaseModel(dictionary=possible_chars)
model.load_weights('AbreivBot/Models/TrainedModels/BaseModel')
print('test3')

from flask import Flask, request, jsonify
print('test4')
app = Flask(__name__)
print('test5')
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print('a')
    data = request.get_json()
    data = fnc.vecotrize_list(data['input'])
    print('b')
    
    output = model.predict(data)
    
    output = fnc.unvectorize_list(output)
    
    return jsonify(output.tolist())

# import os

if __name__ == '__main__':
    print('test6')
    # app.run()
    app.run(port=5000)
    print('test7')