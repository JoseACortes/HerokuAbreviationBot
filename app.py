import numpy as np
import pandas as pd
import tensorflow as tf

import AbreivBot.Functions as fnc
from AbreivBot.Models.BaseModels import BaseModel

import string

possible_chars = string.ascii_letters + string.digits + ' .@'

model = BaseModel(dictionary=possible_chars)
model.load_weights('AbreivBot/Models/TrainedModels/BaseModel')


from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    data = request.get_json()
    data = fnc.vecotrize_list(data['input'])
    
    output = model.predict(data)
    
    output = fnc.unvectorize_list(output)
    
    return jsonify(output.tolist())

import os

if __name__ == '__main__':
    app.run()
    # app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))