import tensorflow as tf

class BaseModel(tf.keras.Model):
    
    def __init__(self, dictionary, **kwargs):
        super(BaseModel, self).__init__(**kwargs)
        self.dictionary = dictionary
        self.layer_to_initial_embedding = tf.keras.layers.Embedding(input_dim=len(dictionary), output_dim=1)
        self.embedding_flattener = tf.keras.layers.Flatten()
        self.lstm = tf.keras.layers.LSTM(len(dictionary), return_sequences=True, return_state=True)
        self.readout = tf.keras.layers.Dense(65, activation='softmax')
    
    def call(self, inputs, training=False):
        embedding = self.layer_to_initial_embedding(inputs[:,0, :], training=training)
        embedding = tf.squeeze(embedding, [-1])
        lstm_out, state_h, state_c = self.lstm(inputs = inputs, training=training, initial_state=[embedding, embedding])
        readout = tf.map_fn(lambda x:self.readout(x, training=training), lstm_out)
        return readout