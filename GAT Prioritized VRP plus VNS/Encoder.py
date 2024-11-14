import tensorflow as tf
from tensorflow import keras 
from keras import layers
import numpy as np
from Attention_Layers import *

class Encoder(layers.Layer):
    def __init__(self, input_dim, num_heads, num_layers, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.init_embedding_layer = layers.Dense(self.input_dim, name="init_embedding")

        self.MhGATs = [Graph_Attention_Module(self.input_dim, self.num_heads) for _ in range(self.num_layers)]

    def call(self, x_custs, x_depots, mask=None):
        N = tf.shape(x_custs)[0] + tf.shape(x_depots)[0]

        x_c = self.init_embedding_layer(x_custs)
        x_d = self.init_embedding_layer(x_depots)

        for i in range(self.num_layers):
            if mask is not None:
                x_c = self.MhGATs[i](x_c, mask)
            else:
                x_c = self.MhGATs[i](x_c, mask=None)
            x_d = self.MhGATs[i](x_d) ## mask for depot should be different from the mask for customers
        
        # x_c = tf.reduce_sum(x_c, axis=0) 
        # x_c = tf.expand_dims(x_c, axis=0)

        x_c = tf.reduce_sum(x_c, axis=1) ## (batch_size, features)
        x_c = tf.expand_dims(x_c, axis=1) ## (batch_size, 1, features) for each batch sample we require one vector in length of features

        embd = tf.add(x_d, x_c)/(N.numpy()+1)

        return embd
