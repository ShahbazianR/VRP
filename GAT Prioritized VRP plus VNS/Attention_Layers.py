import tensorflow as tf
from tensorflow import keras 
from keras import layers
import numpy as np

class GraphAttention(layers.Layer):
    def __init__(self, embedding_dim:int, **kwargs):
        super().__init__(**kwargs)
        self.out_dim = embedding_dim

        # Weight Matrices
        self.w_query = layers.Dense(self.out_dim, use_bias=True)
        self.w_key = layers.Dense(self.out_dim, use_bias=True)
        self.w_value = layers.Dense(self.out_dim, use_bias=True)

        self.w_out = layers.Dense(self.out_dim, use_bias=True)

    def call(self, query_, key_, mask=None):
        # Linearly transform node states
        # Then, split regarding the batch size
        # query => h0_i; 
        # key => h0_j s.that j in NB_i
        query = self.w_query(query_)
        key = self.w_key(key_)

        ## computing u_{ij}
        compatibility = tf.matmul(query, key, transpose_b=True)
        # rescaling
        compatibility /= tf.math.sqrt(tf.cast(tf.shape(key)[-1], tf.float32))

        if mask is not None:
                mask = mask[:, 1:, 1:]
                # we use tf.where since 0*-np.inf returns nan, but not -np.inf
                compatibility = tf.where(mask, tf.ones_like(compatibility) * (-np.inf), compatibility)
                
        ## a_ij
        attention_scores = tf.nn.softmax(compatibility, axis=-1)
        # Replace NaN (from softmax) by zeros 
        attention_scores = tf.where(tf.math.is_nan(attention_scores), tf.zeros_like(attention_scores), attention_scores)
        # sum all query embeddings according to attention scores
        w_v_key = self.w_value(key)
        attention_scores = tf.matmul(attention_scores, w_v_key)

        output = self.w_out(attention_scores)
        return output


class Graph_Attention_Module(layers.Layer):
    def __init__(self, output_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.n_heads = num_heads
        self.attention_modules = [GraphAttention(embedding_dim=output_dim) for _ in range(self.n_heads)]
        self.fc1 = layers.Dense(output_dim, name="fc1")
        self.fc2 = layers.Dense(output_dim, name="fc2")

    def call(self, x, mask=None):
        # multi-head self-attention
        results = []
        for i in range(self.n_heads):
            gat_out = self.attention_modules[i](query_=x, key_=x, mask=mask)
            fc1_out = self.fc1(gat_out)
            h_hat = layers.Add()([x, fc1_out])
            h_hat = layers.BatchNormalization()(h_hat)
            relu1_out = tf.keras.activations.relu(h_hat)

            h = layers.Add()([h_hat, relu1_out])
            fc2_out = self.fc2(h)
            results.append(fc2_out)

        return tf.reduce_sum(results, axis=0)
