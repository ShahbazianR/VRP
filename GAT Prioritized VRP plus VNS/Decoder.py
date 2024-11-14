import tensorflow as tf
from tensorflow import keras
from keras import layers, Model

from Encoder import *

class Decoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, num_att_layers, clipping, decodetype="fixed_order", **kwargs):
        super().__init__(**kwargs)
        self.decode_type = decodetype
        self.clip = clipping

        self.embedding_dim = embed_dim
        self.n_heads = num_heads
        self.n_layers = num_att_layers
        
        self.MhGATs = [Graph_Attention_Module(self.embedding_dim, self.n_heads) for _ in range(self.n_layers)]

        self.w_query = layers.Dense(self.embedding_dim, use_bias=True)
        self.w_key = layers.Dense(self.embedding_dim, use_bias=True)

        self.head_depth = self.embedding_dim // self.n_heads
        self.dk_mha_decoder = tf.cast(self.head_depth, tf.float32)

    def MHGAT(self, h_c, x_custs, mask=None):
        x_c = self.MhGATs[0](h_c) ## h_c: (batch_size, n_features) => n_features: h_depot, s1, s2 ,..., sn
        for i in range(1, self.n_layers):
            x_c = self.MhGATs[i](x_c) 

        query = self.w_query(x_c) ## x_custs: (batch_size, n_custs, custs_features)
        key = self.w_key(x_custs)
        compatibility = tf.tanh(tf.matmul(query, key, transpose_b=True)/tf.math.sqrt(self.dk_mha_decoder)) * self.clip
        # compatibility = tf.clip_by_value(compatibility, -self.clip, self.clip)

        if mask is not None:
            mask = mask[:, :, 1:]
            compatibility = tf.where(mask, tf.ones_like(compatibility) * (-np.inf), compatibility)
            # compatibility = tf.where(mask, tf.ones_like(compat_dcmask) * (-np.inf), compat_dcmask)

        return compatibility
    
    def select_cust(self, comptabilities, clusters, depot_id):
        compts = tf.reduce_mean(comptabilities, axis=1)[:, None]
        # log_p = tf.nn.log_softmax(compts, axis=-1)
        # selected = tf.math.argmax(log_p, axis=-1) ## Simple version

        cmpts_np = compts.numpy()
        clusters_np = clusters.numpy()[:, depot_id, :]

        selected_filtered = []
        for batch in range(cmpts_np.shape[0]):
            custs = cmpts_np[batch][0]
            clust = clusters_np[batch]

            tt = tf.where(clust, tf.ones_like(custs) * (-np.inf), custs)
            log_p_tt = tf.nn.log_softmax(tt, axis=-1)
            selected_tt = tf.math.argmax(log_p_tt, axis=-1)
            selected_filtered.append([selected_tt])
        
        selected = tf.convert_to_tensor(selected_filtered)
        return selected
    
    def fixed_order(self, h, c, dc_agent_mask, depot_id, mask=None):
        compats = self.MHGAT(h, c, mask) # (batch_size, 1, output_dim)
        action = self.select_cust(compats, dc_agent_mask, depot_id)


        # action = self.select_cust(compats)

        return action, compats
        