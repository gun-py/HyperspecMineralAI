import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, Dropout, Conv1D, MaxPooling1D, Concatenate, Add,
                                     BatchNormalization, GlobalAveragePooling1D, Reshape,
                                     Layer, MultiHeadAttention, LayerNormalization, GaussianNoise,
                                     LSTM)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from spektral.layers import GINConv, GATConv
from spektral.utils import normalized_adjacency
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from kerastuner.tuners import RandomSearch

class SEBlock(Layer):
    def __init__(self, reduction_ratio=16, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
    
    def build(self, input_shape):
        channels = input_shape[-1]
        self.global_avg_pool = GlobalAveragePooling1D()
        self.dense1 = Dense(channels // self.reduction_ratio, activation='relu', use_bias=False)
        self.dense2 = Dense(channels, activation='sigmoid', use_bias=False)
        super(SEBlock, self).build(input_shape)
    
    def call(self, inputs):
        x = self.global_avg_pool(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = K.expand_dims(x, axis=1)
        return inputs * x

class BallQueryAttention(Layer):
    def __init__(self, radius, **kwargs):
        super(BallQueryAttention, self).__init__(**kwargs)
        self.radius = radius
    
    def build(self, input_shape):
        super(BallQueryAttention, self).build(input_shape)
    
    def call(self, inputs):
        x, adj = inputs
        distances = tf.sqrt(tf.reduce_sum(tf.square(tf.expand_dims(x, 1) - tf.expand_dims(x, 0)), axis=-1))
        mask = tf.cast(distances <= self.radius, tf.float32)
        attention_scores = tf.nn.softmax(mask, axis=-1)
        attended_features = tf.matmul(attention_scores, x)
        return attended_features

class ResidualAttentionBlock(Layer):
    def __init__(self, units, num_heads, **kwargs):
        super(ResidualAttentionBlock, self).__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
    
    def build(self, input_shape):
        self.attention = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.units)
        self.dense = Dense(self.units, activation='relu')
        self.batch_norm = BatchNormalization()
        super(ResidualAttentionBlock, self).build(input_shape)
    
    def call(self, inputs):
        x, mask = inputs
        att_output = self.attention(x, x, x, attention_mask=mask)
        x_res = Add()([x, att_output])
        x_res = self.batch_norm(x_res)
        x_res = self.dense(x_res)
        return x_res

class CBAM(Layer):
    def __init__(self, reduction_ratio=16, **kwargs):
        super(CBAM, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
    
    def build(self, input_shape):
        channels = input_shape[-1]
        self.channel_avg_pool = GlobalAveragePooling1D()
        self.channel_max_pool = GlobalAveragePooling1D()
        self.dense1 = Dense(channels // self.reduction_ratio, activation='relu', use_bias=False)
        self.dense2 = Dense(channels, activation='sigmoid', use_bias=False)
        self.spatial_conv = Conv1D(1, 7, padding='same', activation='sigmoid')
        super(CBAM, self).build(input_shape)
    
    def call(self, inputs):
        avg_pool = self.channel_avg_pool(inputs)
        max_pool = self.channel_max_pool(inputs)
        channel_attention = self.dense2(self.dense1(avg_pool) + self.dense1(max_pool))
        channel_attention = K.expand_dims(channel_attention, axis=1)
        x = inputs * channel_attention
        spatial_attention = self.spatial_conv(x)
        spatial_attention = K.repeat_elements(spatial_attention, x.shape[-1], axis=-1)
        x = x * spatial_attention
        return x

class GhostConv(Layer):
    def __init__(self, filters, kernel_size=1, **kwargs):
        super(GhostConv, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
    
    def build(self, input_shape):
        self.conv1 = Conv1D(self.filters // 2, self.kernel_size, padding='same', use_bias=False)
        self.conv2 = Conv1D(self.filters // 2, self.kernel_size, padding='same', use_bias=False)
        self.batch_norm = BatchNormalization()
        super(GhostConv, self).build(input_shape)
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.batch_norm(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        return x

class HierarchicalTransformerEncoderLayer(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, **kwargs):
        super(HierarchicalTransformerEncoderLayer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.embed_dim = embed_dim
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
    
    def call(self, inputs):
        x, mask = inputs
        attn_output = self.att(x, x, x, attention_mask=mask)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + ffn_output)

def create_inception_module(input_shape: Tuple[int]) -> Model:
    input_layer = Input(shape=input_shape)
    conv1x1 = Conv1D(64, 1, activation='relu')(input_layer)
    conv1x1 = BatchNormalization()(conv1x1)
    conv3x3 = Conv1D(64, 3, activation='relu', padding='same')(input_layer)
    conv3x3 = BatchNormalization()(conv3x3)
    conv5x5 = Conv1D(64, 5, activation='relu', padding='same')(input_layer)
    conv5x5 = BatchNormalization()(conv5x5)
    maxpool = MaxPooling1D(3, strides=1, padding='same')(input_layer)
    inception_module = Concatenate(axis=-1)([conv1x1, conv3x3, conv5x5, maxpool])
    inception_module = BatchNormalization()(inception_module)
    inception_module = SEBlock()(inception_module)
    x = GlobalAveragePooling1D()(inception_module)
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    return Model(inputs=input_layer, outputs=x)

def create_attention_module(input_shape: Tuple[int]) -> Model:
    input_layer = Input(shape=input_shape)
    transformer_layer = HierarchicalTransformerEncoderLayer(embed_dim=input_shape[1], num_heads=4, ff_dim=64)
    self_attention = transformer_layer([input_layer, None])
    channel_attention = SEBlock()(input_layer)
    spatial_attention = Conv1D(1, 7, padding='same', activation='sigmoid')(input_layer)
    attention_output = Concatenate()([self_attention, channel_attention, spatial_attention])
    attention_output = BatchNormalization()(attention_output)
    return Model(inputs=input_layer, outputs=attention_output)

def create_graph_siamese_network(input_shape: Tuple[int], radius: float) -> Model:
    input_node = Input(shape=input_shape)
    input_adj = Input(shape=(None, None))
    x = BallQueryAttention(radius)([input_node, input_adj])
    x = GhostConv(128)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = create_inception_module(input_shape)(x)
    x = ResidualAttentionBlock(64, num_heads=4)([x, None])
    x = GINConv(64)([x, input_adj])
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = ResidualAttentionBlock(32, num_heads=4)([x, None])
    x = GINConv(32)([x, input_adj])
    x = BatchNormalization()(x)
    x = CBAM()(x)
    return Model(inputs=[input_node, input_adj], outputs=x)

def create_autoencoder(input_shape: Tuple[int]) -> Model:
    input_layer = Input(shape=input_shape)
    x = create_attention_module(input_shape)(input_layer)
    x = GhostConv(128)(x)
    x = LSTM(64, return_sequences=True)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    encoded = Dense(32, activation='relu')(x)
    x = Dense(64, activation='relu')(encoded)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    decoded = Dense(np.prod(input_shape), activation='sigmoid')(x)
    decoded = Reshape(input_shape)(decoded)
    return Model(inputs=input_layer, outputs=decoded)
