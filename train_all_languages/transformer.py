import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def create_transformer(num_layers, num_heads, embedding_dim, ff_dim,
                       input_vocab_size, target_vocab_size,
                       dropout_rate=0.1, max_tokens=128):
  input_tokens = keras.Input(shape=(None,), dtype=tf.int64, name='input_tokens')
  target_tokens = keras.Input(shape=(None,), dtype=tf.int64, name='target_tokens')

  padding_mask, look_ahead_mask = create_masks(input_tokens, target_tokens)

  encoder_out = encoder(input_tokens, padding_mask, num_layers, num_heads,
                        embedding_dim, ff_dim, input_vocab_size, dropout_rate, max_tokens)
  
  decoder_out, attentions = decoder(target_tokens, encoder_out, look_ahead_mask, padding_mask,
                                    num_layers, num_heads, embedding_dim, ff_dim,
                                    target_vocab_size, dropout_rate, max_tokens)
  
  out = layers.Dense(target_vocab_size)(decoder_out)

  transformer = keras.Model(inputs=[input_tokens, target_tokens], outputs=[out, attentions])

  return transformer


def encoder(x, padding_mask, num_layers, num_heads, 
            embedding_dim, ff_dim, vocab_size,
            dropout_rate, max_tokens):
  seq_len = tf.shape(x)[1]

  x = layers.Embedding(vocab_size, embedding_dim)(x)
  x *= tf.math.sqrt(tf.cast(embedding_dim, tf.float32))
  x += positional_encoding(max_tokens, embedding_dim)[:, :seq_len, :]

  x = layers.Dropout(dropout_rate)(x)

  for i in range(num_layers):
    x = encoder_layer(x, padding_mask, num_heads, embedding_dim, ff_dim, dropout_rate)
  
  return x


def decoder(x, encoder_out, look_ahead_mask, padding_mask,
            num_layers, num_heads, embedding_dim, ff_dim, vocab_size,
            dropout_rate, max_tokens):  
  seq_len = tf.shape(x)[1]
  attentions = []

  x = layers.Embedding(vocab_size, embedding_dim)(x)
  x *= tf.math.sqrt(tf.cast(embedding_dim, tf.float32))
  x += positional_encoding(max_tokens, embedding_dim)[:, :seq_len, :]

  x = layers.Dropout(dropout_rate)(x)

  for i in range(num_layers):
    x, attn = decoder_layer(x, encoder_out, look_ahead_mask, padding_mask,
                            num_heads, embedding_dim, ff_dim, dropout_rate)
    attentions.append(attn)

  return x, attentions


def encoder_layer(x, padding_mask, num_heads, embedding_dim, ff_dim, dropout_rate):
  x, _ = multi_head_attention(x, x, padding_mask, num_heads, embedding_dim, dropout_rate)
  x = feed_forward(x, embedding_dim, ff_dim, dropout_rate)
  return x


def decoder_layer(x, encoder_out, look_ahead_mask, padding_mask, num_heads, embedding_dim, ff_dim, dropout_rate):
  x, _ = multi_head_attention(x, x, look_ahead_mask, num_heads, embedding_dim, dropout_rate)
  x, attn = multi_head_attention(x, encoder_out, padding_mask, num_heads, embedding_dim, dropout_rate)
  x = feed_forward(x, embedding_dim, ff_dim, dropout_rate)
  return x, attn


def multi_head_attention(q, k, mask, num_heads, embedding_dim, dropout_rate):
  mha_out, attn = layers.MultiHeadAttention(num_heads, embedding_dim)(q, k, attention_mask=mask, return_attention_scores=True)
  mha_out = layers.Dropout(dropout_rate)(mha_out)
  mha_out = layers.LayerNormalization(epsilon=1e-6)(q + mha_out)
  return mha_out, attn


def feed_forward(x, embedding_dim, ff_dim, dropout_rate):
  ff_out = layers.Dense(ff_dim, activation='relu')(x)
  ff_out = layers.Dense(embedding_dim)(ff_out)
  ff_out = layers.Dropout(dropout_rate)(ff_out)
  ff_out = layers.LayerNormalization(epsilon=1e-6)(x + ff_out)
  return ff_out


def positional_encoding(position, embedding_dim):
  position_range = np.arange(position)[:, np.newaxis]
  i = np.arange(embedding_dim)[np.newaxis, :]

  angle_rads = position_range / (np.power(10000, (2 * (i // 2)) / np.float32(embedding_dim)))
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]
  
  return tf.cast(pos_encoding, dtype=tf.float32)


def create_masks(input_tokens, target_tokens):
  padding_mask = create_padding_mask(input_tokens)
  look_ahead_mask = create_look_ahead_mask(tf.shape(target_tokens)[1])
  
  target_padding_mask = create_padding_mask(target_tokens)
  look_ahead_mask = tf.minimum(target_padding_mask, look_ahead_mask)

  return padding_mask, look_ahead_mask


def create_padding_mask(seq):
  return tf.cast(tf.math.logical_not(tf.math.equal(seq, 0)), tf.float32)[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
  n = tf.cast(size * (size+1) / 2, dtype=tf.int32)
  return tfp.math.fill_triangular(tf.ones((n,), dtype=tf.float32), upper=False)
