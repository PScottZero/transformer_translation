import tensorflow as tf
from tensorflow.keras import optimizers, metrics
import numpy as np
import time
import os

from transformer import create_transformer
from preprocessing import load_dataset
from utils import save_layer_as_pickle


def train_all_languages():
  file_names = sorted(os.listdir('sentence_pairs'))
  languages = [file_name.replace('.txt', '') for file_name in file_names if file_name != '.DS_Store']
  
  for language in languages:
    language = language.replace('.txt', '')
    for invert in [False, True]:
      input_language = language if invert else 'english'
      target_language = 'english' if invert else language
      model_name = f'{input_language}_to_{target_language}'
      train_dataset, test_dataset, input_vectorizer, target_vectorizer,\
        target_token_from_index, input_vocab_size, target_vocab_size = load_dataset(language, invert)
      train((3, 8, 512, 512), model_name, train_dataset, test_dataset, input_vectorizer,\
        target_vectorizer, target_token_from_index, input_vocab_size, target_vocab_size)


def train(transformer_args, model_name, train_dataset, test_dataset, input_vectorizer, target_vectorizer, target_token_from_index, input_vocab_size, target_vocab_size): 
  epochs = min(5, np.ceil(10000 // len(train_dataset)))

  sparse_cce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
  optimizer = optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

  train_loss_mean = metrics.Mean()
  train_accuracy_mean = metrics.Mean()
  test_loss_mean = metrics.Mean()
  test_accuracy_mean = metrics.Mean()

  transformer = create_transformer(*transformer_args, input_vocab_size, target_vocab_size)
  
  train_losses, train_accuracies = [], []
  test_losses, test_accuracies = [], []
  avg_batch_time = 0
  completed_batches = 0
  total_batches = epochs * len(train_dataset)

  test_dataset_iter = test_dataset.as_numpy_iterator()

  print(f'\n\n============ training {model_name} ============\n\n')
  
  for epoch in range(epochs):

    train_loss_mean.reset_states()
    test_loss_mean.reset_states()
    train_accuracy_mean.reset_states()
    test_accuracy_mean.reset_states()

    for i, (input_batch, target_batch) in enumerate(train_dataset):
      start = time.time()

      input_batch = input_vectorizer(input_batch)
      target_batch = target_vectorizer(target_batch)

      try:
        test_input_batch, test_target_batch = test_dataset_iter.next()
      except:
        test_dataset_iter = test_dataset.as_numpy_iterator()
        test_input_batch, test_target_batch = test_dataset_iter.next()

      test_input_batch = input_vectorizer(test_input_batch)
      test_target_batch = target_vectorizer(test_target_batch)

      train_loss, train_accuracy = train_step(transformer, optimizer, sparse_cce_loss, input_batch, target_batch)
      test_loss, test_accuracy = test_step(transformer, sparse_cce_loss, test_input_batch, test_target_batch)

      train_loss_mean(train_loss)
      test_loss_mean(test_loss)
      train_accuracy_mean(train_accuracy)
      test_accuracy_mean(test_accuracy)

      train_losses.append(train_loss)
      test_losses.append(test_loss)
      train_accuracies.append(train_accuracy)
      test_accuracies.append(test_accuracy)

      remaining_time, avg_batch_time = get_remaining_time(start, completed_batches, total_batches, avg_batch_time)
      completed_batches += 1

      epoch_str = f'\rEpoch {epoch+1} of {epochs} | '
      epoch_str += f'Progress: {i+1}/{len(train_dataset)} | '
      epoch_str += f'Train Loss: {train_loss_mean.result():.3f} | '
      epoch_str += f'Train Accuracy: {(train_accuracy_mean.result()*100):.3f}% | '
      epoch_str += f'Test Loss: {test_loss_mean.result():.3f} | '
      epoch_str += f'Test Accuracy: {(test_accuracy_mean.result()*100):.3f}% | '
      epoch_str += f'Remaining Time: {remaining_time}        '
      
      print(epoch_str, end='')

    print()

  print()
  save_model(transformer, model_name, input_vectorizer,target_vectorizer, target_token_from_index,
             (train_losses, test_losses, train_accuracies, test_accuracies))
  
  return train_losses, test_losses, train_accuracies, test_accuracies


def train_step(transformer, optimizer, sparse_cce_loss, inputs, targets):
  targets_in = targets[:, :-1]
  targets_real = targets[:, 1:]

  with tf.GradientTape() as tape:
    predictions, _ = transformer([inputs, targets_in], training=True)
    loss = transformer_loss(sparse_cce_loss, targets_real, predictions)

  gradients = tape.gradient(loss, transformer.trainable_variables)
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

  return loss, accuracy(targets_real, predictions)


def test_step(transformer, sparse_cce_loss, inputs, targets):
  targets_in = targets[:, :-1]
  targets_real = targets[:, 1:]
  
  predictions, _ = transformer([inputs, targets_in], training=False)
  loss = transformer_loss(sparse_cce_loss, targets_real, predictions)
  return loss, accuracy(targets_real, predictions)


def transformer_loss(sparse_cce_loss, real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss = sparse_cce_loss(real, pred)
  
  mask = tf.cast(mask, dtype=loss.dtype)
  loss *= mask

  return tf.reduce_sum(loss) / tf.reduce_sum(mask)


def accuracy(real, pred):
  accuracies = tf.equal(real, tf.argmax(pred, axis=2))

  mask = tf.math.logical_not(tf.math.equal(real, 0))
  accuracies = tf.math.logical_and(mask, accuracies)

  accuracies = tf.cast(accuracies, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  
  return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


def save_model(transformer, model_name, input_vectorizer, target_vectorizer, target_token_from_index, metrics):
  tf.saved_model.save(transformer, f'saved_models/{model_name}')

  text_processors_dir = f'text_processors/{model_name}'

  make_dir(text_processors_dir)

  save_layer_as_pickle(input_vectorizer, f'{text_processors_dir}/input_vectorizer.pkl')
  save_layer_as_pickle(target_vectorizer, f'{text_processors_dir}/target_vectorizer.pkl')
  save_layer_as_pickle(target_token_from_index, f'{text_processors_dir}/target_token_from_index.pkl')

  metrics_dir = f'metrics/{model_name}'

  make_dir(metrics_dir)

  train_losses, test_losses, train_accuracies, test_accuracies = metrics

  np.save(f'{metrics_dir}/train_losses.npy', train_losses)
  np.save(f'{metrics_dir}/test_losses.npy', test_losses)
  np.save(f'{metrics_dir}/train_accuracies.npy', train_accuracies)
  np.save(f'{metrics_dir}/test_accuracies.npy', test_accuracies)


def get_remaining_time(batch_start, batch, total_batches, avg_batch_time):
  time_for_epoch = time.time() - batch_start
  avg_batch_time = (avg_batch_time * batch + time_for_epoch) / (batch + 1)
  remaining_batches = total_batches - (batch + 1)
  
  remaining_time = get_time_string(remaining_batches * avg_batch_time)

  return remaining_time, avg_batch_time


def get_time_string(total_seconds):
  hours = int(total_seconds // 3600)
  remainder = total_seconds % 3600
  minutes = int(remainder // 60)
  seconds = round(remainder % 60, 2)
  time_string = ''
  if hours > 0:
    time_string += f'{hours}h '
  if remainder >= 60:
    time_string += f'{minutes}m '
  time_string += f'{seconds}s'
  return time_string


def make_dir(dir):
  if not os.path.isdir(dir):
      os.makedirs(dir)


train_all_languages()
