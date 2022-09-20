import tensorflow as tf
from tensorflow.keras import optimizers, metrics
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import time
import json
import os

from transformer import create_transformer
from preprocessing import load_dataset
from utils import save_layer_as_pickle


sparse_cce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


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
      train((1, 8, 512, 512), model_name, train_dataset, test_dataset, input_vectorizer,\
        target_vectorizer, target_token_from_index, input_vocab_size, target_vocab_size)
      

def train(transformer_args, model_name, train_dataset, test_dataset, input_vectorizer, target_vectorizer, target_token_from_index, input_vocab_size, target_vocab_size): 
  epochs = max(5, int(np.log(len(train_dataset)) / np.log(0.85)) + 61)

  optimizer = optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

  train_loss_mean = metrics.Mean()
  train_accuracy_mean = metrics.Mean()
  test_loss_mean = metrics.Mean()
  test_accuracy_mean = metrics.Mean()
  train_bleu_mean = metrics.Mean()
  test_bleu_mean = metrics.Mean()

  transformer = create_transformer(*transformer_args, input_vocab_size, target_vocab_size)
  
  train_losses, train_accuracies, train_bleus = [], [], []
  test_losses, test_accuracies, test_bleus = [], [], []
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
    train_bleu_mean.reset_states()
    test_bleu_mean.reset_states()

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

      train_loss, train_accuracy = train_step(transformer, optimizer, input_batch, target_batch)
      test_loss, test_accuracy = test_step(transformer, test_input_batch, test_target_batch)

      train_bleu = batch_bleu_score(transformer, target_token_from_index, input_batch, target_batch)
      test_bleu = batch_bleu_score(transformer, target_token_from_index, test_input_batch, test_target_batch)

      train_loss_mean(train_loss)
      test_loss_mean(test_loss)
      train_accuracy_mean(train_accuracy)
      test_accuracy_mean(test_accuracy)
      train_bleu_mean(train_bleu)
      test_bleu_mean(test_bleu)

      train_losses.append(train_loss)
      test_losses.append(test_loss)
      train_accuracies.append(train_accuracy)
      test_accuracies.append(test_accuracy)
      train_bleus.append(train_bleu)
      test_bleus.append(test_bleu)

      remaining_time, avg_batch_time = get_remaining_time(start, completed_batches, total_batches, avg_batch_time)
      completed_batches += 1

      epoch_str = f'\rEpoch {epoch+1} of {epochs} | '
      epoch_str += f'Progress: {i+1}/{len(train_dataset)} | '
      epoch_str += f'Metrics: Train / Test - '
      epoch_str += f'Loss: {train_loss_mean.result():.3f} / {test_loss_mean.result():.3f} - '
      epoch_str += f'Accuracy: {(train_accuracy_mean.result()*100):.3f}% / {(test_accuracy_mean.result()*100):.3f}% - '
      epoch_str += f'BLEU: {(train_bleu_mean.result()):.3f} / {(test_bleu_mean.result()):.3f} | '
      epoch_str += f'Remaining Time: {remaining_time}        '
      
      print(epoch_str, end='')

    print()

  print()

  save_model(transformer, model_name, input_vectorizer,target_vectorizer, target_token_from_index,
             (train_losses, test_losses, train_accuracies, test_accuracies))
  calculate_and_save_test_metrics(transformer, model_name, test_dataset, input_vectorizer,
                                      target_vectorizer, target_token_from_index)


def train_step(transformer, optimizer, inputs, targets):
  with tf.GradientTape() as tape:
    preds, _, targets_real = make_predictions(transformer, inputs, targets, training=True)
    loss = transformer_loss(targets_real, preds)

  gradients = tape.gradient(loss, transformer.trainable_variables)
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

  return loss, batch_accuracy(targets_real, preds)


def test_step(transformer, inputs, targets):
  preds, _, targets_real = make_predictions(transformer, inputs, targets)
  loss = transformer_loss(targets_real, preds)
  return loss, batch_accuracy(targets_real, preds)


def make_predictions(transformer, inputs, targets, training=False):
  targets_in, targets_real = targets[:, :-1], targets[:, 1:]
  preds, attn = transformer([inputs, targets_in], training=training)
  return preds, attn, targets_real


def transformer_loss(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss = sparse_cce_loss(real, pred)
  
  mask = tf.cast(mask, dtype=loss.dtype)
  loss *= mask

  return tf.reduce_sum(loss) / tf.reduce_sum(mask)


def batch_accuracy(real, pred):
  accuracies = tf.equal(real, tf.argmax(pred, axis=2))

  mask = tf.math.logical_not(tf.math.equal(real, 0))
  accuracies = tf.math.logical_and(mask, accuracies)

  accuracies = tf.cast(accuracies, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  
  return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


def batch_bleu_score(transformer, target_token_from_index, inputs, targets):
  preds, _, targets_real = make_predictions(transformer, inputs, targets)

  # convert predictions to tokens
  preds = np.argmax(preds, axis=-1)
  preds = target_token_from_index(preds).numpy()
  targets_real = target_token_from_index(targets_real).numpy()
  
  # calculate bleu scores for the current batch
  total_score = 0
  for real, pred in zip(targets_real, preds):
    
    # ignore predicted tokens past [END]
    pred = [token.decode() for token in pred]
    try:
      end_index = pred.index('[END]')
    except:
      end_index = len(pred)
    pred = pred[:end_index]

    # ignore real tokens past [END]
    real = [token.decode() for token in real]
    try:
      end_index = real.index('[END]')
    except:
      end_index = len(real)
    real = real[:end_index]

    # calculate bleu score
    total_score += sentence_bleu([pred], real)

  return total_score / len(targets_real)


def calculate_and_save_test_metrics(transformer, model_name, test_dataset, input_vectorizer, target_vectorizer, target_token_from_index):
  # calculate test batch losses and accuracies
  test_accuracies, test_bleus = [], []
  for input_batch, target_batch in test_dataset:
    input_batch = input_vectorizer(input_batch)
    target_batch = target_vectorizer(target_batch)
    preds, _, targets_real = make_predictions(transformer, input_batch, target_batch)
    acc = batch_accuracy(targets_real, preds)
    bleu = batch_bleu_score(transformer, target_token_from_index, input_batch, target_batch)
    test_accuracies.append(acc)
    test_bleus.append(bleu)

  test_accuracy = np.mean(test_accuracies) * 100
  test_bleu = np.mean(test_bleus)

  print(f'{model_name} metrics | accuracy: {test_accuracy}% | bleu: {test_bleu}')

  append_metrics_to_file(model_name, test_accuracy, test_bleu)


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


def append_metrics_to_file(model_name, accuracy, bleu_score):
  metrics_json_file_name = f'accuracy_and_bleu.json'
  if os.path.exists(metrics_json_file_name):
    metrics_json = json.load(open(metrics_json_file_name))
  else:
    metrics_json = {}

  metrics_json[model_name] = {
    'accuracy': accuracy,
    'bleu': bleu_score
  }

  json.dump(metrics_json, open(metrics_json_file_name, 'w'))


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
