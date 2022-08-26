import tensorflow as tf
from tensorflow.keras import layers
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import stanza
import os

from train import test_step
from preprocessing import language_codes
from utils import load_pickle_as_layer


def accuracy(transformer, test_dataset, input_vectorizer, target_vectorizer):
  # calculate test batch accuracies
  test_accuracies = []
  for input_batch, target_batch in test_dataset:
    input_batch = input_vectorizer(input_batch)
    target_batch = target_vectorizer(target_batch)
    _, acc = test_step(transformer, input_batch, target_batch)
    test_accuracies.append(acc)

  # average batch accuracies
  test_accuracy = np.mean(test_accuracies) * 100

  return test_accuracy


def bleu_score(transformer, test_dataset, input_vectorizer, target_vectorizer, target_token_from_index):
  total_score = 0
  count = 0

  for input_batch, target_batch in test_dataset:
    input_batch_vec = input_vectorizer(input_batch)
    target_batch_vec = np.array(target_vectorizer(target_batch))
    target_batch_vec[target_batch_vec >= 9664] = 1
    preds, _ = transformer([input_batch_vec, target_batch_vec], training=False)

    preds = np.argmax(preds, axis=-1)
    preds = target_token_from_index(preds).numpy()
    
    for pred, true in zip(preds, target_batch):
      count += 1
      pred = [token.decode() for token in pred]
      try:
        end_index = pred.index('[END]')
      except:
        end_index = len(pred)

      pred = pred[:end_index]
      true = str(true.numpy().decode()).split()
      true.remove('[START]')
      true.remove('[END]')
      total_score += sentence_bleu([pred], true)

  return total_score / count


def load_model(model_name):
  input_language, target_language = model_name.split('_to_')
  transformer = tf.saved_model.load(f'./saved_models/{model_name}')
  text_processors_dir = f'./text_processors/{model_name}'
  stanza.download(language_codes[input_language])
  input_tokenizer = stanza.Pipeline(lang=language_codes[input_language], processors='tokenize')
  input_vectorizer = load_pickle_as_layer(f'{text_processors_dir}/input_vectorizer.pkl', layers.TextVectorization)
  target_vectorizer = load_pickle_as_layer(f'{text_processors_dir}/target_vectorizer.pkl', layers.TextVectorization)
  target_token_from_index = load_pickle_as_layer(f'{text_processors_dir}/target_token_from_index.pkl', layers.StringLookup)
  return transformer, input_tokenizer, input_vectorizer, target_vectorizer, target_token_from_index, target_language


def get_all_accuracies():
  for model_name in os.listdir('saved_models'):
    transformer, input_tokenizer, input_vectorizer, target_vectorizer, target_token_from_index, target_language = load_model(model_name)
    test_accuracy = accuracy()
    with open('model_stats.txt', 'a') as f:
      f.write(f'{model_name}: {test_accuracy:.3f}\n')
