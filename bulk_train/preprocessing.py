import tensorflow as tf
from tensorflow.keras import layers
from collections import Counter
import stanza
import json


language_codes = json.load(open(f'language_codes.json', 'r'))


def load_dataset(language, invert, batch_size=128, train_test_split=0.9, max_tokens=128):
  input_language = language if invert else 'english'
  target_language = 'english' if invert else language

  # open sentence pair dataset
  with open(f'sentence_pairs/{language}.txt', 'r') as f:
    sentence_pairs = [tuple(line.split('\t')[:2]) for line in f.readlines()]

  input_index = 1 if invert else 0
  target_index = 0 if invert else 1

  # split sentence pairs into inputs and targets
  inputs = [sentence_pair[input_index] for sentence_pair in sentence_pairs]
  targets = [sentence_pair[target_index] for sentence_pair in sentence_pairs]

  # load tokenizers
  input_tokenizer, target_tokenizer = load_tokenizers(input_language, target_language)

  # tokenize sentence pairs
  inputs_tokenized, input_vocab_size = tokenize(input_tokenizer, inputs)
  targets_tokenized, target_vocab_size = tokenize(target_tokenizer, targets, add_start_end=True)

  # get indices of sentence pairs with length greater than max_tokens
  inputs_to_remove = [i for i in range(len(inputs_tokenized)) if len(inputs_tokenized[i]) > max_tokens]
  targets_to_remove = [i for i in range(len(targets_tokenized)) if len(targets_tokenized[i]) > max_tokens]
  indices_to_remove = set(inputs_to_remove + targets_to_remove)

  # remove sentence pairs with length greater than max_tokens
  for index in reversed(sorted(list(indices_to_remove))):
    inputs_tokenized.pop(index)
    targets_tokenized.pop(index)

  # define text vectorization layers
  input_vectorizer = layers.TextVectorization(input_vocab_size, standardize=None)
  target_vectorizer = layers.TextVectorization(target_vocab_size, standardize=None)

  # adapt vectorizers to inputs and targets
  input_vectorizer.adapt(inputs_tokenized)
  target_vectorizer.adapt(targets_tokenized)

  # define target index to token mapping layer
  target_token_from_index = layers.StringLookup(vocabulary=target_vectorizer.get_vocabulary(), mask_token='', invert=True)

  # create tf dataset and shuffle
  dataset = tf.data.Dataset.from_tensor_slices((inputs_tokenized, targets_tokenized))
  dataset = dataset.shuffle(len(dataset), seed=1234, reshuffle_each_iteration=False)

  # split dataset into train and test sets
  train_size = int(len(dataset) * train_test_split)
  test_size = len(dataset) - train_size
  train_dataset = dataset.take(train_size).shuffle(train_size).batch(batch_size)
  test_dataset = dataset.skip(train_size).shuffle(test_size).batch(batch_size)

  return train_dataset, test_dataset, input_vectorizer, target_vectorizer, target_token_from_index, input_vocab_size, target_vocab_size


def tokenize(tokenizer, sentences, add_start_end=False):
  docs = [stanza.Document([], text=s) for s in sentences]
  tokenized = tokenizer(docs)
  tokenized = [[token.text.lower() for token in tokens.iter_tokens()] for tokens in tokenized]

  if add_start_end:
    tokenized = [['[START]'] + tokens + ['[END]'] for tokens in tokenized]
  
  # count tokens
  token_counter = Counter()
  for tokens in tokenized:
    for token in tokens:
      token_counter[token] += 1

  tokenized = [' '.join(tokens) for tokens in tokenized]

  return tokenized, len(token_counter) + 2


def load_tokenizers(input_language, target_language):
  input_language_code = language_codes[input_language]
  target_language_code = language_codes[target_language]

  # download stanza data
  stanza.download(input_language_code)
  stanza.download(target_language_code)

  # create tokenizers
  input_tokenizer = stanza.Pipeline(lang=input_language_code, processors='tokenize')
  target_tokenizer = stanza.Pipeline(lang=target_language_code, processors='tokenize')

  return input_tokenizer, target_tokenizer
