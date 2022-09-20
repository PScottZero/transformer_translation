import pickle


def save_layer_as_pickle(obj, file):
  pkl = {'config': obj.get_config(), 'weights': obj.get_weights()}
  pickle.dump(pkl, open(file, 'wb'))


def load_pickle_as_layer(file, layer_type):
  pkl = pickle.load(open(file, 'rb'))
  vectorizer = layer_type.from_config(pkl['config'])
  vectorizer.set_weights(pkl['weights'])
  return vectorizer
