from copyreg import pickle
import tensorflow as tf
from tensorflow.keras import layers
import gradio as gr
import pickle
import stanza
import re


language_codes = {
  'arabic': 'ar', 'chinese': 'zh', 'dutch': 'nl', 'english': 'en',
  'french': 'fr', 'german': 'de', 'greek': 'el', 'hindi': 'hi',
  'italian': 'it', 'japanese': 'ja', 'korean': 'ko', 'portuguese': 'pt',
  'romanian': 'ro', 'russian': 'ru', 'spanish': 'es', 'ukrainian': 'uk'
}


class Translator():
  def __init__(self, input_language, target_language, max_tokens=128):
    self.input_language = input_language
    self.target_language = target_language
    self.models = self.load_models()
    self.max_tokens = max_tokens


  def change_input_language(self, input_language):
    self.input_language = input_language.lower()
    self.models = self.load_models()

  
  def change_target_language(self, target_language):
    self.target_language = target_language.lower()
    self.models = self.load_models()

  
  def load_models(self):
    return [self.load_model(model_name) for model_name in self.get_model_names(self.input_language, self.target_language)]


  def load_model(self, model_name):
    target_language = model_name.split('_to_')[1]
    transformer = tf.saved_model.load(f'./saved_models/{model_name}')
    text_processors_dir = f'./text_processors/{model_name}'
    stanza.download(language_codes[self.input_language])
    input_tokenizer = stanza.Pipeline(lang=language_codes[self.input_language], processors='tokenize')
    input_vectorizer = self.load_pickle_as_layer(f'{text_processors_dir}/input_vectorizer.pkl', layers.TextVectorization)
    target_token_from_index = self.load_pickle_as_layer(f'{text_processors_dir}/target_token_from_index.pkl', layers.StringLookup)
    return transformer, input_tokenizer, input_vectorizer, target_token_from_index, target_language


  def get_model_names(self, input_language, target_language):
    if 'english' in (input_language, target_language):
      model_names = [f'{input_language}_to_{target_language}']
    else:
      model_names = [f'{input_language}_to_english', f'english_to_{target_language}']
    return model_names


  def translate(self, sentence):
    translation = sentence
    for i, (transformer, input_tokenizer, input_vectorizer, target_token_from_index, target_language) in enumerate(self.models):
      translation = self.translate_step(translation, transformer, input_tokenizer, input_vectorizer,
                                           target_token_from_index, target_language, i!=0, i==(len(self.models)-1))
    return translation


  def translate_step(self, sentence, transformer, input_tokenizer, input_vectorizer,
                target_token_from_index, target_language, skip_tokenization=False, prettify=True):
    
    if not skip_tokenization:
      sentence = input_tokenizer(sentence)
      sentence = ' '.join([token.text.lower() for token in sentence.iter_tokens()])
  
    vec = input_vectorizer(sentence)[tf.newaxis]

    start_token_index = tf.constant(2, dtype=tf.int64)[tf.newaxis]
    end_token_index = tf.constant(3, dtype=tf.int64)[tf.newaxis]

    token_indices = [start_token_index]

    # iterate until end token is generated or max_tokens reached
    for i in range(self.max_tokens):
      target_indices = tf.transpose(tf.convert_to_tensor(token_indices))

      predictions, _ = transformer([vec, target_indices], training=False)
      predictions = predictions[:, -1:, :]
      predicted_index = tf.argmax(predictions, axis=-1)

      token_indices.append(predicted_index[0])

      if predicted_index == end_token_index:
        break

    # convert indices to tokens
    tokens = [target_token_from_index(index)[0].numpy().decode() for index in token_indices[1:-1]]
    translation = ' '.join(tokens)

    # prettify translation
    if prettify:
      translation = self.prettify_translation(translation, target_language)

    return translation


  def save_layer_as_pickle(self, obj, file):
    pkl = {'config': obj.get_config(), 'weights': obj.get_weights()}
    pickle.dump(pkl, open(file, 'wb'))


  def load_pickle_as_layer(self, file, layer_type):
    pkl = pickle.load(open(file, 'rb'))
    vectorizer = layer_type.from_config(pkl['config'])
    vectorizer.set_weights(pkl['weights'])
    return vectorizer


  def prettify_translation(self, translation, language=''):
    translation = re.sub(r' ([\'!?,.])', r'\1', translation)
    translation = translation.replace(' n\'t', 'n\'t')
    translation = translation.capitalize()

    no_space_languages = ['chinese', 'japanese', 'korean']

    if language in no_space_languages:
      translation = translation.replace(' ', '')
    
    return translation.capitalize()



translator = Translator('english', 'german')

available_languages = [language.capitalize() for language in language_codes.keys()]

with gr.Blocks() as demo:
  gr.Markdown('''
    <center>
      <h1>Transformer Translation</h1>
      <p>Created by Paul Scott</p>
    </center>
  ''')
  
  with gr.Column():
    with gr.Row():
      with gr.Column():
        input_language_dropdown = gr.Dropdown(label='Input Language', choices=available_languages)
        input_textbox = gr.Textbox(label='Input', lines=5, placeholder=f'Type text here...')

        with gr.Row():
          submit_button = gr.Button('Submit')
          clear_button = gr.Button('Clear')

      with gr.Column():
        target_language_dropdown = gr.Dropdown(label='Output Language', choices=available_languages)
        output_textbox = gr.Textbox(label='Output', lines=5, interactive=False)
  
  submit_button.click(fn=translator.translate, inputs=input_textbox, outputs=output_textbox)
  clear_button.click(fn=lambda: '', inputs=[], outputs=input_textbox)

  input_language_dropdown.change(fn=translator.change_input_language, inputs=input_language_dropdown, outputs=[])
  target_language_dropdown.change(fn=translator.change_target_language, inputs=target_language_dropdown, outputs=[])

demo.launch()
