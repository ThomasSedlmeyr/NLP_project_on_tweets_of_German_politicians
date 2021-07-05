import collections
import os
import pathlib
import re
import string
import sys
import tempfile
import time
import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow_text as text
import tensorflow as tf

tf.get_logger().setLevel('ERROR')
pwd = pathlib.Path.cwd()

def createEnglishGermanSentenceDataSet():
    dataSet = []
    germanSentence = ""
    englishSentence = ""
    isGerman = False
    counter = 1
    with open("Dataset_TED_English_German.txt", encoding="utf8") as file:
        for line in file:
            #An empty line indicates a new pair of English and German sentences 
            if(line == "\n"):
                dataSet.append([englishSentence, germanSentence])
            elif(isGerman):
                germanSentence = line
                isGerman = False
            else: 
                englishSentence = line
                isGerman = True

    numpyArray = np.array(dataSet)
    #np.save("Dataset_TED_English_German", numpyArray)
    print("DataSet was created!")
    trainIndex = int(0.85 * len(numpyArray))
    valIndex = int(0.9 * len(numpyArray))
    transposed = np.transpose(numpyArray)
    raw_train_ds = tf.data.Dataset.from_tensor_slices(((transposed[0][:trainIndex]), (transposed[1][:trainIndex])))
    raw_val_ds = tf.data.Dataset.from_tensor_slices((((transposed[0][trainIndex:valIndex]), (transposed[1][trainIndex:valIndex]))))
    raw_test_ds = tf.data.Dataset.from_tensor_slices(((transposed[0][valIndex:]), ((transposed[1][valIndex:]))))
    return (raw_train_ds, raw_val_ds)

#examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                               ##as_supervised=True)
(train_examples, val_examples) = createEnglishGermanSentenceDataSet()


for en, de in train_examples.take(1):
  print("English: ", en.numpy().decode('utf-8'))
  print("German:   ", de.numpy().decode('utf-8'))

train_de = train_examples.map(lambda en, de: de)
train_en = train_examples.map(lambda en, de: en)

from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

bert_tokenizer_params=dict(lower_case=True)
reserved_tokens=["[PAD]", "[UNK]", "[START]", "[END]"]

bert_vocab_args = dict(
    # The target vocabulary size
    vocab_size = 8000,
    # Reserved tokens that must be included in the vocabulary
    reserved_tokens=reserved_tokens,
    # Arguments for `text.BertTokenizer`
    bert_tokenizer_params=bert_tokenizer_params,
    # Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
    learn_params={},
)



en_vocab = bert_vocab.bert_vocab_from_dataset(
    train_en.batch(1000).prefetch(2),
    **bert_vocab_args
)

print(en_vocab[:10])
print(en_vocab[100:110])
print(en_vocab[1000:1010])
print(en_vocab[-10:])


def write_vocab_file(filepath, vocab):
  with open(filepath, 'w', encoding="utf8") as f:
    for token in vocab:
      print(token, file=f)

write_vocab_file('en_vocab.txt', en_vocab)


de_vocab = bert_vocab.bert_vocab_from_dataset(
    train_de.batch(1000).prefetch(2),
    **bert_vocab_args
)

print(de_vocab[:10])
print(de_vocab[100:110])
print(de_vocab[1000:1010])
print(de_vocab[-10:])

write_vocab_file('de_vocab.txt', de_vocab)


en_tokenizer = text.BertTokenizer('en_vocab.txt', **bert_tokenizer_params)
de_tokenizer = text.BertTokenizer('de_vocab.txt', **bert_tokenizer_params)

for en_examples, de_examples in train_examples.batch(3).take(1):
  for ex in de_examples:
    print(ex.numpy())

# Tokenize the examples -> (batch, word, word-piece)
token_batch = de_tokenizer.tokenize(de_examples)
# Merge the word and word-piece axes -> (batch, tokens)
token_batch = token_batch.merge_dims(-2,-1)

for ex in token_batch.to_list():
  print(ex)

# Lookup each token id in the vocabulary.
txt_tokens = tf.gather(de_vocab, token_batch)
# Join with spaces.
tf.strings.reduce_join(txt_tokens, separator=' ', axis=-1)

words = de_tokenizer.detokenize(token_batch)
tf.strings.reduce_join(words, separator=' ', axis=-1)

START = tf.argmax(tf.constant(reserved_tokens) == "[START]")
END = tf.argmax(tf.constant(reserved_tokens) == "[END]")

def add_start_end(ragged):
  count = ragged.bounding_shape()[0]
  starts = tf.fill([count,1], START)
  ends = tf.fill([count,1], END)
  return tf.concat([starts, ragged, ends], axis=1)

words = de_tokenizer.detokenize(add_start_end(token_batch))
tf.strings.reduce_join(words, separator=' ', axis=-1)

def cleanup_text(reserved_tokens, token_txt):
  # Drop the reserved tokens, exceen for "[UNK]".
  bad_tokens = [re.escape(tok) for tok in reserved_tokens if tok != "[UNK]"]
  bad_token_re = "|".join(bad_tokens)

  bad_cells = tf.strings.regex_full_match(token_txt, bad_token_re)
  result = tf.ragged.boolean_mask(token_txt, ~bad_cells)

  # Join them into strings.
  result = tf.strings.reduce_join(result, separator=' ', axis=-1)

  return result

de_examples.numpy()


token_batch = de_tokenizer.tokenize(en_examples).merge_dims(-2,-1)
words = de_tokenizer.detokenize(token_batch)
words

cleanup_text(reserved_tokens, words).numpy()


class CustomTokenizer(tf.Module):
  def __init__(self, reserved_tokens, vocab_path):
    self.tokenizer = text.BertTokenizer(vocab_path, lower_case=True)
    self._reserved_tokens = reserved_tokens
    self._vocab_path = tf.saved_model.Asset(vocab_path)

    vocab = pathlib.Path(vocab_path).read_text(encoding="utf8").splitlines()
    self.vocab = tf.Variable(vocab)

    ## Create the signatures for export:   

    # Include a tokenize signature for a batch of strings. 
    self.tokenize.get_concrete_function(
        tf.TensorSpec(shape=[None], dtype=tf.string))

    # Include `detokenize` and `lookup` signatures for:
    #   * `Tensors` with shapes [tokens] and [batch, tokens]
    #   * `RaggedTensors` with shape [batch, tokens]
    self.detokenize.get_concrete_function(
        tf.TensorSpec(shape=[None, None], dtype=tf.int64))
    self.detokenize.get_concrete_function(
          tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

    self.lookup.get_concrete_function(
        tf.TensorSpec(shape=[None, None], dtype=tf.int64))
    self.lookup.get_concrete_function(
          tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

    # These `get_*` methods take no arguments
    self.get_vocab_size.get_concrete_function()
    self.get_vocab_path.get_concrete_function()
    self.get_reserved_tokens.get_concrete_function()

  @tf.function
  def tokenize(self, strings):
    enc = self.tokenizer.tokenize(strings)
    # Merge the `word` and `word-piece` axes.
    enc = enc.merge_dims(-2,-1)
    enc = add_start_end(enc)
    return enc

  @tf.function
  def detokenize(self, tokenized):
    words = self.tokenizer.detokenize(tokenized)
    return cleanup_text(self._reserved_tokens, words)

  @tf.function
  def lookup(self, token_ids):
    return tf.gather(self.vocab, token_ids)

  @tf.function
  def get_vocab_size(self):
    return tf.shape(self.vocab)[0]

  @tf.function
  def get_vocab_path(self):
    return self._vocab_path

  @tf.function
  def get_reserved_tokens(self):
    return tf.constant(self._reserved_tokens)


tokenizers = tf.Module()
tokenizers.en = CustomTokenizer(reserved_tokens, 'en_vocab.txt')
tokenizers.de = CustomTokenizer(reserved_tokens, 'de_vocab.txt')

model_name = 'testModel'
tf.saved_model.save(tokenizers, model_name)

reloaded_tokenizers = tf.saved_model.load(model_name)
reloaded_tokenizers.en.get_vocab_size().numpy()

tokens = reloaded_tokenizers.de.tokenize(['Hello TensorFlow!'])
tokens.numpy()

text_tokens = reloaded_tokenizers.de.lookup(tokens)
text_tokens

round_trip = reloaded_tokenizers.de.detokenize(tokens)

print(round_trip.numpy()[0].decode('utf-8'))





