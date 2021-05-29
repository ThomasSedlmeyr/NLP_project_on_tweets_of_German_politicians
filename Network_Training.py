import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf
import pandas as pd
import numpy as np

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

tf.config.threading.set_intra_op_parallelism_threads(8) # Model uses 10 CPUs while training. + GPU

data = np.load('TweetAndParty.npy', allow_pickle=True)
print(data.shape)
texts = list(data[0])
labels = list(data[1])

# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

train_per = 80
val_per = 15

batch_size = 32
seed = 42
sequence_length = 250
epochs = 40


print(type(texts))
print(type(labels))
trainIndex = int((1.0 *train_per/100)*len(texts))
valIndex = int((1.0 *val_per/100)*len(texts)+trainIndex)

dataset = tf.data.Dataset.from_tensor_slices((texts, labels))
dataset = dataset.shuffle(len(texts), seed = seed)

raw_train_ds = dataset.take(trainIndex)
raw_val_ds = dataset.take(valIndex)
raw_val_ds = raw_val_ds.skip(trainIndex)
raw_test_ds = dataset.skip(valIndex)


batched_train_ds = raw_train_ds.batch(batch_size)
batched_val_ds = raw_val_ds.batch(batch_size)
batched_test_ds = raw_test_ds.batch(batch_size)


AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = batched_train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = batched_val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = batched_test_ds.cache().prefetch(buffer_size=AUTOTUNE)

trains_ds = train_ds.shuffle(len(list(train_ds)), seed = seed)
#for text_batch, label_batch in batched_train_ds.take(1):
  #for i in range(0, 4):
    #print("Review", text_batch.numpy()[i])
    #print("Label", label_batch.numpy()[i])

def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation),
                                  '')

max_features = 10000

vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

# Make a text-only dataset (without labels), then call adapt
train_text = trains_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label

# retrieve a batch (of 32 reviews and labels) from the dataset
text_batch, label_batch = next(iter(trains_ds))
first_review, first_label = text_batch[0], label_batch[0]
print("Review", first_review)
print("Vectorized review", vectorize_text(first_review, first_label))

print("1287 ---> ",vectorize_layer.get_vocabulary()[1287])
print(" 313 ---> ",vectorize_layer.get_vocabulary()[313])
print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))

train_ds = trains_ds.map(vectorize_text)
val_ds = val_ds.map(vectorize_text)
test_ds = test_ds.map(vectorize_text)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True)


embedding_dim = 16

model = tf.keras.Sequential([
  layers.Embedding(max_features + 1, embedding_dim),
  layers.Dropout(0.2),
  layers.GlobalAveragePooling1D(),
  layers.Dropout(0.2),
  layers.Dense(7, activation='sigmoid')])

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=tf.metrics.AUC())

epochs = 10

def train():
  history = model.fit(
      train_ds,
      validation_data=val_ds,
      callbacks=[cp_callback],
      epochs=epochs)

  loss, accuracy = model.evaluate(test_ds)

  print("Loss: ", loss)
  print("Accuracy: ", accuracy)

  history_dict = history.history
  history_dict.keys()


  acc = history_dict['auc']
  val_acc = history_dict['val_auc']
  loss = history_dict['loss']
  val_loss = history_dict['val_loss']

  epochs = range(1, len(acc) + 1)

  # "bo" is for "blue dot"
  plt.plot(epochs, loss, 'bo', label='Training loss')
  # b is for "solid blue line"
  plt.plot(epochs, val_loss, 'b', label='Validation loss')
  plt.title('Training and validation loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()

  plt.show()


  plt.plot(epochs, acc, 'bo', label='Training acc')
  plt.plot(epochs, val_acc, 'b', label='Validation acc')
  plt.title('Training and validation accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend(loc='lower right')

  plt.show()


  # Test it with `raw_test_ds`, which yields raw strings
  #loss, accuracy = export_model.evaluate(test_ds)
  #print(accuracy)


def evaluate():
  
  parties = ['CDU', 'LINKE', 'GRÜNE', 'SPD', 'CSU', 'AFD']

  result = model.predict(text)
  for i in range(0, len(texts)):
      print(parties)           
      print(raw_test_ds.get)


evaluate()