import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf
import pandas as pd
import numpy as np
import datetime

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

tf.config.threading.set_intra_op_parallelism_threads(6) # Model uses 6 CPUs while training. + GPU

# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

batch_size = 32
sequence_length = 250
epochs = 5

lkrTweets = np.load('LKR.npy', allow_pickle=True)
trainNp = np.load('Train.npy', allow_pickle=True)
testNp = np.load('Test.npy', allow_pickle=True)
valNp = np.load('Val.npy', allow_pickle=True)
raw_train_ds = tf.data.Dataset.from_tensor_slices((list(trainNp[0]), list(trainNp[1])))
raw_test_ds = tf.data.Dataset.from_tensor_slices((list(testNp[0]), list(testNp[1])))
raw_val_ds = tf.data.Dataset.from_tensor_slices((list(valNp[0]), list(valNp[1])))


batched_train_ds = raw_train_ds.batch(batch_size)
batched_val_ds = raw_val_ds.batch(batch_size)
batched_test_ds = raw_test_ds.batch(batch_size)


AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = batched_train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = batched_val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = batched_test_ds.cache().prefetch(buffer_size=AUTOTUNE)


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
train_text = train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label

# retrieve a batch (of 32 reviews and labels) from the dataset
text_batch, label_batch = next(iter(train_ds))
first_review, first_label = text_batch[0], label_batch[0]
print("Review", first_review)
print("Vectorized review", vectorize_text(first_review, first_label))

print("1287 ---> ",vectorize_layer.get_vocabulary()[1287])
print(" 313 ---> ",vectorize_layer.get_vocabulary()[313])
print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))

train_ds = train_ds.map(vectorize_text)
val_ds = val_ds.map(vectorize_text)
test_ds = test_ds.map(vectorize_text)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

embedding_dim = 16

model = tf.keras.Sequential([
  layers.Embedding(max_features + 1, embedding_dim),
  layers.Dropout(0.2),
  layers.GlobalAveragePooling1D(),
  layers.Dropout(0.2),
  layers.Dense(10000, activation='relu'),
  layers.Dense(7, activation='sigmoid')])

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])



def train():
  history = model.fit(
      train_ds,
      validation_data=val_ds,
      callbacks=[cp_callback, tensorboard_callback],
      epochs=epochs)

  loss, accuracy = model.evaluate(test_ds)

  print("Loss: ", loss)
  print("Accuracy: ", accuracy)

  history_dict = history.history
  history_dict.keys()


  acc = history_dict['accuracy']
  val_acc = history_dict['val_accuracy']
  loss = history_dict['loss']
  val_loss = history_dict['val_loss']

 # epochs = range(1, len(acc) + 1)

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
  latest = tf.train.latest_checkpoint(checkpoint_dir)
  model.load_weights(latest)
  export_model = tf.keras.Sequential([
    vectorize_layer,
    model
    #,
    #layers.Activation('sigmoid')
  ])

  export_model.compile(
      loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
  )


  evaluateLKR(export_model)
  #result = export_model.predict(lkrTweets)
  #for i in range(0, len(result)):
  #    print(parties)           
  #    print("Network: " + str(result[i]))
  #result = export_model.predict(testNp[0])
  #result2 = export_model.evaluate(testNp[0])
  #for i in range(0, len(testNp[0])):
  #    print(parties)           
    #  print("Network: " + str(result[i]))
   # #  print("Expected: " + str(testNp[1][i]))

def evaluateLKR(export_model):
  result = export_model.predict(lkrTweets)
  sum = np.zeros(len(parties)+1)
  for i in range(0, len(result)):
    sum = np.add(sum, result[i])

  print(parties)           
  print("Network: " + str(sum))

parties = ['CDU', 'LINKE', 'FDP', 'GRÜNE','SPD', 'CSU', 'AFD']
train()
evaluate()