import os
import re
import shutil
import string
import tensorflow as tf
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

tf.config.threading.set_intra_op_parallelism_threads(6) # Model uses 6 CPUs while training. + GPU

import keras_tuner as kt


pathToData = "Party_Classifier/Data_Generation/"
checkpoint_path = "Party_Classifier/Model/Checkpoints/Keras_Tuner/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

batch_size = 32
sequence_length = 250
epochs = 100


trainNp = np.load(pathToData + 'Train.npy', allow_pickle=True)
valNp = np.load(pathToData + 'Val.npy', allow_pickle=True)
testNp = np.load(pathToData + 'Test.npy', allow_pickle=True)


raw_train_ds = tf.data.Dataset.from_tensor_slices((list(trainNp[0]), list(trainNp[1])))
raw_val_ds = tf.data.Dataset.from_tensor_slices((list(valNp[0]), list(valNp[1])))
raw_test_ds = tf.data.Dataset.from_tensor_slices((list(testNp[0]), list(testNp[1])))

batched_train_ds = raw_train_ds.batch(batch_size)
batched_val_ds = raw_val_ds.batch(batch_size)
batched_test_ds = raw_test_ds.batch(batch_size)

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = batched_train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = batched_val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = batched_test_ds.cache().prefetch(buffer_size=AUTOTUNE)

lkrTweets = np.load(pathToData + 'LKR.npy', allow_pickle=True)

def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  lowercase = tf.strings.regex_replace(lowercase,
                                  '[%s]' % re.escape(string.punctuation),
                                  '')
  lowercase = tf.strings.regex_replace(lowercase, '\n', '')
  return lowercase

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
print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))

vocabulary = vectorize_layer.get_vocabulary()

train_ds = train_ds.map(vectorize_text)
val_ds = val_ds.map(vectorize_text)
test_ds = test_ds.map(vectorize_text)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True)

log_dir = "Party_Classifier/Model/Checkpoints/Logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

embedding_dim = 16

def model_builder(hp):
  model = tf.keras.Sequential([
  tf.keras.layers.Embedding(max_features + 1, embedding_dim),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.GlobalAveragePooling1D()])
  hp_units = hp.Int('units', min_value=500, max_value=20000, step=1000)
  model.add(tf.keras.layers.Dense(units=hp_units, activation='relu'))
  model.add(tf.keras.layers.Dropout(0.2))
  model.add(tf.keras.layers.Dense(7, activation='sigmoid'))

  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4,1e-5])

  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss='binary_crossentropy',
                metrics=['accuracy'])

  return model

def buildBestModel():
  tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='Party_Classifier/hyperParams_opt',
                     project_name='PartyClassifier')
  best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
  model = tuner.hypermodel.build(best_hps)
  latest = tf.train.latest_checkpoint(checkpoint_dir)
  model.load_weights(latest)
  return model

def hyperTrain():
  tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='Party_Classifier/hyperParams_opt',
                     project_name='PartyClassifier')

  stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
  tuner.search(train_ds, epochs=50, validation_data=val_ds, callbacks=[stop_early])

  # Get the optimal hyperparameters
  best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

  print(f"""
  The hyperparameter search is complete. The optimal number of units in the first densely-connected
  layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
  is {best_hps.get('learning_rate')}.
  """)

def train():
  model = buildBestModel()
  model.summary()
  history = model.fit(
      train_ds,
      validation_data=val_ds,
      callbacks=[cp_callback, tensorboard_callback],
      epochs=epochs)

  loss, accuracy = model.evaluate(test_ds)
  print("Loss: ", loss)
  print("Accuracy: ", accuracy)


def evaluate():
  model = buildBestModel()
  export_model = tf.keras.Sequential([
    vectorize_layer,
    model
  ])

  export_model.compile(
      loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
  )
  labels = []
  guesses = []
  wrongClassifications = []
  predictions = export_model.predict(testNp[0])
  for i in range(0, len(predictions)):
    guess = np.argmax(predictions[i])
    guesses.append(guess)
    labels.append(np.argmax(testNp[1][i]))
    if (testNp[1][i][guess] == 0):
      wrongClassifications.append(i)

  cm = tf.math.confusion_matrix(labels, guesses, num_classes=7).numpy()
  con_mat_norm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
  classes=parties
  con_mat_df = pd.DataFrame(con_mat_norm,
                     index = classes, 
                     columns = classes)

  figure = plt.figure(figsize=(7, 7))
  sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()
  print(len(wrongClassifications))
  testTransposed = np.transpose(testNp)
  evaluateLKR(export_model)

def evaluateLKR(export_model):
  result = export_model.predict(lkrTweets)
  sum = np.zeros(len(parties))
  for i in range(0, len(result)):
    sum = np.add(sum, result[i])

  print(parties)           
  sum /= 1.0 * len(lkrTweets)
  print("Network: " + str(sum))
  #With our best model we got: 
  #['CDU', 'LINKE', 'FDP', 'GRÜNE','SPD', 'CSU', 'AFD']
  #[0.18223546 0.12500428 0.18392617 0.1105402  0.20657062 0.02684399 0.19835847]
 

parties = ['CDU', 'LINKE', 'FDP', 'GRÜNE','SPD', 'CSU', 'AFD']
#hyperTrain()
#trainBestHyperParameters()
#train()
evaluate()

