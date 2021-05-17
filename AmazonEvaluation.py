import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf
import pandas as pd

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

#Hier steht der Text drin
texts = []
#Hier steht die Bewertung drin
values = []

with open('test.ft.txt') as f:
    lines = f.readlines()

for line in lines:
  label = line[:10]
  text = line[11:]
  if label == "__label__2":
    values.append(1)
  else: 
    values.append(0)
  texts.append(text)


print("Hello")
print(tf.__version__)
var = (["test", "test2"],[1,2])

batch_size = 32
seed = 42
sequence_length = 250

raw_train_ds = tf.data.Dataset.from_tensor_slices((texts[2500:], values[2500:]))#Macht keine Batches
raw_val_ds = tf.data.Dataset.from_tensor_slices((texts[2000:2500], values[2000:2500]))
raw_test_ds = tf.data.Dataset.from_tensor_slices((texts[:2000], values[:2000])) 

batched_train_ds = raw_train_ds.batch(batch_size) #Hier werden Batches gemacht
batched_val_ds = raw_val_ds.batch(batch_size) #Hier werden Batches gemacht
batched_test_ds = raw_test_ds.batch(batch_size) #Hier werden Batches gemacht

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
train_text = batched_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label

# retrieve a batch (of 32 reviews and labels) from the dataset
text_batch, label_batch = next(iter(batched_train_ds))
first_review, first_label = text_batch[0], label_batch[0]
print("Review", first_review)
print("Vectorized review", vectorize_text(first_review, first_label))

print("1287 ---> ",vectorize_layer.get_vocabulary()[1287])
print(" 313 ---> ",vectorize_layer.get_vocabulary()[313])
print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))

train_ds = batched_train_ds.map(vectorize_text)
val_ds = batched_val_ds.map(vectorize_text)
test_ds = batched_test_ds.map(vectorize_text)

embedding_dim = 16

model = tf.keras.Sequential([
  layers.Embedding(max_features + 1, embedding_dim),
  layers.Dropout(0.2),
  layers.GlobalAveragePooling1D(),
  layers.Dropout(0.2),
  layers.Dense(1)])

model.summary()

model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

epochs = 2
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs)

loss, accuracy = model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)

history_dict = history.history
history_dict.keys()


acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
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

export_model = tf.keras.Sequential([
  vectorize_layer,
  model,
  layers.Activation('sigmoid')
])

export_model.compile(
    loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
)

# Test it with `raw_test_ds`, which yields raw strings
loss, accuracy = export_model.evaluate(batched_test_ds)
print(accuracy)
