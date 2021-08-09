import os
import shutil
import glob
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain


#tf.keras.backend.set_floatx('float64')
tf.config.threading.set_intra_op_parallelism_threads(4) # Model uses 10 CPUs while training. + GPU
tf.get_logger().setLevel('ERROR')

# path to csv-file
csv_file = 'Big5Essay/essays.csv'
csv_test = 'Big5Essay/essays_test.csv'
# some csv need encoding
encoding = "ISO-8859-1"
# text column
t_col = "TEXT"
# responses
r_col = ['cEXT','cNEU','cAGR','cCON','cOPN']

# set percentage for train, val and rest for test
train_per = 80
val_per = 15

# model path
#saved_model_path = 'Big5Essay/Model/l8_h512_a8/'
# Include the epoch in the file name (uses `str.format`)
#checkpoint_path = "Big5Essay/Model/l8_h512_a8/Checkpoints/cp-{epoch:01d}.ckpt"
#checkpoint_dir = os.path.dirname(checkpoint_path)

# retrain existing model (only if there is one saved)
retrain = False

#  model parameters
AUTOTUNE = tf.data.AUTOTUNE
#batch_size = 32
batch_size = 8
seed = 42

epochs = 30


# all_files = glob.glob(csv_file + "/*.csv")
#
# li = []
#
# for filename in all_files:
#     df = pd.read_csv(filename, index_col=None, header=0, encoding=encoding, lineterminator='\n')
#     li.append(df)
#
# dataframe = pd.concat(li, axis=0, ignore_index=True)

dataframe = pd.read_csv(csv_file, encoding=encoding)
print(dataframe.columns.tolist())
#dataframe = dataframe.head(1000)
print(dataframe.head())

texts = dataframe.pop(t_col)
texts = tf.data.Dataset.from_tensor_slices(texts)
output = dataframe.drop(columns=[col for col in dataframe if col not in r_col])
# transform y n to 1 0
output = np.where(output=='y', 1, 0)
output = tf.data.Dataset.from_tensor_slices(output)
dataset = tf.data.Dataset.zip((texts, output)) # datasets have to be one (!) tupple.

for line in dataset.take(5):
  print(f'{line[0].numpy()}') # Notice indexing bc zip!
  print(line[1])

lenData = len(list(dataset))

dataset = dataset.shuffle(lenData, seed = seed)
train_split = lenData//100*train_per
val_split = lenData//100*(train_per + val_per)

raw_train_ds = dataset.take(train_split)
raw_train_ds_batched = raw_train_ds.batch(batch_size=batch_size)
train_ds = raw_train_ds_batched.cache().prefetch(buffer_size=AUTOTUNE)

raw_val_ds = dataset.take(val_split)
raw_val_ds = raw_val_ds.skip(train_split)
raw_val_ds_batched = raw_val_ds.batch(batch_size=batch_size)
val_ds = raw_val_ds_batched.cache().prefetch(buffer_size=AUTOTUNE)

raw_test_ds = dataset.skip(val_split)
raw_test_ds_batched = raw_test_ds.batch(batch_size=batch_size)
test_ds = raw_test_ds_batched.cache().prefetch(buffer_size=AUTOTUNE)

# save test data as csv for later evaluation
# probably not the most pythonic way.
test_list = list(raw_test_ds.as_numpy_iterator())
test_out = [x[1] for x in test_list]
test_df = pd.DataFrame(test_out, columns=r_col)
test_df[t_col] = [t[0] for t in test_list]
test_df.to_csv(csv_test, index=False)


print("Length of datasets: {}".format(len(list(dataset))))
print("Length of train: {}".format(len(list(raw_train_ds))))
print("Length of val: {}".format(len(list(raw_val_ds))))
print("Length of test: {}".format(len(list(raw_test_ds))))


#bert_model_name = 'small_bert/bert_en_uncased_L-4_H-512_A-8'
#bert_model_name = 'small_bert/bert_en_uncased_L-8_H-512_A-8'
map_name_to_handle = {
    'bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',
    'bert_en_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3',
    'bert_multi_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3',
    'small_bert/bert_en_uncased_L-2_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-2_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-2_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-2_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-4_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-4_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-4_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-4_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-6_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-6_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-6_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-6_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-8_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-8_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-8_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-8_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-10_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-10_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-10_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-10_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-12_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-12_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-12_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/1',
    'albert_en_base':
        'https://tfhub.dev/tensorflow/albert_en_base/2',
    'electra_small':
        'https://tfhub.dev/google/electra_small/2',
    'electra_base':
        'https://tfhub.dev/google/electra_base/2',
    'experts_pubmed':
        'https://tfhub.dev/google/experts/bert/pubmed/2',
    'experts_wiki_books':
        'https://tfhub.dev/google/experts/bert/wiki_books/2',
    'talking-heads_base':
        'https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/1',
}

map_model_to_preprocess = {
    'bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'bert_en_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'bert_multi_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3',
    'albert_en_base':
        'https://tfhub.dev/tensorflow/albert_en_preprocess/3',
    'electra_small':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'electra_base':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'experts_pubmed':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'experts_wiki_books':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'talking-heads_base':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
}

#method for training different bert models and saving the result
def train_different_variations(epochs):
    models_to_train = ['small_bert/bert_en_uncased_L-8_H-256_A-4', 'small_bert/bert_en_uncased_L-8_H-512_A-8', 'small_bert/bert_en_uncased_L-4_H-512_A-8']
    for model_name in models_to_train:
        path_for_saving = 'Big5Essay/Model/' + model_name[11:] + '/'
        print(path_for_saving)
        checkpoint_path = path_for_saving + "Checkpoints/cp-{epoch:01d}.ckpt"
        print(checkpoint_path)
        model = build_model(model_name)
        model.save_weights(checkpoint_path.format(epoch=0))
        train(model, epochs, path_for_saving, checkpoint_path)

# initializer = tf.keras.initializers.LecunNormal()
def build_model(bert_model_name):

    tfhub_handle_encoder = map_name_to_handle[bert_model_name]
    tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name]

    print(f'BERT model selected           : {tfhub_handle_encoder}')
    print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')

    bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)

    text_test = ['this is such an amazing movie!']
    text_preprocessed = bert_preprocess_model(text_test)

    print(f'Keys       : {list(text_preprocessed.keys())}')
    print(f'Shape      : {text_preprocessed["input_word_ids"].shape}')
    print(f'Word Ids   : {text_preprocessed["input_word_ids"][0, :12]}')
    print(f'Input Mask : {text_preprocessed["input_mask"][0, :12]}')
    print(f'Type Ids   : {text_preprocessed["input_type_ids"][0, :12]}')

    bert_model = hub.KerasLayer(tfhub_handle_encoder)

    bert_results = bert_model(text_preprocessed)

    print(f'Loaded BERT: {tfhub_handle_encoder}')
    print(f'Pooled Outputs Shape:{bert_results["pooled_output"].shape}')
    print(f'Pooled Outputs Values:{bert_results["pooled_output"][0, :12]}')
    print(f'Sequence Outputs Shape:{bert_results["sequence_output"].shape}')
    print(f'Sequence Outputs Values:{bert_results["sequence_output"][0, :12]}')  


    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    big5 = tf.keras.layers.Dense(5, activation='sigmoid', name='big5')(net)
    model = tf.keras.Model(text_input, big5)
    model = compileModel(model)
    return model

def compileModel(model):
    metrics = tf.metrics.BinaryAccuracy()
    steps_per_epoch = len(list(raw_train_ds))//epochs
    num_train_steps = steps_per_epoch * epochs
    steps_per_epoch = len(list(raw_train_ds))//epochs
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1*num_train_steps)
    init_lr = 3e-5
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                            num_train_steps=num_train_steps,
                                            num_warmup_steps=num_warmup_steps,
                                            optimizer_type='adamw')

    model.compile(optimizer=optimizer,
                            loss='binary_crossentropy',
                            metrics=metrics)
    return model


def testModel(modelPath):
    model = build_model() 
    model.load_weights(modelPath)
    # Save the weights using the `checkpoint_path` format
    model.save_weights(checkpoint_path.format(epoch=0))

def train(model, epochs, path_for_saving, checkpoint_path):

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        mode='auto',
        save_weights_only=True,
        save_freq='epoch',
        save_best_only=True,
        monitor = 'val_loss')

    earlystop_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=5)
    history = model.fit(x=train_ds,
                        validation_data=val_ds,
                        callbacks=[cp_callback, earlystop_callback],
                        epochs=epochs)

    loss, accuracy = model.evaluate(test_ds)

    model.save(path_for_saving)
    f = open(path_for_saving + 'acc.txt', "a+")
    f.write(str(accuracy))
    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')
    #testModel(path_for_saving)

def plot_result(history):
    history_dict = history.history
    print(history_dict.keys())

    acc = history_dict['auc']
    val_acc = history_dict['val_auc']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)
    fig = plt.figure(figsize=(10, 6))
    fig.tight_layout()

    plt.subplot(2, 1, 1)
    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'r', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    # plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()


    plt.subplot(2, 1, 2)
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.savefig("Plot1.png")


def loadModel(modelPath):
    reloaded_model = tf.saved_model.load(modelPath)
    return reloaded_model

def print_my_examples(inputs, results):
  result_for_printing = \
    [f'input: {inputs[i]:<30} : EXP: {results[i][0]:.6f} : NEU: {results[i][1]:.6f} : ARG: {results[i][2]:.6f} : CON: {results[i][3]:.6f} : OPN: {results[i][4]:.6f}'
                         for i in range(len(inputs))]
  print(*result_for_printing, sep='\n')
  print()

examples = [
    'this is such an amazing movie!',  # this is the same sentence tried earlier
    'The movie was great!',
    'The movie was meh.',
    'The movie was okish.',
    'The movie was terrible...',
    'I went to a party and talked to everybody the whole night.',
    'I always feel uncomfortable around other people.'
]

def testModel(model_path):
    #checkpoint_dir = 'Big5Essay/Model/bert_en_uncased_L-8_H-512_A-8/Checkpoints'
    #latest = tf.train.latest_checkpoint(checkpoint_dir)
    #model = build_model('small_bert/bert_en_uncased_L-8_H-512_A-8')
    #model = compileModel(model)
    #model.load_weights(latest)
    #model = tf.saved_model.load(model_path)
    
    model = tf.keras.models.load_model(model_path, compile=False)
    model = compileModel(model)
    #model.summary()
    #results = model(tf.constant(examples))
    #print_my_examples(examples, results)
    #model.evaluate(test_ds)

    #dataframe = pd.read_csv('Big5Essay/Data_Generation/essays_german_to_english.csv', encoding=encoding, error_bad_lines=False)
    #dataframe = pd.read_csv('Big5Essay/Data_Generation/test.csv', encoding=encoding)
    dataframe = pd.read_csv('Big5Essay/Data_Generation/essays.csv', encoding=encoding)
    #dataframe = dataframe.head(1000)
    print(len(list(dataframe)))
    print(dataframe.columns.tolist())
    texts = dataframe.pop(t_col)
    texts = tf.strings.regex_replace(texts, '"', "")
    print(texts)
    texts = tf.data.Dataset.from_tensor_slices(texts)
    output = dataframe.drop(columns=[col for col in dataframe if col not in r_col])
    # transform y n to 1 0
    output = np.where(output=='y', 1, 0)
    output = tf.data.Dataset.from_tensor_slices(output)
    dataset = tf.data.Dataset.zip((texts, output))
    raw_test_ds_batched = dataset.batch(batch_size=batch_size)
    test_ds2 = raw_test_ds_batched.cache().prefetch(buffer_size=AUTOTUNE)
    loss, accuracy = model.evaluate(test_ds2) 
    print("Accuracy " + str(accuracy))
    #lenData = len(list(dataset))


#train_different_variations(epochs)
testModel('Big5Essay/Model/bert_en_uncased_L-8_H-512_A-8')