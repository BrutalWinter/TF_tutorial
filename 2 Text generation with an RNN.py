import tensorflow as tf
import numpy as np
import os
import time

path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')


# First, look in the text: Read, then decode for py2 compat.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
print('Length of text: {} characters'.format(len(text)))
# Take a look at the first 250 characters in text
print(text[:250])
# The unique characters in the file
vocab = sorted(set(text))
print('{} unique characters'.format(len(vocab)))

# Vectorize the text
# Before training, you need to map strings to a numerical representation. Create two lookup tables: one mapping characters to numbers, and another for numbers to characters.
# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
print('idx2char',idx2char)
text_as_int = np.array([char2idx[c] for c in text])

# Now you have an integer representation for each character. Notice that you mapped the character as indexes from 0 to len(unique).
print('{')
for char,_ in zip(char2idx, range(30)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('}')

# Show how the first 13 characters from the text are mapped to integers
print('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))

########################           The prediction task
# 1. Given a character, or a sequence of characters, what is the most probable next character?
# This is the task you're training the model to perform. The input to the model will be a sequence of characters,
# and you train the model to predict the output—the following character at each time step.

# 2. Since RNNs maintain an internal state that depends on the previously seen elements,  given all the characters computed until this moment, what is the next character?

# 3. Create training examples and targets:
# Next divide the text into example sequences. Each input sequence will contain seq_length characters from the text.
# For each input sequence, the corresponding targets contain the same length of text, except shifted one character to the right.
# So break the text into chunks of seq_length+1. For example, say seq_length is 4 and our text is "Hello". The input sequence would be "Hell", and the target sequence "ello".
# To do this first use the tf.data.Dataset.from_tensor_slices function to convert the text vector into a stream of character indices.

# The maximum length sentence you want for a single input in characters
seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
for i in char_dataset.take(15):
    print(idx2char[i.numpy()])

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
for item in sequences.take(2):
    print(item, len(item))
    print(idx2char[item.numpy()])
    A=repr(''.join(idx2char[item.numpy()]))
    print(A,len(A))

# For each sequence, duplicate and shift it to form the input and target text by using the map method to apply a simple function to each batch:
print('sequences',sequences)
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text
dataset = sequences.map(split_input_target)
print('dataset',dataset)
# Each index of these vectors is processed as a one time step.
# For the input at time step 0, the model receives the index for "F" and tries to predict the index for "i" as the next character.
# At the next timestep, it does the same thing but the RNN considers the previous step context in addition to the current input character.
for input_example, target_example in  dataset.take(2):
    print('Input data: ', repr(''.join(idx2char[input_example.numpy()])),input_example.shape)
    print('Target data:', repr(''.join(idx2char[target_example.numpy()])),target_example.shape)

    for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
        print("Step {:4d}".format(i))
        print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
        print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))


BATCH_SIZE = 64
# Buffer size to shuffle the dataset (TF data is designed to work with possibly infinite sequences, so it doesn't attempt to shuffle the entire sequence in memory.
# Instead, it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
print('dataset',dataset)

# Use tf.keras.Sequential to define the model. For this simple example three layers are used to define our model:
# 1.tf.keras.layers.Embedding: The input layer. A trainable lookup table that will map the numbers of each character to a vector with embedding_dim dimensions;
# 2.tf.keras.layers.GRU: A type of RNN with size units=rnn_units (You can also use an LSTM layer here.)
# 3.tf.keras.layers.Dense: The output layer, with vocab_size outputs.


vocab_size = len(vocab)# Length of the vocabulary in chars
embedding_dim = 256# The embedding dimension
rnn_units = 1024# Number of RNN units

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model
model = build_model(vocab_size=len(vocab),embedding_dim=embedding_dim,rnn_units=rnn_units,batch_size=BATCH_SIZE)
# First check the shape of the output:
for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
# In the above example the sequence length of the input is 100 but the model can be run on inputs of any length:
model.summary()


########################     Try the model







































