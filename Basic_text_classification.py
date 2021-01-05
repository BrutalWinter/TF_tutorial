#   This tutorial demonstrates text classification starting from plain text files stored on disk.You'll train a binary
# classifier to perform sentiment analysis on an IMDB dataset.
import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

#   This notebook trains a sentiment analysis model to classify movie reviews as positive or negative, based on the text of the review.
# This is an example of binary—or two-class—classification, an important and widely applicable kind of machine learning problem.
# You'll use the Large Movie Review Dataset that contains the text of 50,000 movie reviews from the Internet Movie Database.
# These are split into 25,000 reviews for training and 25,000 reviews for testing. The training and testing sets are balanced,
# meaning they contain an equal number of positive and negative reviews.

url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
dataset_dir0 = tf.keras.utils.get_file("aclImdb_v1.tar.gz", url, untar=True, cache_dir='.', cache_subdir='')
print(dataset_dir0)
# dataset_dir0=r'./aclImdb_v1.tar.gz'
dataset_dir = os.path.join(os.path.dirname(dataset_dir0), 'aclImdb')
print(os.listdir(dataset_dir))
print(dataset_dir)

train_dir = os.path.join(dataset_dir, 'train')
print(os.listdir(train_dir))
print(train_dir)

# The aclImdb/train/pos and aclImdb/train/neg directories contain many text files, each of which is a single movie review. Let's take a look at one of them:
sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
with open(sample_file) as f:
  print(f.read())

# Next, you will load the data off disk and prepare it into a format suitable for training. To do so, you will use the helpful
# text_dataset_from_directory utility, which expects a directory structure as follows.
# main_directory/class_a/a_text_1.txt, main_directory/class_a/a_text_2.txt
# main_directory/class_b/b_text_1.txt, main_directory/class_b/b_text_2.txt

# To prepare a dataset for binary classification, you will need two folders on disk, corresponding to class_a and class_b.
# These will be the positive and negative movie reviews, which can be found in aclImdb/train/pos and aclImdb/train/neg.
# As the IMDB dataset contains additional folders, you will remove them before using this utility.
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)
# When running a machine learning experiment, it is a best practice to divide your dataset into three splits: train, validation, and test.
# The IMDB dataset has already been divided into train and test, but it lacks a validation set.
# Let's create a validation set using an 80:20 split of the training data by using the validation_split argument below.
batch_size = 32
seed = 42

raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory('aclImdb/train', batch_size=batch_size, validation_split=0.2,subset='training', seed=seed)

# As you can see above, there are 25,000 examples in the training folder, of which you will use 80% (or 20,000) for training.
# As you will see in a moment, you can train a model by passing a dataset directly to model.fit.
# If you're new to tf.data, you can also iterate over the dataset and print out a few examples as follows:
for text_batch, label_batch in raw_train_ds.take(1):
  for i in range(5):
    print("Review", text_batch.numpy()[i])
    print("Label", label_batch.numpy()[i])

# Notice the reviews contain raw text (with punctuation and occasional HTML tags like <br/>). You will show how to handle these in the following section.
# The labels are 0 or 1. To see which of these correspond to positive and negative movie reviews, you can check the class_names property on the dataset.
print("Label 0 corresponds to", raw_train_ds.class_names[0])
print("Label 1 corresponds to", raw_train_ds.class_names[1])

# Next, you will create a validation and test dataset. You will use the remaining 5,000 reviews from the training set for validation.
raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory('aclImdb/train', batch_size=batch_size, validation_split=0.2, subset='validation', seed=seed)
raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory('aclImdb/test', batch_size=batch_size)

##############################    Prepare the dataset for training  ##############################
# Next, you will standardize, tokenize, and vectorize the data using the helpful preprocessing.TextVectorization layer.
# Standardization refers to preprocessing the text, typically to remove punctuation or HTML elements to simplify the dataset.
# Tokenization refers to splitting strings into tokens (for example, splitting a sentence into individual words, by splitting on whitespace).
# Vectorization refers to converting tokens into numbers so they can be fed into a neural network. All of these tasks can be accomplished with this layer.

# As you saw above, the reviews contain various HTML tags like <br />. These tags will not be removed by the default standardizer
# in the TextVectorization layer (which converts text to lowecase and strips punctuation by default, but doesn't strip HTML).
# You will write a custom standardization function to remove the HTML.

def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,'[%s]' % re.escape(string.punctuation), '')

# Next, you will create a TextVectorization layer. you will use this layer to standardize, tokenize, and vectorize our data.
# You set the output_mode to int to create unique integer indices for each token.
# Note that you're using the default split function, and the custom standardization function you defined above.
# You'll also define some constants for the model, like an explicit maximum sequence_length, which will cause the layer to pad or truncate sequences to exactly sequence_length values.
max_features = 10000
sequence_length = 250

vectorize_layer = TextVectorization(standardize=custom_standardization,max_tokens=max_features,output_mode='int',output_sequence_length=sequence_length)
# Next, you will call adapt to fit the state of the preprocessing layer to the dataset. This will cause the model to build an index of strings to integers.
# Note: it's important to only use your training data when calling adapt (using the test set would leak information)
# Make a text-only dataset (without labels), then call adapt
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

# Let's create a function to see the result of using this layer to preprocess some data.
def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label

# retrieve a batch (of 32 reviews and labels) from the dataset
text_batch, label_batch = next(iter(raw_train_ds))
first_review, first_label = text_batch[0], label_batch[0]
print("Review", first_review)
print("Label", raw_train_ds.class_names[first_label])
print("Vectorized review", vectorize_text(first_review, first_label))

# As you can see above, each token has been replaced by an integer. You can lookup the token (string) that each integer corresponds to by calling
# .get_vocabulary() on the layer.
print("1287 ---> ",vectorize_layer.get_vocabulary()[1287])
print(" 313 ---> ",vectorize_layer.get_vocabulary()[313])
print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))
# You are nearly ready to train your model. As a final preprocessing step, you will apply the TextVectorization layer you created earlier
# to the train, validation, and test dataset.
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)
##############################    Configure the dataset for performance  ##############################
# .cache(): keeps data in memory after it's loaded off disk.
# This will ensure the dataset does not become a bottleneck while training your model.
# If your dataset is too large to fit into memory, you can also use this method to create a performant on-disk cache,
# which is more efficient to read than many small files.

# .prefetch() overlaps data preprocessing and model execution while training.
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


###########    Create the model
# The layers are stacked sequentially to build the classifier:
#   1. The first layer is an Embedding layer. This layer takes the integer-encoded reviews and looks up an embedding vector for each word-index.
# These vectors are learned as the model trains. The vectors add a dimension to the output array. The resulting dimensions are: (batch, sequence, embedding).
#   2. a GlobalAveragePooling1D layer returns a fixed-length output vector for each example by averaging over the sequence dimension.
# This allows the model to handle input of variable length, in the simplest way possible.
#   3. This fixed-length output vector is piped through a fully-connected (Dense) layer with 16 hidden units. The last layer is densely connected with a single output node.
embedding_dim = 16
model = tf.keras.Sequential([
  layers.Embedding(max_features + 1, embedding_dim),
  layers.Dropout(0.2),
  layers.GlobalAveragePooling1D(),
  layers.Dropout(0.2),
  layers.Dense(1)])

model.summary()

###########    Loss function and optimizer
model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs)

# Let's see how the model performs. Two values will be returned. Loss (a number which represents our error, lower values are better), and accuracy.
loss, accuracy = model.evaluate(test_ds)
print("Loss: ", loss)
print("Accuracy: ", accuracy)

# model.fit() returns a History object that contains a dictionary with everything that happened during training:
history_dict = history.history
print(history_dict.keys())

acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = tf.range(1, len(acc) + 1)
#########################
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

###################### Export the model
# In the code above, you applied the TextVectorization layer to the dataset before feeding text to the model.
# If you want to make your model capable of processing raw strings (for example, to simplify deploying it),
# you can include the TextVectorization layer inside your model. To do so, you can create a new model using the weights you just trained.
export_model = tf.keras.Sequential([
  vectorize_layer,
  model,
  layers.Activation('sigmoid')
])

export_model.compile(
    loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
)

# Test it with `raw_test_ds`, which yields raw strings
loss, accuracy = export_model.evaluate(raw_test_ds)
print(accuracy)

# To get predictions for new examples, you can simply call model.predict().
examples = [
  "The movie was great!",
  "The movie was okay.",
  "The movie was terrible..."
]

export_model.predict(examples)
#Including the text preprocessing logic inside your model enables you to export a model for production that simplifies deployment,
# and reduces the potential for train/test skew.

# There is a performance difference to keep in mind when choosing where to apply your TextVectorization layer.
# Using it outside of your model enables you to do asynchronous CPU processing and buffering of your data when training on GPU.
# So, if you're training your model on the GPU, you probably want to go with this option to get the best performance while developing your model,
# then switch to including the TextVectorization layer inside your model when you're ready to prepare for deployment.





















