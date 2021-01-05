import numpy as np
import tensorflow as tf
import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
############################################################################################################
            ###   tf.data: Build TensorFlow input pipelines              ###
'''
The tf.data API enables you to build complex input pipelines from simple, reusable pieces. For example: 
    1.the pipeline for an image model might aggregate data from files in a distributed file system, apply random perturbations 
to each image, and merge randomly selected images into a batch for training. 
    2.The pipeline for a text model might involve extracting symbols from raw text data, converting them to embedding identifiers with a lookup table, 
and batching together sequences of different lengths. 
    3.The tf.data API makes it possible to handle large amounts of data, read from different data formats, and perform complex transformations.
'''

'''
The tf.data API introduces a tf.data.Dataset abstraction that represents a sequence of elements, in which each element consists of one or more components. 
For example, in an image pipeline, an element might be a single training example, with a pair of tensor components representing the image and its label.

There are two distinct ways to create a dataset:
1.A data source constructs a Dataset from data stored in memory or in one or more files.
2.A data transformation constructs a dataset from one or more tf.data.Dataset objects.
'''
############################################################################################################
# For example, to construct a Dataset from data in memory, you can use tf.data.Dataset.from_tensors() or tf.data.Dataset.from_tensor_slices().
dataset = tf.data.Dataset.from_tensor_slices([8, 3, 0, 8, 2, 1])
print('dataset',dataset)
# The Dataset object is a Python iterable. This makes it possible to consume its elements using a for loop:
for elem in dataset:
    print(elem.numpy())
#Or by explicitly creating a Python iterator using iter and consuming its elements using next:
it = iter(dataset)
print('next(it).numpy()=',next(it).numpy())
# dataset elements can be consumed using the reduce transformation, which reduces all elements to produce a single result.
A=dataset.reduce(0, lambda state, value: state + value)
print('print(A.numpy()=',A.numpy())############################################################################################################
print('A=',A)

############################################################################################################
                                    ###   Dataset structure     ###
'''
# A dataset contains elements that each have the same (nested) structure.
# the individual components of the structure can be of any type representable, tf.TypeSpec:
# including tf.Tensor, tf.sparse.SparseTensor, tf.RaggedTensor, tf.TensorArray, or tf.data.Dataset.
'''
#The Dataset.element_spec property allows you to inspect the type of each element component. T
# he property returns a nested structure of tf.TypeSpec objects, matching the structure of the element,
# which may be a single component, a tuple of components, or a nested tuple of components. For example:
dataset1 = tf.data.Dataset.from_tensor_slices(tf.random.uniform([4, 10]))
print('dataset1.element_spec=',dataset1.element_spec)
print('dataset1.take(1)=',dataset1.take(1))

dataset2 = tf.data.Dataset.from_tensor_slices(
   (tf.random.uniform([4]),
    tf.random.uniform([4, 100], maxval=100, dtype=tf.int32)))
print('dataset2.element_spec',dataset2.element_spec)

dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
print('dataset3.element_spec=',dataset3.element_spec)

# Dataset containing a sparse tensor.
dataset4 = tf.data.Dataset.from_tensors(tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4]))
print('dataset4.element_spec',dataset4.element_spec)
# Use value_type to see the type of value represented by the element spec
print('dataset4.element_spec.value_type=',dataset4.element_spec.value_type)

#The Dataset transformations support datasets of any structure.
# When using the Dataset.map(), and Dataset.filter() transformations, which apply a function to each element,
# the element structure determines the arguments of the function:
dataset1 = tf.data.Dataset.from_tensor_slices(tf.random.uniform([4, 10], minval=1, maxval=10, dtype=tf.int32))
for z in dataset1:
  print(z.numpy())
dataset2 = tf.data.Dataset.from_tensor_slices(
   (tf.random.uniform([4]),
    tf.random.uniform([4, 100], maxval=100, dtype=tf.int32)))
dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
for a, (b,c) in dataset3:
  print('shapes: {a.shape}, {b.shape}, {c.shape}'.format(a=a, b=b, c=c))

print('#####################################\n'*5)
############################################################################################################
            ###   Reading input data     ###
############################################################################################################
                                 ###   Consuming NumPy arrays     ###

#If all of your input data fits in memory,memory,memory,
# the simplest way to create a Dataset from them is to convert them to tf.Tensor objects and use 'Dataset.from_tensor_slices()'.
train, test = tf.keras.datasets.fashion_mnist.load_data()
images, labels = train
images = images/255
dataset = tf.data.Dataset.from_tensor_slices((images, labels))
print(dataset)

print('Note: The above code snippet will embed the features and labels arrays in your TensorFlow graph as tf.constant() operations. '
      'This works well for a small dataset, but wastes memory---because the contents of the array will be copied multiple times'
      '---and can run into the 2GB limit for the tf.GraphDef protocol buffer.')

# a example:  Loading NumPy arrays
DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
path = tf.keras.utils.get_file('mnist.npz', DATA_URL)
with np.load(path) as data:
  train_examples = data['x_train']
  train_labels = data['y_train']
  test_examples = data['x_test']
  test_labels = data['y_test']

#Assuming you have an array of examples and a corresponding array of labels,
# pass the two arrays as a tuple into tf.data.Dataset.from_tensor_slices to create a tf.data.Dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100
train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)
print(train_dataset)
print(test_dataset)
print('#####################################\n'*5)

############################################################################################################
                                       ###   Consuming Python generators     ###
# Another common data source that can easily be ingested as a tf.data.Dataset is the python generator.
def count(stop):
  i = 0
  while i<stop:
    yield i # iterator stops at i (where yield exist), until next call (next())
    i += 1

for n in count(5):
  print(n)

#The Dataset.from_generator constructor converts the python generator to a fully functional tf.data.Dataset.
# The constructor takes a callable as input, not an iterator. This allows it to restart the generator when it reaches the end.
# It takes an optional args argument, which is passed as the callable's arguments.
# The output_types argument is required because tf.data builds a tf.Graph internally, and graph edges require a tf.dtype.
ds_counter = tf.data.Dataset.from_generator(count, args=[25], output_types=tf.int32, output_shapes = (), )
for count_batch in ds_counter.repeat(3).batch(10).take(10): # every take is a batch
  print(count_batch.numpy())


# The output_shapes argument is not required but is highly recomended as many tensorflow operations do not support tensors with unknown rank.

def gen_series():
  i = 0
  while True:
    size = np.random.randint(0, 10)
    yield i, np.random.normal(size=(size,))
    i += 1
#一个带有 yield 的函数就是一个 generator，它和普通函数不同，生成一个 generator 看起来像函数调用，但不会执行任何函数代码，直到对其调用 next()（在 for 循环中会自动调用 next()）才开始执行。
# https://www.runoob.com/w3cnote/python-yield-used-analysis.html
for i, series in gen_series():
  print(i, ":", str(series))
  if i > 5:
    break
print()
ds_series = tf.data.Dataset.from_generator(
    gen_series,
    output_types=(tf.int32, tf.float32),
    output_shapes=((), (None,)))
print(ds_series)

# Now it can be used like a regular tf.data.Dataset. Note that when batching a dataset with a variable shape, you need to use Dataset.padded_batch.

ds_series_batch = ds_series.shuffle(20).padded_batch(10)

ids, sequence_batch = next(iter(ds_series_batch))
print(ids.numpy(),'\n')
print(sequence_batch.numpy())
# ################################################
# For a more realistic example, try wrapping 'preprocessing.image.ImageDataGenerator' as a 'tf.data.Dataset'.
flowers = tf.keras.utils.get_file(
    'flower_photos',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True)

img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=20) #Create the image.ImageDataGenerator
images, labels = next(img_gen.flow_from_directory(flowers))
print(images.dtype, images.shape)
print(labels.dtype, labels.shape)

ds = tf.data.Dataset.from_generator(
    lambda: img_gen.flow_from_directory(flowers),
    output_types=(tf.float32, tf.float32),
    output_shapes=([32,256,256,3], [32,5])
)

print(ds.element_spec)
for images, label in ds.take(1):
  print('images.shape: ', images.shape)
  print('labels.shape: ', labels.shape)
# ################################################
print('#########   code above have data_set in memory   ###########\n'*5)


############################################################################################################
            ###   Consuming TFRecord data     ###
'''
The tf.data API supports a variety of file formats so that you can process large datasets that do not fit in memory.
'''

# For example, the TFRecord file format is a simple record-oriented binary format that many TensorFlow applications use for training data.
# The tf.data.TFRecordDataset class enables you to stream over the contents of one or more TFRecord files as part of an input pipeline.

# Creates a dataset that reads all of the examples from two files.
fsns_test_file = tf.keras.utils.get_file("fsns.tfrec", "https://storage.googleapis.com/download.tensorflow.org/data/fsns-20160927/testdata/fsns-00000-of-00001")
# The filenames argument to the TFRecordDataset initializer can either be a 'string', 'a list of strings', or a 'tf.Tensor of strings'
#  if you have two sets of files for training and validation purposes, you can create a factory method that produces the dataset,
#  taking filenames as an input argument:
dataset = tf.data.TFRecordDataset(filenames = [fsns_test_file])
print(dataset)

print('Many TensorFlow projects use serialized tf.train.Example records in their TFRecord files. These need to be decoded before they can be inspected:')
raw_example = next(iter(dataset))
parsed = tf.train.Example.FromString(raw_example.numpy())

print(parsed.features.feature['image/text'])
print('#####################################\n'*5)


############################################################################################################
            ###   Consuming text data     ###
print('The tf.data API supports a variety of file formats so that you can process large datasets that do not fit in memory.')

print('Many datasets are distributed as one or more text files. The tf.data.TextLineDataset provides an easy way to extract lines from one or more text files. '
      'Given one or more filenames, a TextLineDataset will produce one string-valued element per line of those files.')

directory_url = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
file_names = ['cowper.txt', 'derby.txt', 'butler.txt']

file_paths = [tf.keras.utils.get_file(file_name, directory_url + file_name) for file_name in file_names]
dataset = tf.data.TextLineDataset(file_paths)

for line in dataset.take(5):#Here are the first few lines of the first file:
  print(line.numpy())

files_ds = tf.data.Dataset.from_tensor_slices(file_paths)
lines_ds = files_ds.interleave(tf.data.TextLineDataset, cycle_length=3)

for i, line in enumerate(lines_ds.take(9)):
  if i % 3 == 0:
    print()
  print(line.numpy())



print('By default, a TextLineDataset yields every line of each file, which may not be desirable, '
      'for example, if the file starts with a header line, or contains comments. '
      'These lines can be removed using the Dataset.skip() or Dataset.filter() transformations. '
      'Here, you skip the first line, then filter to find only survivors.')
titanic_file = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")
titanic_lines = tf.data.TextLineDataset(titanic_file)
for line in titanic_lines.take(15):
  print('first 15 lines',line.numpy())

def survived(line): # For each string in the input Tensor, creates a substring starting at index pos with a total length of len.
  # return tf.math.equal(tf.strings.substr(line, 0, 1), "1")  #tf.strings.substr(input, position, length...)
  return tf.math.not_equal(tf.strings.substr(line, 0, 1), "0")  # tf.strings.substr(input, position, length...)

survivors = titanic_lines.skip(1).filter(survived)
for line in survivors.take(10):
  print('first 10 survivors lines',line.numpy())

############################################################################################################
            ###   Consuming CSV data     ###
print('The CSV file format is a popular format for storing tabular data in plain text.')
titanic_file = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")
df = pd.read_csv(titanic_file, index_col=None)
pd.set_option('display.max_rows', None) #WebSite'pandas中关于DataFrame行，列显示不完全（省略）的解决办法 https://blog.csdn.net/weekdawn/article/details/81389865'
pd.set_option('display.max_columns', None)
print(df.head())
print('If your data fits in memory the same Dataset.from_tensor_slices method works on dictionaries, allowing this data to be easily imported:')

titanic_slices = tf.data.Dataset.from_tensor_slices(dict(df))

for feature_batch in titanic_slices.take(1):
  for key, value in feature_batch.items():
    print("  {!r:20s}: {} ".format(key, value))
  print('\n')

print('A more scalable approach is to load from disk as necessary!!!!!!!!!!!!!!!!!!!!!')
print('The tf.data module provides methods to extract records from one or more CSV files that comply with RFC-4180.')
# The experimental.make_csv_dataset function is the high level interface for reading sets of csv files.
# It supports column type inference and many other features, like batching and shuffling, to make usage simple.
titanic_batches = tf.data.experimental.make_csv_dataset(titanic_file, batch_size=4,label_name="survived")
for feature_batch, label_batch in titanic_batches.take(3):
  print("'survived': {}".format(label_batch))
  print("features:")
  for key, value in feature_batch.items():
    print("  {!r:20s}: {}".format(key, value))
  print("\n")

print('There is also a lower-level experimental.CsvDataset class which provides finer grained control. '
      'It does not support column type inference. Instead you must specify the type of each column.')

titanic_types  = [tf.int32, tf.string, tf.float32, tf.int32, tf.int32, tf.float32, tf.string, tf.string, tf.string, tf.string]
dataset = tf.data.experimental.CsvDataset(titanic_file, titanic_types , header=True)

for line in dataset.take(10):
  print([item.numpy() for item in line])

list=[[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]]
name=['o','ha','os','p']
test=pd.DataFrame(columns=name,data=list)
test.to_csv('/home/brutal/PycharmProjects/Project-YOLOv3/docs/data_test0.csv')
# list1='1,2,3,4,,2,3,4,1,2,,4,1,2,3,,,,'
# test=pd.DataFrame(name, data=list1)
# test.to_csv('/home/brutal/PycharmProjects/Project-YOLOv3/docs/data_test1.csv')
############################################################################################################
            ###   Consuming sets of files    ###
#There are many datasets distributed as a set of files, where each file is an example.
flowers_root = tf.keras.utils.get_file(
    'flower_photos',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True)
print(flowers_root)
flowers_root = pathlib.Path(flowers_root)
print(flowers_root)
print(flowers_root.glob("*")) # Path.glob(pattern):获取路径下的所有符合pattern的文件，返回一个generator
# The root directory contains a directory for each class:
for item in flowers_root.glob("*"):
  print(item.name)

print('str(flowers_root/*/*)',str(flowers_root/'*/*'))
list_ds = tf.data.Dataset.list_files(str(flowers_root/'*/*')) #The files in each class directory are examples
for f in list_ds.take(5):
  print(f.numpy())
# Read the data using the tf.io.read_file function and extract the label from the path, returning (image, label) pairs:
print('os.sep',os.sep)
print('tf.strings.split(file_path, os.sep)',tf.strings.split('/home/brutal/.keras/datasets/flower_photos/daisy/12348343085_d4c396e5b5_m.jpg', os.sep))
def process_path(file_path):
  label = tf.strings.split(file_path, os.sep)[-2]
  return tf.io.read_file(file_path), label

labeled_ds = list_ds.map(process_path)

for image_raw, label_text in labeled_ds.take(1):
  print(repr(image_raw.numpy()[:100]))
  print()
  print(label_text.numpy())
path = pathlib.Path.cwd()# currently working directory
print(path)
print('#####################################\n'*5)


############################################################################################################
            ###  Batching dataset elements    ###
# The simplest form of batching stacks n consecutive elements of a dataset into a single element.
    # The Dataset.batch() transformation does exactly this, with the same constraints as the tf.stack() operator, applied to each component of the elements:
    # i.e. for each component i, all elements must have a tensor of the exact same shape.
inc_dataset = tf.data.Dataset.range(100)
dec_dataset = tf.data.Dataset.range(0, -100, -1)
dataset = tf.data.Dataset.zip((inc_dataset, dec_dataset))
batched_dataset = dataset.batch(4)
for batch in batched_dataset.take(4):
    #While tf.data tries to propagate shape information, the default settings of Dataset.
    # batch result in an unknown batch size because the 'last batch may not be full'. Note the Nones in the shape:
    print('batched_dataset',batched_dataset)
    print([arr.numpy() for arr in batch])

# Use the drop_remainder argument to ignore that last batch, and get full shape propagation
batched_dataset = dataset.batch(7, drop_remainder=True)
print(batched_dataset)

############################################################################################################
            ###  Batching tensors with padding    ###
print('The above recipe works for tensors that all have the same size. '
      'However, many models (e.g. sequence models) work with input data that can have varying size (e.g. sequences of different lengths). '
      'To handle this case, the Dataset.padded_batch transformation enables you to batch tensors of different shape by specifying one or more dimensions '
      'in which they may be padded.')

dataset = tf.data.Dataset.range(100)
# tf.fill: Creates a tensor filled with a scalar value into the designed shape
dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
dataset = dataset.padded_batch(3, padded_shapes=(None,))

for batch in dataset.take(4):
  print(batch.numpy())
  print()
print('The Dataset.padded_batch transformation allows you to set different padding for each dimension of each component, '
      'and it may be variable-length (signified by None in the example above) or constant-length. '
      'It is also possible to override the padding value, which defaults to 0.')
############################################################################################################
            ###  Training workflows    ###
print('The simplest way to iterate over a dataset in multiple epochs is to use the \'Dataset.repeat()\' transformation. '
      'First, create a dataset of titanic data:')
titanic_file = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")
print(titanic_file)
titanic_lines = tf.data.TextLineDataset(titanic_file)
def plot_batch_sizes(ds):
  batch_sizes = [batch.shape[0] for batch in ds]
  plt.bar(range(len(batch_sizes)), batch_sizes)
  plt.xlabel('Batch number')
  plt.ylabel('Batch size')
  plt.show()
# Applying the Dataset.repeat() transformation with no arguments will repeat the input indefinitely:
    # The Dataset.repeat transformation concatenates its arguments without signaling the end of one epoch and the beginning of the next epoch.
    # Because of this a Dataset.batch applied after Dataset.repeat will yield batches that straddle epoch boundaries:
titanic_batches = titanic_lines.repeat(3).batch(128)
plot_batch_sizes(titanic_batches)
# If you need clear epoch separation, put Dataset.batch before the repeat:
titanic_batches = titanic_lines.batch(128).repeat(3)
plot_batch_sizes(titanic_batches)

#If you would like to perform a custom computation (e.g. to collect statistics) at the end of each epoch
# then it's simplest to restart the dataset iteration on each epoch:
epochs = 3
dataset = titanic_lines.batch(128)
for epoch in range(epochs):
  for batch in dataset:
    print(batch.shape)
  print("End of epoch: ", epoch)
print('#####################################\n'*5)



print('Randomly shuffling input data: The Dataset.shuffle() transformation maintains a fixed-size buffer '
      'and chooses the next element uniformly at random from that buffer.')
print('Note: While large buffer_sizes shuffle more thoroughly, they can take a lot of memory, and significant time to fill. '
      'Consider using Dataset.interleave across files if this becomes a problem.')
# Add an index to the dataset so you can see the effect:
lines = tf.data.TextLineDataset(titanic_file)
counter = tf.data.experimental.Counter()

dataset = tf.data.Dataset.zip((counter, lines))
dataset = dataset.shuffle(buffer_size=100)#出几个进几个 then randomly pick
dataset = dataset.batch(20)
print(dataset)
# Since the buffer_size is 100, and the batch size is 20, the first batch contains no 'elements with an index over 120'.
n,line_batch = next(iter(dataset))
print(n.numpy())

# As with Dataset.batch the order relative to Dataset.repeat matters:
print('Dataset.shuffle does not signal the end of an epoch until the shuffle buffer is empty. '
      'So a shuffle placed before a repeat will show every element of one epoch before moving to the next:')
dataset = tf.data.Dataset.zip((counter, lines))
shuffled = dataset.shuffle(buffer_size=200).batch(10).repeat(2)

print("Here are the item ID's near the epoch boundary:\n")
for n, line_batch in shuffled.skip(60).take(5):
  print(n.numpy())
shuffle_repeat = [n.numpy().mean() for n, line_batch in shuffled]
plt.plot(shuffle_repeat, label="shuffle().repeat()")
plt.ylabel("Mean item ID")
plt.legend()
plt.show()

#But a repeat before a shuffle mixes the epoch boundaries together:
dataset = tf.data.Dataset.zip((counter, lines))
shuffled = dataset.repeat(2).shuffle(buffer_size=100).batch(10)

print("Here are the item ID's near the epoch boundary:\n")
for n, line_batch in shuffled.skip(55).take(15):
  print(n.numpy())
repeat_shuffle = [n.numpy().mean() for n, line_batch in shuffled]

plt.plot(shuffle_repeat, label="shuffle().repeat()")
plt.plot(repeat_shuffle, label="repeat().shuffle()")
plt.ylabel("Mean item ID")
plt.legend()
plt.show()


############################################################################################################
            ###  Preprocessing data   ###






