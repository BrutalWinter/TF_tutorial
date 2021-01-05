############################################################################################################
                #   package   #
'''If you are familiar with NumPy, tensors are (kind of) like np.arrays.
All tensors are immutable like Python numbers and strings:
you can never update the contents of a tensor, only create a new one.'''
############################################################################################################
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import time

############################################################################################################
                #   Basic   #
############################################################################################################
rank_3_tensor = tf.constant([
  [[0, 1, 2, 3, 4],
   [5, 6, 7, 8, 9]],
  [[10, 11, 12, 13, 14],
   [15, 16, 17, 18, 19]],
  [[20, 21, 22, 23, 24],
   [25, 26, 27, 28, 29]],])
print(rank_3_tensor)
print('np.array(rank_3_tensor)',np.array(rank_3_tensor))
print('rank_3_tensor.numpy()',rank_3_tensor.numpy())

a = tf.constant([[1, 2],
                 [3, 4]])
b = tf.constant([[1, 1],
                 [1, 1]]) # Could have also said `tf.ones([2,2])`
c = tf.constant([[4.0, 5.0], [10.0, 1.0]])
print(a + b, "\n") # element-wise addition
print(a * b, "\n") # element-wise multiplication
print(a @ b, "\n") # matrix multiplication
print(tf.add(a, b), "\n")
print(tf.multiply(a, b), "\n")
print(tf.matmul(a, b), "\n")
print(tf.reduce_max(c))# Find the largest value
print(tf.argmax(c))# Find the index of the largest value
print(tf.nn.softmax(c))# Compute the softmax
print('############\n'*5)
############################################################################################################
                #   About shapes   #
'''
Shape: The length (number of elements) of each of the dimensions of a tensor.
Rank: Number of tensor dimensions. A scalar has rank 0, a vector has rank 1, a matrix is rank 2.
Axis or Dimension: A particular dimension of a tensor.
Size: The total number of items in the tensor, the product shape vector
'''
############################################################################################################
rank_4_tensor = tf.ones([3, 2, 4, 5])
print("Type of every element:", rank_4_tensor.dtype)
print("Number of dimensions:", rank_4_tensor.ndim)
print("Shape of tensor:", rank_4_tensor.shape)
print("Elements along axis 0 of tensor:", rank_4_tensor.shape[0])
print("Elements along the last axis of tensor:", rank_4_tensor.shape[-1])
print("Total number of elements: ", tf.size(rank_4_tensor).numpy())#(3*2*4*5)
#While axes are often referred to by their indices, you should always keep track of the meaning of each.
# Often axes are ordered from global to local: like (3*2*4*5) (batch, width, height, Features)
# The batch axis first, followed by spatial dimensions, and features for each location last. This way feature vectors are contiguous regions of memory.
print('############\n'*5)
############################################################################################################
                #   Indexing   #
'''
indexes start at 0
negative indices count backwards from the end
colons, :, are used for slices: start:stop:step
'''
############################################################################################################
# Single-axis indexing
rank_1_tensor = tf.constant([0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
print("First:", rank_1_tensor[0].numpy())
print("Second:", rank_1_tensor[1].numpy())
print("Last:", rank_1_tensor[-1].numpy())
print("Everything:", rank_1_tensor[:].numpy())
print("Before 4:", rank_1_tensor[:4].numpy())
print("From 4 to the end:", rank_1_tensor[4:].numpy())
print("From 2, before 7:", rank_1_tensor[2:7].numpy())
print("Every other item:", rank_1_tensor[::2].numpy())
print("Reversed:", rank_1_tensor[::-1].numpy())

# Multi-axis indexing
'''
Higher rank tensors are indexed by passing multiple indices.
The single-axis exact same rules as in the single-axis case apply to each axis independently.
'''
rank_2_tensor = tf.constant([[1, 2],
                             [3, 4],
                             [5, 6]], dtype=tf.float16)
print(rank_2_tensor[1, 1].numpy())# Pull out a single value from a 2-rank tensor
# Get row and column tensors
print("Second row:", rank_2_tensor[1, :].numpy())
print("Second column:", rank_2_tensor[:, 1].numpy())
print("Last row:", rank_2_tensor[-1, :].numpy())
print("First item in last column:", rank_2_tensor[0, -1].numpy())
print("Skip the first row:")
print(rank_2_tensor[1:, :].numpy(), "\n")
#Here is an example with a 3-axis tensor
print(rank_3_tensor[:, :, 4])#reduce the dimension
print(rank_3_tensor[:, :, 4:5])#keep the dimension
print(rank_3_tensor[:, :, 4:])#keep the dimension
print(rank_3_tensor[:, :, ::-1])
print(rank_3_tensor[:, ::-1, ::-1])
print(rank_3_tensor[::-1, ::-1, ::-1])

#Manipulating Shapes:
x = tf.constant([[1], [2], [3]])
print(x.shape)# Shape returns a `TensorShape` object that shows the size on each dimension
print(x.shape.as_list())# You can convert this object into a Python list, too
#You can reshape a tensor into a new shape.
# The tf.reshape operation is fast and cheap as the underlying data does not need to be duplicated.
reshaped = tf.reshape(x, [1, 3])
print(x.shape)
print(reshaped.shape)
'''
The data maintains its layout in memory and a new tensor is created, with the requested shape, pointing to the same data. 
TensorFlow uses C-style "row-major" memory ordering, where incrementing the rightmost index corresponds to a single step in memory.
'''
print(rank_3_tensor)
#If you flatten a tensor you can see what order it is laid out in memory.
print(tf.reshape(rank_3_tensor, [-1]))# `-1` passed in the `shape` argument says "Whatever fits".
print(tf.reshape(rank_3_tensor, [3*2, 5]), "\n")
print(tf.reshape(rank_3_tensor, [3, -1]))
'''
Reshaping will "work" for any new shape with the same total number of elements, 
but it will not do anything useful if you do not respect the order of the axes.
'''
# Bad examples: don't do this
# You can't reorder axes with reshape.
print(tf.reshape(rank_3_tensor, [2, 3, 5]), "\n")
# This is a mess
print(tf.reshape(rank_3_tensor, [5, 6]), "\n")
# This doesn't work at all
try:
  tf.reshape(rank_3_tensor, [7, -1])
except Exception as e:
  print(f"{type(e).__name__}: {e}")


the_f64_tensor = tf.constant([2.2, 3.3, 4.4], dtype=tf.float64)
the_f16_tensor = tf.cast(the_f64_tensor, dtype=tf.float16)
# Now, cast to an uint8 and lose the decimal precision
the_u8_tensor = tf.cast(the_f16_tensor, dtype=tf.uint8)#You can cast from type to type.
print(the_u8_tensor)
print('############\n'*5)
############################################################################################################
                #   BroadCasting   #
'''
In short, under certain conditions, smaller tensors are "stretched" automatically to fit larger tensors when running combined operations on them.
The simplest and most common case is when you attempt to multiply or add a tensor to a scalar. 
In that case, the scalar is broadcast to be the same shape as the other argument.
'''
############################################################################################################
x = tf.constant([1, 2, 3])

y = tf.constant(2)
z = tf.constant([2, 2, 2])
# All of these are the same computation
print(tf.multiply(x, 2))
print(x * y)
print(x * z)
#Likewise, 1-sized dimensions can be stretched out to match the other arguments. Both arguments can be stretched in the same computation.
# These are the same computations
x = tf.reshape(x,[3,1])
y = tf.range(1, 5)
print(x, "\n")
print(y, "\n")
print(tf.multiply(x, y))
print('############\n'*5)
#tf.convert_to_tensor, Most ops, like tf.matmul and tf.reshape take arguments of class tf.Tensor.
# Python objects shaped like tensors are accepted. Most, but not all, ops call convert_to_tensor on non-tensor arguments.
# There is a registry of conversions, and most object classes like NumPy's ndarray, TensorShape, Python lists,
# and tf.Variable will all convert automatically.

############################################################################################################
                #   dtype of Tensors   #
'''
A tensor with variable numbers of elements along some axis is called "ragged". Use tf.ragged.RaggedTensor for ragged data.
The strings are atomic and cannot be indexed the way Python strings are.
The length of the string is not one of the dimensions of the tensor.
'''
############################################################################################################
ragged_list = [
    [0, 1, 2, 3],
    [4, 5],
    [6, 7, 8],
    [9]]
try:
  tensor = tf.constant(ragged_list)
except Exception as e:
  print(f"{type(e).__name__}: {e}")

# Instead create a tf.RaggedTensor using tf.ragged.constant
ragged_tensor = tf.ragged.constant(ragged_list)
print(ragged_tensor)
print(ragged_tensor.shape)# The shape of a tf.RaggedTensor contains unknown dimensions

# tf.string is a dtype, which is to say you can represent data as strings (variable-length byte arrays) in tensors.
scalar_string_tensor = tf.constant("Gray wolf")# Tensors can be strings, too here is a scalar string.
print(scalar_string_tensor)
# If you have three string tensors of different lengths, this is OK.
tensor_of_strings = tf.constant(["Gray wolf",
                                 "Quick brown fox",
                                 "Lazy dog"])
print(tensor_of_strings)# Note that the shape is (3,). The string length is not included.
#>> tf.Tensor([b'Gray wolf' b'Quick brown fox' b'Lazy dog'], shape=(3,), dtype=string)
# In the above printout the b prefix indicates that tf.string dtype is not a unicode string, but a byte-string.
print(tf.constant("🥳👍"))
# You can use split to split a string into a set of tensors
print(tf.strings.split(scalar_string_tensor, sep=" "))

# ...but it turns into a `RaggedTensor` if you split up a tensor of strings,
# as each string might be split into a different number of parts.
print(tf.strings.split(tensor_of_strings))

text = tf.constant("1 10 100")
print(tf.strings.to_number(tf.strings.split(text, " ")))

'''
Although you can't use tf.cast to turn a string tensor into numbers, you can convert it into bytes, and then into numbers.
'''
byte_strings = tf.strings.bytes_split(tf.constant("Duck"))
byte_ints = tf.io.decode_raw(tf.constant("Duck"), tf.uint8)#Although you can't use tf.cast to turn a string tensor into numbers,
# you can convert it into bytes, and then into numbers.
print("Byte strings:", byte_strings)
print("Bytes:", byte_ints)


# Or split it up as unicode and then decode it
unicode_bytes = tf.constant("アヒル 🦆")
unicode_char_bytes = tf.strings.unicode_split(unicode_bytes, "UTF-8")
unicode_values = tf.strings.unicode_decode(unicode_bytes, "UTF-8")

print("\nUnicode bytes:", unicode_bytes)
print("\nUnicode chars:", unicode_char_bytes)
print("\nUnicode values:", unicode_values)
# The tf.string dtype is used for all raw bytes data in TensorFlow.
# The tf.io module contains functions for converting data to and from bytes, including decoding images and parsing csv.
print('############\n'*5)


# Sparse tensors store values by index in a memory-efficient manner
sparse_tensor = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]],
                                       values=[1, 2],
                                       dense_shape=[3, 4])
print(sparse_tensor, "\n")
# You can convert sparse tensors to dense
print(tf.sparse.to_dense(sparse_tensor))
print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n'*10)

############################################################################################################
                #   Variable   #
'''
A TensorFlow variable is the recommended way to represent shared, persistent state your program manipulates. 
This guide covers how to create, update, and manage instances of tf.Variable in TensorFlow.

Variables are created and tracked via the tf.Variable class. 
A tf.Variable represents a tensor whose value can be changed by running ops on it. 
Specific ops allow you to read and modify the values of this tensor. 
Higher level libraries like tf.keras use tf.Variable to store model parameters.
'''
############################################################################################################
# To create a variable, provide an initial value. The tf.Variable will have the same dtype as the initialization value.

my_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
my_variable = tf.Variable(my_tensor)

# Variables can be all kinds of types, just like tensors
bool_variable = tf.Variable([False, False, False, True])
complex_variable = tf.Variable([5 + 4j, 6 + 1j])

# A variable looks and acts like a tensor, and, in fact, is a data structure backed by a tf.Tensor.
# Like tensors, they have a dtype and a shape, and can be exported to NumPy.

print("Shape: ", my_variable.shape)
print("DType: ", my_variable.dtype)
print("As NumPy: ", my_variable.numpy())

# Most tensor operations work on variables as expected, although variables cannot be reshaped.
print("A variable:", my_variable)
print("\nViewed as a tensor:", tf.convert_to_tensor(my_variable))
print("\nIndex of highest value:", tf.argmax(my_variable))

# This creates a new tensor; it does not reshape the variable.
print("\nCopying and reshaping: ", tf.reshape(my_variable, ([1,4])))

# As noted above, variables are backed by tensors.
# You can reassign the tensor using tf.Variable.assign.
# Calling assign does not (usually) allocate a new tensor; instead, the existing tensor's memory is reused.
a = tf.Variable([2.0, 3.0])
a.assign([1, 2]) # This will keep the same dtype, float32
try:
  a.assign([1.0, 2.0, 3.0])# Not allowed as it resizes the variable:
except Exception as e:
  print(f"{type(e).__name__}: {e}")

# If you use a variable like a tensor in operations, you will usually operate on the backing tensor.
#
# Creating new variables from existing variables duplicates the backing tensors. Two variables will not share the same memory.
a = tf.Variable([2.0, 3.0])
# Create b based on the value of a
b = tf.Variable(a)
a.assign([5, 6])

# a and b are different
print(a.numpy())
print(b.numpy())

# There are other versions of assign
print(a.assign_add([2,3]).numpy())  # [7. 9.]
print(a.assign_sub([7,9]).numpy())  # [0. 0.]
print('############\n'*5)
############################################################################################################
                #   Variable:Lifecycles, naming, and watching   #
'''
In Python-based TensorFlow, tf.Variable instance have the same lifecycle as other Python objects. 
When there are no references to a variable it is automatically deallocated.

Variables can also be named which can help you track and debug them. You can give two variables the same name.
'''
############################################################################################################
# Create a and b; they will have the same name but will be backed by different tensors.
my_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
a = tf.Variable(my_tensor, name="Mark")
# A new variable with the same name, but different value
# Note that the scalar add is broadcast
b = tf.Variable(my_tensor + 1, name="Mark")

# These are elementwise-unequal, despite having the same name
print(a == b)
# Variable names are preserved when saving and loading models.
# By default, variables in models will acquire unique variable names automatically, so you don't need to assign them yourself unless you want to.

# Although variables are important for differentiation, some variables will not need to be differentiated. Y
# ou can turn off gradients for a variable by setting trainable to false at creation.
# An example of a variable that would not need gradients is a training step counter:
step_counter = tf.Variable(1, trainable=False)
print('############\n'*5)
############################################################################################################
                #   Placing variables and tensors   #
'''
For better performance, TensorFlow will attempt to place tensors and variables on the fastest device compatible with its dtype. 
This means most variables are placed on a GPU if one is available.

However, you can override this. In this snippet, place a float tensor and a variable on the CPU, 
even if a GPU is available. By turning on device placement logging (see Setup), you can see where the variable is placed.
'''
############################################################################################################
# Note: Although manual placement works, using distribution strategies can be a more convenient and scalable way to optimize your computation.
with tf.device('CPU:0'):
  # Create some tensors
  a = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
  c = tf.matmul(a, b)

print(c)


# It's possible to set the location of a variable or tensor on one device and do the computation on another device.
# This will introduce delay, as data needs to be copied between the devices.

# You might do this, however, if you had multiple GPU workers but only want one copy of the variables.
with tf.device('CPU:0'):
  a = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  b = tf.Variable([[1.0, 2.0, 3.0]])

with tf.device('GPU:0'):
  # Element-wise multiply
  k = a * b

print(k)
# Because tf.config.set_soft_device_placement is turned on by default, even if you run this code on a device without a GPU,
# it will still run. The multiplication step will happen on the CPU.

############################################################################################################
                #    Two tensors can be combined into one Dataset object.   #
print('############\n'*5)
############################################################################################################


features = tf.constant([[1, 3], [2, 1], [3, 3]]) # ==> 3x2 tensor
labels = tf.constant(['A', 'B', 'A']) # ==> 3x1 tensor
dataset = tf.data.Dataset.from_tensor_slices((features, labels))
for element in dataset.as_numpy_iterator():
  print(element)
# Both the features and the labels tensors can be converted to a Dataset object separately and combined after.
features_dataset = tf.data.Dataset.from_tensor_slices(features)
labels_dataset = tf.data.Dataset.from_tensor_slices(labels)
dataset = tf.data.Dataset.zip((features_dataset, labels_dataset))
for element1,element2 in dataset.as_numpy_iterator():
  print(element1,element2)
# A batched feature and label set can be converted to a Dataset
# in similar fashion.
batched_features = tf.constant([[[1, 3], [2, 3]],
                                [[2, 1], [1, 2]],
                                [[3, 3], [3, 2]]], shape=(3, 2, 2))
batched_labels = tf.constant([['A', 'A'],
                              ['B', 'B'],
                              ['A', 'B']], shape=(3, 2, 1))
dataset = tf.data.Dataset.from_tensor_slices((batched_features, batched_labels))
for element in dataset.as_numpy_iterator():
  print(element)



x_train = tf.random.normal([30, 416, 416, 3], 0, 1, tf.float32, seed=1)
x_test = tf.random.normal([10, 416, 416, 3], 0, 1, tf.float32, seed=2)
# Add a channels dimension
y_train_13 = tf.random.normal([30, 13, 13, 255], 0, 1, tf.float32, seed=3)
y_train_26 = tf.random.normal([30, 26, 26, 255], 0, 1, tf.float32, seed=3)
y_train_52 = tf.random.normal([30, 52, 52, 255], 0, 1, tf.float32, seed=3)
y_train=[y_train_13,y_train_26,y_train_52]

y_test_13 = tf.random.normal([30, 13, 13, 255], 0, 1, tf.float32, seed=3)
y_test_26 = tf.random.normal([30, 26, 26, 255], 0, 1, tf.float32, seed=3)
y_test_52 = tf.random.normal([30, 52, 52, 255], 0, 1, tf.float32, seed=3)
y_test=[y_test_13,y_test_26,y_test_52]



features_dataset = tf.data.Dataset.from_tensor_slices(x_train)
labels1_dataset = tf.data.Dataset.from_tensor_slices(y_train_13)
labels2_dataset = tf.data.Dataset.from_tensor_slices(y_train_26)
labels3_dataset = tf.data.Dataset.from_tensor_slices(y_train_52)
dataset1 = tf.data.Dataset.zip((features_dataset, labels1_dataset,labels2_dataset,labels3_dataset))
c = tf.data.Dataset.range(7, 13).batch(2)## The `datasets` argument may contain an arbitrary number of datasets.
# for element1, element2,element3,element4 in dataset1.as_numpy_iterator():
  # print(element1)
  # print(element2)
  # print(element3)
  # print(element4)
  # print(element1.shape)
  # print(element2.shape)
  # print(element3.shape)
  # print(element4.shape)

dataset2 = tf.data.Dataset.zip((features_dataset, (labels1_dataset,labels2_dataset,labels3_dataset)))
for element1, element2 in dataset2.as_numpy_iterator():
  # print(element1)
  # print(element2)

  print(element1.shape)
  print(element2[0].shape)
  print(element2[1].shape)
  print(element2[2].shape)
