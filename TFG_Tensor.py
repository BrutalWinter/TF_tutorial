'''Tensors are multi-dimensional arrays with a uniform type (called a dtype).
You can see all supported dtypes at tf.dtypes.DType.

If you are familiar with NumPy, tensors are (kind of) like np.arrays.

All tensors are immutable like Python numbers and strings:
you can never update the contents of a tensor, only create a new one!!!!!!!!!!
you can never update the contents of a tensor, only create a new one!!!!!!!!!!
you can never update the contents of a tensor, only create a new one!!!!!!!!!!
you can never update the contents of a tensor, only create a new one!!!!!!!!!!
you can never update the contents of a tensor, only create a new one!!!!!!!!!!!
you can never update the contents of a tensor, only create a new one!!!!!!!!!!
you can never update the contents of a tensor, only create a new one!!!!!!!!!!
you can never update the contents of a tensor, only create a new one!!!!!!!!!!
you can never update the contents of a tensor, only create a new one!!!!!!!!!!
you can never update the contents of a tensor, only create a new one!!!!!!!!!!!'''

import tensorflow as tf
import numpy as np


#######################################################
print('<<=======================   section 1  =======================>>')
# This will be an int32 tensor by default; see "dtypes" below.
rank_0_tensor = tf.constant(4) # rank0 means scalar
print(rank_0_tensor)

# Let's make this a float tensor.
rank_1_tensor = tf.constant([2.0, 3.0, 4.0])
print(rank_1_tensor)

# If you want to be specific, you can set the dtype (see below) at creation time
rank_2_tensor = tf.constant([
                             [1, 2],
                             [3, 4],
                             [5, 6]], dtype=tf.float16)
print(rank_2_tensor)



rank_3_tensor = tf.constant([
  [[0, 1, 2, 3, 4],
   [5, 6, 7, 8, 9]],
  [[10, 11, 12, 13, 14],
   [15, 16, 17, 18, 19]],
  [[20, 21, 22, 23, 24],
   [25, 26, 27, 28, 29]],
])
print(rank_3_tensor)
#You can convert a tensor to a NumPy array either using np.array or the tensor.numpy method:
print('np.array(rank_3_tensor)',np.array(rank_3_tensor))# first way
print('rank_3_tensor.numpy()',rank_3_tensor.numpy())# second way

#Tensors often contain floats and ints,
#but have many other types, including: complex numbers, strings
# The base tf.Tensor class requires tensors to be "rectangular"---that is, along each axis: every element is the same size.
# However, there are specialized types of tensors that can handle different shapes: Ragged tensors and Sparse tensors

'''You can do basic math on tensors'''

a = tf.constant([[1, 2],
                 [3, 4]])
b = tf.constant([[1, 1],
                 [1, 1]],dtype=tf.int32) # Could have also said `tf.ones([2,2])`
c = tf.constant([[4.0, 5.0], [10.0, 1.0]])
print('a',a)
print('b',b)
print('c',c)
print('a + b',a + b, "\n") # element-wise addition
print('a * b',a * b, "\n") # element-wise multiplication
print('a @ b',a @ b, "\n") # matrix multiplication
print('tf.add(a, b)',tf.add(a, b), "\n")
print('tf.multiply(a, b)',tf.multiply(a, b), "\n")
print('tf.matmul(a, b)',tf.matmul(a, b), "\n")
print('tf.reduce_max(c)',tf.reduce_max(c))# Find the largest value
print('tf.argmax(c)',tf.argmax(c))# Find the index of the largest value
print('tf.nn.softmax(c)',tf.nn.softmax(c))# Compute the softmax





#######################################################
print('<<=======================   section 2  =======================>>')
'''
Shape: The length (number of elements) of each of the dimensions of a tensor.
Rank: Number of tensor dimensions. A scalar has rank 0, a vector has rank 1, a matrix is rank 2.
Axis or Dimension: A particular dimension of a tensor.
Size: The total number of items in the tensor, the product shape vector
'''

rank_4_tensor = tf.ones([3, 2, 4, 5])
print("Type of every element:", rank_4_tensor.dtype)
print("Number of dimensions:", rank_4_tensor.ndim)
print("Shape of tensor:", rank_4_tensor.shape)
print("Elements along axis 0 of tensor:", rank_4_tensor.shape[0])
print("Elements along the last axis of tensor:", rank_4_tensor.shape[-1])
print("Total number of elements: ", tf.size(rank_4_tensor).numpy())#(3*2*4*5)
# While axes are often referred to by their indices, you should always keep track of the meaning of each.

# Often axes are ordered from global to local: like (3*2*4*5) (batch, width, height, Features)
# The batch axis first, followed by spatial dimensions, and features for each location last.
# This way feature vectors are contiguous regions of memory.

#######################################################
print('<<=======================   section 3  =======================>>')
'''TensorFlow follows standard Python indexing rules, similar to indexing a list or a string in Python, 
and the basic rules for NumPy indexing.

1.indexes start at 0
2.negative indices count backwards from the end
3.colons, :, are used for slices: start:stop:step'''

'''
Higher rank tensors are indexed by passing multiple indices.
The single-axis exact same rules as in the single-axis case apply to each axis independently.
'''
#Single-axis indexing
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
rank_2_tensor = tf.constant([[1, 2],
                             [3, 4],
                             [5, 6]], dtype=tf.float16)
print(rank_2_tensor[1, 1].numpy())# Pull out a single value from a 2-rank tensor
# Get row and column tensors
print("Second row:", rank_2_tensor[1, :].numpy())
print("Second column:", rank_2_tensor[:, 1].numpy())
print("Last row:", rank_2_tensor[-1, :].numpy())
print("First item in last column:", rank_2_tensor[0, -1].numpy())
print("Skip the first row:",rank_2_tensor[1:, :].numpy(), "\n")

#Here is an example with a 3-axis tensor
print('rank_3_tensor[:, :, 4]',rank_3_tensor[:, :, 4])#reduce the dimension
print('rank_3_tensor[:, :, 4:5]',rank_3_tensor[:, :, 4:5])#keep the dimension
print('rank_3_tensor[:, :, 4:]',rank_3_tensor[:, :, 4:])#keep the dimension
print('rank_3_tensor',rank_3_tensor[:, :, ::-1])
print('rank_3_tensor[:, ::-1, ::-1]',rank_3_tensor[:, ::-1, ::-1])
print('rank_3_tensor[::-1, ::-1, ::-1]',rank_3_tensor[::-1, ::-1, ::-1])


#######################################################
print('<<=======================   section 4  =======================>>')
'''
The data maintains its layout in memory and a new tensor is created, with the requested shape, pointing to the same data.
TensorFlow uses C-style "row-major" memory ordering, where incrementing the rightmost index corresponds to a single step in memory.
'''
'''
Reshaping will "work" for any new shape with the same total number of elements,
but it will not do anything useful if you do not respect the order of the axes.

Swapping axes in tf.reshape does not work; you need tf.transpose for that.
'''
#Manipulating Shapes:
x = tf.constant([[1], [2], [3]])
# Shape returns a `TensorShape` object that shows the size on each dimension
print('x.shape',x.shape)
# You can convert this object into a Python list, too
print('x.shape.as_list()',x.shape.as_list())
# You can reshape a tensor into a new shape.
# The tf.reshape operation is fast and cheap as the underlying data does not need to be duplicated.
reshaped = tf.reshape(x, [1, 3])
print('x.shape',x.shape)
print('reshaped.shape',reshaped.shape)

print('rank_3_tensor',rank_3_tensor)
#If you flatten a tensor you can see what order it is laid out in memory.
print('tf.reshape(rank_3_tensor, [-1])',tf.reshape(rank_3_tensor, [-1]),"\n") # `-1` passed in the `shape` argument says "Whatever fits".
print('tf.reshape(rank_3_tensor, [3*2, 5])',tf.reshape(rank_3_tensor, [3*2, 5]), "\n")
print('tf.reshape(rank_3_tensor, [3, -1])',tf.reshape(rank_3_tensor, [3, -1]))


# Bad examples: don't do this
# You can't reorder axes with reshape.
print('tf.reshape(rank_3_tensor, [2, 3, 5])',tf.reshape(rank_3_tensor, [2, 3, 5]), "\n")
# This is a mess
print('tf.reshape(rank_3_tensor, [5, 6])',tf.reshape(rank_3_tensor, [5, 6]), "\n")
# This doesn't work at all
try:
  tf.reshape(rank_3_tensor, [7, -1])
except Exception as e:
  print(f"{type(e).__name__}: {e}")


#If you don't, TensorFlow chooses a datatype that can represent your data.
# TensorFlow converts Python integers to tf.int32 and Python floating point numbers to tf.float32.
# Otherwise TensorFlow uses the same rules NumPy uses when converting to arrays.#

# To inspect a tf.Tensor's data type use the Tensor.dtype property.
# When creating a tf.Tensor from a Python object you may optionally specify the datatype.
the_f64_tensor = tf.constant([2.2, 3.3, 4.4], dtype=tf.float64)
the_f16_tensor = tf.cast(the_f64_tensor, dtype=tf.float16)
# Now, cast to an uint8 and lose the decimal precision
the_u8_tensor = tf.cast(the_f16_tensor, dtype=tf.uint8)#You can cast from type to type.
print('the_u8_tensor ',the_u8_tensor)

#######################################################
print('<<=======================   section 5  =======================>>')
'''
In short, under certain conditions, smaller tensors are "stretched" automatically to fit larger tensors when running combined operations on them.
The simplest and most common case is when you attempt to multiply or add a tensor to a scalar.
In that case, the scalar is broadcast to be the same shape as the other argument.
'''

x = tf.constant([1, 2, 3])
y = tf.constant(2)
z = tf.constant([2, 2, 2])
# All of these are the same computation
print('tf.multiply(x, 2)',tf.multiply(x, 2))
print('x * y',x * y)
print('x * z',x * z)
# Likewise, 1-sized dimensions can be stretched out to match the other arguments.
# Both arguments can be stretched in the same computation. These are the same computations
x = tf.reshape(x,[3,1])
y = tf.range(1, 5)
print('x',x, "\n")
print('y',y, "\n")
print('tf.multiply(x, y)',tf.multiply(x, y))

# tf.convert_to_tensor:
# Most ops, like tf.matmul and tf.reshape take arguments of class tf.Tensor.
# However, you'll notice in the above case, Python objects shaped like tensors are accepted.

# Most, but not all, ops call convert_to_tensor on non-tensor arguments.
# There is a registry of conversions, and most object classes like NumPy's ndarray, TensorShape,
# Python lists and tf.Variable will all convert automatically.



#######################################################
print('<<=======================   Ragged Tensors  =======================>>')
'''
A tensor with variable numbers of elements along some axis is called "ragged". Use tf.ragged.RaggedTensor for ragged data.
The strings are atomic and cannot be indexed the way Python strings are.
The length of the string is not one of the dimensions of the tensor.
'''
#A tensor with variable numbers of elements along some axis is called "ragged". Use tf.ragged.RaggedTensor for ragged data.
# For example, This cannot be represented as a regular tensor:

ragged_list = [
    [0, 1, 2, 3],
    [4, 5],
    [6, 7, 8],
    [9]]
try:
  tensor = tf.constant(ragged_list)
except Exception as e:
  print(f"{type(e).__name__}: {e}")

# Instead, create a tf.RaggedTensor using tf.ragged.constant
ragged_tensor = tf.ragged.constant(ragged_list)
print(ragged_tensor)
print(ragged_tensor.shape)# The shape of a tf.RaggedTensor contains unknown dimensions


#######################################################
print('<<=======================   String tensors  =======================>>')
'''
tf.string is a dtype, which is to say you can represent data as strings (variable-length byte arrays) in tensors.
The strings are atomic and cannot be indexed the way Python strings are. The length of the string is not one of the axes of the tensor. 
See tf.strings for functions to manipulate them.
'''


scalar_string_tensor = tf.constant("Gray wolf")# Tensors can be strings, too here is a scalar string.
print('scalar_string_tensor',scalar_string_tensor)
# If you have three string tensors of different lengths, this is OK.
tensor_of_strings = tf.constant(["Gray wolf",
                                 "Quick brown fox",
                                 "Lazy dog"])
print('tensor_of_strings',tensor_of_strings)# Note that the shape is (3,). The string length is not included.

# In the above printout the b prefix indicates that tf.string dtype is not a unicode string, but a byte-string.
print('tf.constant("🥳👍")',tf.constant("🥳👍"))


# You can use split to split a string into a set of tensors
print('tf.strings.split(scalar_string_tensor, sep=" ")',tf.strings.split(scalar_string_tensor, sep=" "))
# ...but it turns into a `RaggedTensor` if you split up a tensor of strings,
# as each string might be split into a different number of parts.
print('tf.strings.split(tensor_of_strings)',tf.strings.split(tensor_of_strings))



#######################################################
print('<<=======================   String tensors2  =======================>>')
'''
Although you can't use tf.cast to turn a string tensor into numbers, you can convert it into bytes, and then into numbers.
'''

text = tf.constant("1 10 100")
print('tf.strings.to_number(tf.strings.split(text, " "))',tf.strings.to_number(tf.strings.split(text, " ")))

byte_strings = tf.strings.bytes_split(tf.constant("Duck"))
byte_ints = tf.io.decode_raw(tf.constant("Duck"), tf.uint8) #Although you can't use tf.cast to turn a string tensor into numbers,
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


#######################################################
print('<<=======================   Sparse tensors  =======================>>')
# Sparse tensors store values by index in a memory-efficient manner
sparse_tensor = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]],
                                       values=[1, 2],
                                       dense_shape=[3, 4])
print('sparse_tensor',sparse_tensor, "\n")
# You can convert sparse tensors to dense
print('tf.sparse.to_dense(sparse_tensor)',tf.sparse.to_dense(sparse_tensor))



