'''
A TensorFlow variable is the recommended way to represent shared, persistent state your program manipulates.
This guide covers how to create, update, and manage instances of tf.Variable in TensorFlow.

Variables are created and tracked via the tf.Variable class.
A tf.Variable represents a tensor whose value can be changed by running ops on it!!!!!

Specific ops allow you to read and modify the values of this tensor.
Higher level libraries like tf.keras use tf.Variable to store model parameters.
'''
import tensorflow as tf
'''This notebook discusses variable placement. If you want to see on what device your variables are placed, uncomment this line.'''
# Uncomment below command to see where your variables get placed (see below):
# tf.debugging.set_log_device_placement(True)




#######################################################
print('<<=======================   section 1  =======================>>')
# To create a variable, provide an initial value. The tf.Variable will have the same dtype as the initialization value.
number_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
my_variable = tf.Variable(number_tensor)

# Variables can be all kinds of types, just like tensors
bool_variable = tf.Variable([False, False, False, True])
complex_variable = tf.Variable([5 + 4j, 6 + 1j])

# A variable looks and acts like a tensor, and, in fact, is a data structure backed by a tf.Tensor.
# Like tensors, they have a dtype and a shape, and can be exported to NumPy.
print('my_variable',my_variable)
print("Shape: ", my_variable.shape)
print("DType: ", my_variable.dtype)
print("As NumPy: ", my_variable.numpy())

# Most tensor operations work on variables as expected, although variables cannot be reshaped.
print("my_variable:", my_variable)
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

# ############################################################################################################
#                 #   Variable:Lifecycles, naming, and watching   #
# '''
# In Python-based TensorFlow, tf.Variable instance have the same lifecycle as other Python objects.
# When there are no references to a variable it is automatically deallocated.
#
# Variables can also be named which can help you track and debug them. You can give two variables the same name.
# '''
# ############################################################################################################
# # Create a and b; they will have the same name but will be backed by different tensors.
# my_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
# a = tf.Variable(my_tensor, name="Mark")
# # A new variable with the same name, but different value
# # Note that the scalar add is broadcast
# b = tf.Variable(my_tensor + 1, name="Mark")
#
# # These are elementwise-unequal, despite having the same name
# print(a == b)
# # Variable names are preserved when saving and loading models.
# # By default, variables in models will acquire unique variable names automatically, so you don't need to assign them yourself unless you want to.
#
# # Although variables are important for differentiation, some variables will not need to be differentiated. Y
# # ou can turn off gradients for a variable by setting trainable to false at creation.
# # An example of a variable that would not need gradients is a training step counter:
# step_counter = tf.Variable(1, trainable=False)
# print('############\n'*5)
# ############################################################################################################
#                 #   Placing variables and tensors   #
# '''
# For better performance, TensorFlow will attempt to place tensors and variables on the fastest device compatible with its dtype.
# This means most variables are placed on a GPU if one is available.
#
# However, you can override this. In this snippet, place a float tensor and a variable on the CPU,
# even if a GPU is available. By turning on device placement logging (see Setup), you can see where the variable is placed.
# '''
# ############################################################################################################
# # Note: Although manual placement works, using distribution strategies can be a more convenient and scalable way to optimize your computation.
# with tf.device('CPU:0'):
#   # Create some tensors
#   a = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
#   b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
#   c = tf.matmul(a, b)
#
# print(c)
#
#
# # It's possible to set the location of a variable or tensor on one device and do the computation on another device.
# # This will introduce delay, as data needs to be copied between the devices.
#
# # You might do this, however, if you had multiple GPU workers but only want one copy of the variables.
# with tf.device('CPU:0'):
#   a = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
#   b = tf.Variable([[1.0, 2.0, 3.0]])
#
# with tf.device('GPU:0'):
#   # Element-wise multiply
#   k = a * b
#
# print(k)
# # Because tf.config.set_soft_device_placement is turned on by default, even if you run this code on a device without a GPU,
# # it will still run. The multiplication step will happen on the CPU.
#
# ############################################################################################################
#                 #    Two tensors can be combined into one Dataset object.   #
# print('############\n'*5)
# ############################################################################################################
#
#
# features = tf.constant([[1, 3], [2, 1], [3, 3]]) # ==> 3x2 tensor
# labels = tf.constant(['A', 'B', 'A']) # ==> 3x1 tensor
# dataset = tf.data.Dataset.from_tensor_slices((features, labels))
# for element in dataset.as_numpy_iterator():
#   print(element)
# # Both the features and the labels tensors can be converted to a Dataset object separately and combined after.
# features_dataset = tf.data.Dataset.from_tensor_slices(features)
# labels_dataset = tf.data.Dataset.from_tensor_slices(labels)
# dataset = tf.data.Dataset.zip((features_dataset, labels_dataset))
# for element1,element2 in dataset.as_numpy_iterator():
#   print(element1,element2)
# # A batched feature and label set can be converted to a Dataset
# # in similar fashion.
# batched_features = tf.constant([[[1, 3], [2, 3]],
#                                 [[2, 1], [1, 2]],
#                                 [[3, 3], [3, 2]]], shape=(3, 2, 2))
# batched_labels = tf.constant([['A', 'A'],
#                               ['B', 'B'],
#                               ['A', 'B']], shape=(3, 2, 1))
# dataset = tf.data.Dataset.from_tensor_slices((batched_features, batched_labels))
# for element in dataset.as_numpy_iterator():
#   print(element)
#
#
#
# x_train = tf.random.normal([30, 416, 416, 3], 0, 1, tf.float32, seed=1)
# x_test = tf.random.normal([10, 416, 416, 3], 0, 1, tf.float32, seed=2)
# # Add a channels dimension
# y_train_13 = tf.random.normal([30, 13, 13, 255], 0, 1, tf.float32, seed=3)
# y_train_26 = tf.random.normal([30, 26, 26, 255], 0, 1, tf.float32, seed=3)
# y_train_52 = tf.random.normal([30, 52, 52, 255], 0, 1, tf.float32, seed=3)
# y_train=[y_train_13,y_train_26,y_train_52]
#
# y_test_13 = tf.random.normal([30, 13, 13, 255], 0, 1, tf.float32, seed=3)
# y_test_26 = tf.random.normal([30, 26, 26, 255], 0, 1, tf.float32, seed=3)
# y_test_52 = tf.random.normal([30, 52, 52, 255], 0, 1, tf.float32, seed=3)
# y_test=[y_test_13,y_test_26,y_test_52]
#
#
#
# features_dataset = tf.data.Dataset.from_tensor_slices(x_train)
# labels1_dataset = tf.data.Dataset.from_tensor_slices(y_train_13)
# labels2_dataset = tf.data.Dataset.from_tensor_slices(y_train_26)
# labels3_dataset = tf.data.Dataset.from_tensor_slices(y_train_52)
# dataset1 = tf.data.Dataset.zip((features_dataset, labels1_dataset,labels2_dataset,labels3_dataset))
# c = tf.data.Dataset.range(7, 13).batch(2)## The `datasets` argument may contain an arbitrary number of datasets.
# # for element1, element2,element3,element4 in dataset1.as_numpy_iterator():
#   # print(element1)
#   # print(element2)
#   # print(element3)
#   # print(element4)
#   # print(element1.shape)
#   # print(element2.shape)
#   # print(element3.shape)
#   # print(element4.shape)
#
# dataset2 = tf.data.Dataset.zip((features_dataset, (labels1_dataset,labels2_dataset,labels3_dataset)))
# for element1, element2 in dataset2.as_numpy_iterator():
#   # print(element1)
#   # print(element2)
#
#   print(element1.shape)
#   print(element2[0].shape)
#   print(element2[1].shape)
#   print(element2[2].shape)