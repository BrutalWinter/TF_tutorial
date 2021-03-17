'''Overview
This guide goes beneath the surface of TensorFlow and Keras to see how TensorFlow works.
If you instead want to immediately get started with Keras, please see our collection of Keras guides.

In this guide you'll see the core of how TensorFlow allows you to make simple changes to your code to get graphs,
how graphs are stored and represented, and how you can use them to accelerate your models.

Note: For those of you who are only familiar with TensorFlow 1.x, this guide demonstrates a very different view of graphs.
This is a big-picture overview that covers how tf.function allows you to switch from eager execution to graph execution.
For a more complete specification of tf.function, see the tf.function guide.'''


'''What are graphs?
In the previous three guides, you have seen TensorFlow running eagerly. 
This means TensorFlow operations are executed by Python, operation by operation, and returning results back to Python.

While eager execution has several unique advantages, 
graph execution enables portability outside Python and tends to offer better performance. 
Graph execution means that tensor computations are executed as a TensorFlow graph, 
sometimes referred to as a tf.Graph or simply a "graph."'''


import tensorflow as tf
import timeit
from datetime import datetime
#######################################################
print('<<=======================   section 1  =======================>>')
# Define a Python function.
def a_regular_function(x, y, b):
  x = tf.matmul(x, y)
  x = x + b
  return x

# `a_function_that_uses_a_graph` is a TensorFlow `Function`.
a_function_that_uses_a_graph = tf.function(lambda x, y, b :a_regular_function(x,y,b))

# Make some tensors.
x1 = tf.constant([[1.0, 2.0]])
y1 = tf.constant([[2.0], [3.0]])
b1 = tf.constant(4.0)

orig_value = a_regular_function(x1, y1, b1)
print('orig_value',orig_value,orig_value.numpy())
# Call a `Function` like a Python function.
tf_function_value = a_function_that_uses_a_graph(x1, y1, b1)
print('tf_function_value',tf_function_value,tf_function_value.numpy())
assert(orig_value == tf_function_value)



#######################################################
print('<<=======================   section 2  =======================>>')
def inner_function(x, y, b):
  x = tf.matmul(x, y)
  x = x + b
  return x

# Use the decorator to make `outer_function` a `Function`.
@tf.function
def outer_function(x):
  y = tf.constant([[2.0], [3.0]])
  b = tf.constant(4.0)

  return inner_function(x, y, b)

# Note that the callable will create a graph that
# includes `inner_function` as well as `outer_function`.
value=outer_function(tf.constant([[1.0, 2.0]]))
print('value',value,value.numpy())




#######################################################
print('<<=======================   section 3  =======================>>')
'''Any function you write with TensorFlow will contain a mixture of native TF operations and Python logic, 
such as if-then clauses, loops, break, return, continue, and more. 

While TensorFlow operations are easily captured by a tf.Graph, 
Python-specific logic needs to undergo an extra step in order to become part of the graph. 
tf.function uses a library called AutoGraph (tf.autograph) to convert Python code into graph-generating code.'''

def simple_relu(x):
  if tf.greater(x, 0):
    return x
  else:
    return 0

# `tf_simple_relu` is a TensorFlow `Function` that wraps `simple_relu`.
tf_simple_relu = tf.function(simple_relu)

print("First branch, with graph:", tf_simple_relu(tf.constant(1)).numpy())
print("Second branch, with graph:", tf_simple_relu(tf.constant(-1)).numpy())

# This is the graph-generating output of AutoGraph.
print(tf.autograph.to_code(simple_relu))

# This is the graph itself.
print(tf_simple_relu.get_concrete_function(tf.constant(1)).graph.as_graph_def())





#######################################################
print('<<=======================   section 4  =======================>>')
'''A tf.Graph is specialized to a specific type of inputs (for example, tensors with a specific dtype or objects with the same id()).

Each time you invoke a Function with new dtypes and shapes in its arguments, 
Function creates a new tf.Graph for the new arguments. 
The dtypes and shapes of a tf.Graph's inputs are known as an input signature or just a signature.'''


@tf.function
def my_relu(x):
  return tf.maximum(0., x)

# `my_relu` creates new graphs as it sees more signatures.
print(my_relu(tf.constant(5.5))) # first time, dtype tensor float
print(my_relu([1, -1]))# python navive not tensor
print(my_relu(tf.constant([3., -3.])))# shape differ


# If the Function has already been called with that signature, Function does not create a new tf.Graph.
# These two calls do *not* create new graphs.
print(my_relu(tf.constant(-2.5))) # Signature matches `tf.constant(5.5)`.
print(my_relu(tf.constant([-1., 1.]))) # Signature matches `tf.constant([3., -3.]


# There are three `ConcreteFunction`s (one for each graph) in `my_relu`.
# The `ConcreteFunction` also knows the return type and shape!
print(my_relu.pretty_printed_concrete_signatures())


#######################################################
print('<<=======================   section 5  =======================>>')
'''So far, you've seen how you can convert a Python function into a graph simply by using tf.function as a decorator or wrapper. 
But in practice, getting tf.function to work correctly can be tricky! 
In the following sections, you'll learn how you can make your code work as expected with tf.function.

The code in a Function can be executed both eagerly and as a graph. 
By default, Function executes its code as a graph:'''




def get_MSE(y_true, y_pred):
  sq_diff = tf.pow(y_true - y_pred, 2)
  return tf.reduce_mean(sq_diff)

get_MSE_function=tf.function(lambda y_true, y_pred :get_MSE(y_true, y_pred))

y_true = tf.random.uniform([5], maxval=10, dtype=tf.int32)
y_pred = tf.random.uniform([5], maxval=10, dtype=tf.int32)
print(y_true)
print(y_pred)
print(get_MSE_function(y_true, y_pred))


'''To verify that your Function's graph is doing the same computation as its equivalent Python function, 
we can make it execute eagerly with tf.config.run_functions_eagerly(True). 
This is a switch that turns off Function's ability to create and run graphs, instead executing the code normally.'''
tf.config.run_functions_eagerly(True)
print(get_MSE_function(y_true, y_pred))
# Don't forget to set it back when you are done.
tf.config.run_functions_eagerly(False)


#######################################################
print('<<=======================   section 6  =======================>>')


@tf.function
def get_MSE1(y_true, y_pred):
  print("Calculating MSE!") #Python code
  sq_diff = tf.pow(y_true - y_pred, 2) # tf code
  return tf.reduce_mean(sq_diff)

# Observe what is printed:
error = get_MSE1(y_true, y_pred)
error = get_MSE1(y_true, y_pred)
error = get_MSE1(y_true, y_pred)
#  get_MSE only printed once even though it was called three times!!
print('\\')



'''To explain, the print statement is executed when Function runs the original code in order to create the graph in a process known as "tracing".
 Tracing captures the TensorFlow operations into a graph, and print is not captured in the graph. 
 That graph is then executed for all three calls without ever running the Python code again.'''
tf.config.run_functions_eagerly(True)
# Observe what is printed below.
error = get_MSE1(y_true, y_pred)
error = get_MSE1(y_true, y_pred)
error = get_MSE1(y_true, y_pred)
tf.config.run_functions_eagerly(False)
#Note: If you would like to print values in both eager and graph execution, use tf.print instead.


#######################################################
print('<<=======================   section 7  =======================>>')
# tf.function usually improves the performance of your code, but the amount of speed-up depends on the kind of computation you run
x = tf.random.uniform(shape=[10, 10], minval=-1, maxval=2, dtype=tf.dtypes.int32)

def power(x, y):
  result = tf.eye(10, dtype=tf.dtypes.int32)
  for _ in range(y):
    result = tf.matmul(x, result)
  return result

print("Eager execution:", timeit.timeit(lambda: power(x, 100), number=1000))

power_as_graph = tf.function(power)
print("Graph execution:", timeit.timeit(lambda: power_as_graph(x, 100), number=1000))



#######################################################
print('<<=======================   section 8  =======================>>')
'''Performance and trade-offs
Graphs can speed up your code, but the process of creating them has some overhead. 
For some functions, the creation of the graph takes more time than the execution of the graph. 
This investment is usually quickly paid back with with the performance boost of subsequent executions, 
but it's important to be aware that the first few steps of any large model training can be slower due to tracing.

No matter how large your model, you want to avoid tracing frequently. 
The tf.function guide discusses how to set input specifications and use tensor arguments to avoid retracing. 
If you find you are getting unusually poor performance, it's a good idea to check if you are retracing accidentally.'''



# To figure out when your Function is tracing, add a print statement to its code.
# As a rule of thumb, Function will execute the print statement every time it traces.


@tf.function
def a_function_with_python_side_effect(x):
  print("Tracing!") # An eager-only side effect.
  return x * x + 2

# This is traced the first time.
print(a_function_with_python_side_effect(tf.constant(2)))
# The second time through, you won't see the side effect.
print(a_function_with_python_side_effect(tf.constant(3)))


# This retraces each time the Python argument changes,
# as a Python argument could be an epoch count or other
# hyperparameter.
print(a_function_with_python_side_effect(2))# each python naive number is viewed as differenct
print(a_function_with_python_side_effect(3))















