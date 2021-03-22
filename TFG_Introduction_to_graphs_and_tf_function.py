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

print("Eager execution:", timeit.timeit(lambda: power(x, 100), number=100))

power_as_graph = tf.function(power)
print("Graph execution:", timeit.timeit(lambda: power_as_graph(x, 100), number=100))



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


#######################################################
print('<<===============  tf.function API  ==============>>')
# tf.function(
#     func=None, input_signature=None, autograph=True, experimental_implements=None,
#     experimental_autograph_options=None, experimental_relax_shapes=False,
#     experimental_compile=None, experimental_follow_type_hints=None
# )
'''tf.function constructs a callable that executes a TensorFlow graph (tf.Graph) created 
by trace-compiling the TensorFlow operations in func, effectively executing func as a TensorFlow graph.'''

@tf.function
def f(x, y):
  print('Tracing happens')
  return x ** 2 + y

x1 = tf.constant([2, 3])
y1 = tf.constant([3, -2])
print('f(x, y)',f(x1, y1))
print('f(x, y)',f(x1, y1))
print('f(x, y)',f(y1, x1))
print('\n')

# Features: func may use data-dependent control flow, including if, for, while break, continue and return statements,
# those if, for, while break, continue and return will executed normally:
@tf.function
def f(x):
  print('Tracing happens')
  if tf.reduce_sum(x) > 0:
    return x * x
  else:
    return -x // 2

print('f(tf.constant(-2))',f(tf.constant(-2)))
print('f(tf.constant(2))',f(tf.constant(2)))
print('f(2)',f(2))
print('f(-2)',f(-2))
print('\n')

# func's closure may include tf.Tensor and tf.Variable objects:
@tf.function
def f0():
  print('Tracing happens')
  return x2 ** 2 + y2
x2 = tf.constant([-2, -3])
y2 = tf.Variable([3, -2])
print(f0())
print(f0())

v = tf.Variable(1)
@tf.function
def f1(x):
  print('Tracing happens')
  for i in tf.range(x):
    print('inner Tracing happens')
    v.assign_add(i+1)
    tf.print(v)
f1(3)
f1(3)
f1(3)


#######################################################
print('<<===============  tf.function API 2 ==============>>')
'''Key Point: Any Python side-effects (appending to a list, printing with print, etc) will only happen once, when func is traced. 
To have side-effects executed into your tf.function they need to be written as TF ops:'''

l = []
@tf.function
def f2(x):
  print('Tracing happens')
  for i in x:
    print('inner Tracing happens')
    l.append(i)    # Caution! Will only happen once when tracing, means l only contains single i, since it is a graph, no specific value. only i
    tf.print(l)
f2(tf.constant([1, 2, 3]))
f2(tf.constant([2, 4, 6]))
f2(tf.constant([3, 6, 9]))
print('l',l)



@tf.function
def f3(x):
  ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  for i in range(len(x)):
    ta = ta.write(i, x[i] + 1)
  return ta.stack()

@tf.function
def f3_2(x):
  ta = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
  step=tf.constant(0)
  for i in x:# can not use enurmate
    ta = ta.write(step, i + 1)
    step+=1
  return ta.stack()
print(f3(tf.constant([1., 2, 3])))
print(f3(tf.constant([1., 2, 3])))
print(f3(tf.constant([1.1, 2.2, 3.3],dtype=tf.float32)))
print(f3_2(tf.constant([2, 3, 4])))

print(f3.pretty_printed_concrete_signatures())
#######################################################
print('<<===============  tf.function API 3 ==============>>')
'''Internally, tf.function can build more than one graph, to support arguments with different data types or shapes, 
since TensorFlow can build more efficient graphs that are specialized on shapes and dtypes. 

tf.function also treats any pure Python value as opaque objects, 
and builds a separate graph for each set of Python arguments that it encounters.

To obtain an individual graph, use the get_concrete_function method of the callable created by tf.function. 
It can be called with the same arguments as func and returns a special tf.Graph object:'''

@tf.function
def f(x):
  return x + 1
print(f(tf.constant([1., 2, 3])))
print(f(tf.constant([2., 2, 3])))
print('f3.pretty_printed_concrete_signatures()',f3.pretty_printed_concrete_signatures())
print('isinstance=',isinstance(f.get_concrete_function(1).graph, tf.Graph))
# Caution: Passing python scalars or lists as arguments to tf.function will always build a new graph.
# To avoid this, pass numeric arguments as Tensors whenever possible:


@tf.function
def f(x):
  return tf.abs(x)
f1 = f.get_concrete_function(1)
f2 = f.get_concrete_function(2)  # Slow - builds new graph
print('f1 is f2',f1 is f2)

#######################################################
print('<<===============  tf.function API 4 ==============>>')
f1 = f.get_concrete_function(tf.constant(1))
f2 = f.get_concrete_function(tf.constant(2))  # Fast - reuses f1
print('f1 is f2',f1 is f2)
# Python numerical arguments should only be used when they take few distinct values,
# such as hyperparameters like the number of layers in a neural network.


#Input signatures
#For Tensor arguments, tf.function instantiates a separate graph for every unique set of input shapes and datatypes.
# The example below creates two separate graphs, each specialized to a different shape:

@tf.function
def f(x):
  return x + 1
vector = tf.constant([1.0, 1.0])
matrix = tf.constant([[3.0]])
print('f.get_concrete_function(vector) is f.get_concrete_function(matrix)=',f.get_concrete_function(vector) is f.get_concrete_function(matrix))
print('f.pretty_printed_concrete_signatures()',f.pretty_printed_concrete_signatures())


# An "input signature" can be optionally provided to tf.function to control the graphs traced.
# The input signature specifies the shape and type of each Tensor argument to the function using a tf.TensorSpec object.
# More general shapes can be used.
#
# This is useful to avoid creating multiple graphs when Tensors have dynamic shapes.
# It also restricts the shape and datatype of Tensors that can be used:

@tf.function(
    input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
def f(x):
  return x + 1
vector = tf.constant([1.0, 1.0])
matrix = tf.constant([[3.0]])
print('f.get_concrete_function(vector) is f.get_concrete_function(matrix)=',f.get_concrete_function(vector) is f.get_concrete_function(matrix))
print('f.pretty_printed_concrete_signatures()',f.pretty_printed_concrete_signatures())


# Variables may only be created once

# tf.function only allows creating new tf.Variable objects when it is called for the first time:

class MyModule(tf.Module):
  def __init__(self):
    self.x = None

  @tf.function
  def __call__(self, x):
    # v = tf.Variable(tf.constant(x)) # ValueError: tf.function-decorated function tried to create variables on non-first call.
    # return v * (x ** 2)
    if self.v is None:
      # self.v = tf.Variable(tf.ones_like(x))
      self.v = tf.Variable(tf.constant(x))
    return self.v * (x**2)

AA=MyModule()
print('MyModule(3)',AA(3))
print('MyModule(3)',AA(4))
print('MyModule(3)',AA(5))
# In general, it is recommended to create stateful objects like tf.Variable outside of tf.function and passing them as arguments.



