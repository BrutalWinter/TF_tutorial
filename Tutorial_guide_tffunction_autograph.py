import tensorflow as tf
import timeit
from datetime import datetime
############################################################################################################
            ###          Introduction to Graphs and tf.function               ###
'''
In this guide you'll see the core of how TensorFlow allows you to make simple changes to your code to get graphs,
and how they are stored and represented, and how you can use them to accelerate and export your models.

TensorFlow running eagerly. This means TensorFlow operations are executed by Python, operation by operation, and returning results back to Python.
1.Eager TensorFlow takes advantage of GPUs, allowing you to place variables, tensors, and even operations on GPUs and TPUs.
2.It is also easy to debug.
3.For some users, you may never need or want to leave Python.
'''
############################################################################################################
#Graphs are data structures that contain a set of tf.Operation objects, which represent units of computation;
# and tf.Tensor objects, which represent the units of data that flow between operations.
# They are defined in a tf.Graph context. Since these graphs are data structures, they can be saved, run, and restored all without the original Python code.

# Graphs are also easily optimized, allowing the compiler to do transformations like:
# 1.Statically infer the value of tensors by folding constant nodes in your computation ("constant folding").
# 2.Separate sub-parts of a computation that are independent and split them between threads or devices.
# 3.Simplify arithmetic operations by eliminating common subexpressions.
# 1-3.In short, graphs are extremely useful and let your TensorFlow run fast, run in parallel, and run efficiently on multiple devices.

#check devices:
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
print("GPUs:", physical_devices)

physical_devices = tf.config.list_physical_devices('CPU')
print("Num CPUs:", len(physical_devices))
print("CPUs:", physical_devices)

with tf.device("/cpu:0"):
    # Define a 'Python function'
    def function_to_get_faster(x, y, b):
      x = tf.matmul(x, y)
      x = x + b
      return x

    # Create a `Function` object that contains a graph
    a_function_that_uses_a_graph = tf.function(function_to_get_faster)

    # Make some tensors
    x1 = tf.constant([[1.0, 2.0]])
    y1 = tf.constant([[2.0], [3.0]])
    b1 = tf.constant(4.0)

    # It just works!
    result=a_function_that_uses_a_graph(x1, y1, b1).numpy()
    print(result)




# 'tf.function'-ized functions are Python callables that work the same as their Python equivalents.
# They have a particular class (python.eager.def_function.Function), but to you they act just as the non-traced version.
#
# tf.function recursively recursively recursively traces any Python function it calls.
with tf.device("/cpu:0"):
    def inner_function(x, y, b):
      x = tf.matmul(x, y)
      x = x + b
      return x

    # Use the decorator
    @tf.function
    def outer_function(x):
      y = tf.constant([[2.0], [3.0]])
      b = tf.constant(4.0)

      return inner_function(x, y, b)

    # Note that the callable will create a graph that
    # includes inner_function() as well as outer_function()
    print(outer_function(tf.constant([[1.0, 2.0]])).numpy())
    print('----------------' * 5, 'Next part0', '----------------' * 5)

############################################################################################################
            ###          Flow control and side effects               ###
############################################################################################################
# Flow control and loops are converted to TensorFlow via 'tf.autograph' by default.
# Autograph uses a combination of methods, including standardizing loop constructs, unrolling, and 'AST' manipulation.
def my_function(x):
  if tf.reduce_sum(x) <= 1:
    return x * x
  else:
    return x-1

a_function = tf.function(my_function)

print("First branch, with graph:", a_function(tf.constant(1.0)).numpy())
print("Second branch, with graph:", a_function(tf.constant([5.0, 5.0])).numpy())

'''You can directly call the Autograph conversion to see how Python is converted into TensorFlow ops.
This is, mostly, unreadable, but you can see the transformation:'''
# Don't read the output too carefully.
print(tf.autograph.to_code(my_function))# Autograph automatically converts if-then clauses, loops, break, return, continue, and more.
# Most of the time, Autograph will work without special considerations. However, there are some caveats.
# tf.function guide can help here, as well as the complete autograph reference
print('----------------' * 5, 'Next part1', '----------------' * 5)


############################################################################################################
            ###         Seeing the speed up               ###
'''
Just wrapping a tensor-using function in tf.function does not automatically speed up your code. For small functions called a few times on a single machine,
the overhead of calling a graph or graph fragment may dominate runtime. Also, if most of the computation was already happening on an accelerator,
such as stacks of GPU-heavy convolutions, the graph speedup won't be large.

For complicated computations, graphs can provide a significant speedup. This is because graphs reduce the Python-to-device communication and perform some speedups.
'''
# in order to work right, you need to Ctrl+/ to close other section
############################################################################################################
#This code times a few runs on some small dense layers.

class SequentialModel(tf.keras.Model):
  def __init__(self, **kwargs):
    super(SequentialModel, self).__init__(**kwargs)
    self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
    self.dense_1 = tf.keras.layers.Dense(128, activation="relu")
    self.dropout = tf.keras.layers.Dropout(0.2)
    self.dense_2 = tf.keras.layers.Dense(10)

  def call(self, x, **kwargs):
    x = self.flatten(x)
    x = self.dense_1(x)
    x = self.dropout(x)
    x = self.dense_2(x)
    return x

input_data = tf.random.uniform([60, 28, 28])

eager_model = SequentialModel()
graph_model = tf.function(eager_model)
#timeit.timeit(stmt='pass', setup='pass', timer=<default timer>, number=1000000)：
# 创建一个Timer实例，参数分别是stmt（需要测量的语句或函数），setup（初始化代码或构建环境的导入语句），timer（计时函数），number（每一次测量中语句被执行的次数）
# close below to speed up other section
print("Eager time:", timeit.timeit(lambda: eager_model(input_data), number=10000))
print("Graph time:", timeit.timeit(lambda: graph_model(input_data), number=10000))
print('----------------' * 5, 'Next part2', '----------------' * 5)


############################################################################################################
            ###         Polymorphic functions               ###
# When you trace a function, you create a Function object that is polymorphic.
# A polymorphic function is a Python callable that encapsulates several concrete function graphs behind one API.
'''
Conceptually, then:
    'A tf.Graph' is the raw, portable data structure describing a computation
        'A Function' is a caching, tracing, dispatcher over ConcreteFunctions
            'A ConcreteFunction' is an eager-compatible wrapper around a graph that lets you execute the graph from Python
'''

############################################################################################################
#Inspecting polymorphic functions
# You can inspect a_function. In this example, calling a_function with three kinds of arguments results in three different concrete functions.
def my_function1(x):
  if tf.reduce_sum(x) <= 1:
    return x * x
  else:
    return x-1

a_function1 = tf.function(my_function1)

print(a_function1)

print("Calling a `Function`:")
print("Int:", a_function1(tf.constant(2)))
print("Float:", a_function1(tf.constant(2.0)))
print("Rank-1 tensor of floats", a_function1(tf.constant([2.0, 2.0, 2.0])))


# Get the concrete function that works on floats
print("Inspecting concrete functions")
print("Concrete function for float:")
print(a_function.get_concrete_function(tf.TensorSpec(shape=[], dtype=tf.float32)))
print("Concrete function for tensor of floats:")
print(a_function.get_concrete_function(tf.constant([2.0, 2.0, 2.0])))

# Concrete functions are callable
# But!!!!!!!!!!, You won't normally do this, but instead just call the containing `Function`!!!!!!!!!!!!!!!!!!!!!
cf = a_function.get_concrete_function(tf.constant(2))
print("Directly calling a concrete function:", cf(tf.constant(4)))
# In this example, you are seeing pretty far into the stack.
# Unless you are specifically managing tracing, you will not normally need to call concrete functions directly as shown above.
print('----------------' * 5, 'Next part3', '----------------' * 5)

############################################################################################################
            ###         Polymorphic functions               ###
# You may find yourself looking at long stack traces, specially ones that refer to tf.Graph or with tf.Graph().as_default().
# This means you are likely running in a graph context. Core functions in TensorFlow use graph contexts, such as Keras's model.fit().
#
# It is often much easier to debug eager execution. Stack traces should be relatively short and easy to comprehend.
#
# In situations where the graph makes debugging tricky, you can revert to using eager execution to debug.
'''
Here are ways you can make sure you are running eagerly:
    1.Call models and layers directly as callables
    2.When using Keras compile/fit, at compile time use model.compile(run_eagerly=True)
    3.Set global execution mode via tf.config.run_functions_eagerly(True)
'''
############################################################################################################
# Method 1 : Using run_eagerly=True:
class EagerLayer(tf.keras.layers.Layer):# Define an identity layer with an eager side effect
  def __init__(self, **kwargs):
    super(EagerLayer, self).__init__(**kwargs)
    # Do some kind of initialization here
    self.value=1

  def call(self, inputs, **kwargs):
    print("\nCurrently running eagerly", str(datetime.now()))# here a python func, an eager func
    return inputs*self.value



class SequentialModel1(tf.keras.Model):# Create an override model to classify pictures, adding the custom layer
  def __init__(self, **kwargs):
    super(SequentialModel1, self).__init__(**kwargs)
    self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
    self.dense_1 = tf.keras.layers.Dense(128, activation="relu")
    self.dropout = tf.keras.layers.Dropout(0.2)
    self.dense_2 = tf.keras.layers.Dense(10)
    self.eager = EagerLayer()

  def call(self, x,**kwargs):
    x = self.flatten(x)
    x = self.dense_1(x)
    x = self.dropout(x)
    x = self.dense_2(x)
    return self.eager(x)

# Create an instance of this model
model1 = SequentialModel1()

# Generate some nonsense pictures and labels
input_data1 = tf.random.uniform([60, 28, 28])
labels1 = tf.random.uniform([60])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# First, compile the model without eager. Note that the model is not traced;
# despite its name, compile only sets up loss functions, optimization, and other training parameters.
model1.compile(run_eagerly=False, loss=loss_fn)
#Now, call fit and see that the function is traced (twice) and then the eager effect never runs again due to tf.function autograph.
model1.fit(input_data1, labels1, epochs=3)
'''
compile(optimizer='rmsprop', loss=None, metrics=None, loss_weights=None,
    weighted_metrics=None, run_eagerly=None, **kwargs)
run_eagerly	Bool. Defaults to False. If True, this Model's logic will not be wrapped in a tf.function. 
Recommended to leave this as None unless your Model cannot be run inside a tf.function.
'''

print("Running eagerly")
# When compiling the model, set it to run eagerly
model1.compile(run_eagerly=True, loss=loss_fn)
model1.fit(input_data1, labels1, epochs=2)



# Method 2 : Using run_functions_eagerly:
# You can also globally set everything to run eagerly.
# This is a switch that bypasses the polymorphic function's traced functions and calls the original function directly.
# You can use this for debugging~~~~~~~~~~~~~~~

# Now, globally set everything to run eagerly
tf.config.run_functions_eagerly(True)
print("Run all functions eagerly.")

# Create a polymorphic function
polymorphic_function = tf.function(model1)

print("Tracing")
# This does, in fact, trace the function
print(polymorphic_function.get_concrete_function(input_data1))

print("\nCalling twice eagerly")
# When you run the function again, you will see the side effect twice, as the function is running eagerly.
result = polymorphic_function(input_data1)
result = polymorphic_function(input_data1)
# Don't forget to set it back when you are done
tf.config.experimental_run_functions_eagerly(False)

print('----------------' * 5, 'Next part4', '----------------' * 5)
############################################################################################################
            ###         Tracing and performance              ###
'''
Tracing costs some overhead. Although tracing small functions is quick, large models can take noticeable wall-clock time to trace. 
    1.This investment is usually quickly paid back with a performance boost.
    2.but it's important to be aware that the first few epochs of any large model training can be slower due to tracing.
    
You can add an 'eager-only' side effect such as 'printing a Python argument', so you can see when the function is being traced. 
Here, you see extra retracing because new Python arguments always trigger retracing.
'''
############################################################################################################
# No matter how large your model, you want to avoid tracing frequently.
# This section of the tf.function guide discusses how to set input specifications and use tensor arguments to avoid retracing.
# If you find you are getting unusually poor performance, it's good to check to see if you are retracing accidentally.

# Use @tf.function decorator
@tf.function
def a_function_with_python_side_effect(x):
  print("Tracing!")  # This eager
  return x * x + tf.constant(2)

# This is traced the first time
print(a_function_with_python_side_effect(tf.constant(2)))
# The second time through, you won't see the side effect
print(a_function_with_python_side_effect(tf.constant(3)))

# This retraces each time the Python argument changes,as a Python argument could be an epoch count or other hyperparameter
print(a_function_with_python_side_effect(2))
print(a_function_with_python_side_effect(3))

#key point using tensor at anytime anywhere if it is possible!
















