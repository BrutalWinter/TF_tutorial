'''
In TensorFlow 2, eager execution is turned on by default.
The user interface is intuitive and flexible (running one-off operations is much easier and faster),
but this can come at the expense of performance and deployability.

You can use tf.function to make graphs out of your programs.
It is a transformation tool that creates Python-independent dataflow graphs out of your Python code.
This will help you create performant and portable models, and it is required to use SavedModel.

This guide will help you conceptualize how tf.function works under the hood so you can use it effectively.

The main takeaways and recommendations are:
Debug in eager mode, then decorate with @tf.function.
Don't rely on Python side effects like object mutation or list appends.
tf.function works best with TensorFlow ops!
tf.function works best with TensorFlow ops!
tf.function works best with TensorFlow ops!
tf.function works best with TensorFlow ops!
tf.function works best with TensorFlow ops!
tf.function works best with TensorFlow ops!
tf.function works best with TensorFlow ops!
tf.function works best with TensorFlow ops!
tf.function works best with TensorFlow ops!
tf.function works best with TensorFlow ops!
tf.function works best with TensorFlow ops!
tf.function works best with TensorFlow ops!
tf.function works best with TensorFlow ops!
tf.function works best with TensorFlow ops!
tf.function works best with TensorFlow ops!
tf.function works best with TensorFlow ops!
NumPy and Python calls are converted to constants.
'''


import tensorflow as tf
import traceback
import contextlib
import timeit
#######################################################
print('<<=======================   section Usage  =======================>>')

# Some helper code to demonstrate the kinds of errors you might encounter.
@contextlib.contextmanager
def assert_raises(error_class):
  try:
    yield
  except error_class as e:
    print('Caught expected exception \n  {}:'.format(error_class))
    traceback.print_exc(limit=2)
  except Exception as e:
    raise e
  else:
    raise Exception('Expected {} to be raised but no error was raised!'.format(error_class))


# A Function you define (by applying the @tf.function decorator) is just like a core TensorFlow operation:
# You can execute it eagerly; you can compute gradients; and so on.

def add(a, b):
  return a + b
add_graph=tf.function(lambda a,b : add(a,b))# The decorator converts `add` into a `Function`.

print('add_graph:',add_graph(tf.ones([2, 2]), tf.ones([2, 2])))

v = tf.Variable(1.0)
with tf.GradientTape() as tape:
  result = add_graph(v, 1.0)
tape.gradient(result, v)

# You can use Functions inside other Functions.

@tf.function
def dense_layer(x, w, b):
  return add_graph(tf.matmul(x, w), b)

print('dense_layer=',dense_layer(tf.ones([3, 2]), tf.ones([2, 2]), tf.ones([2])))

# Functions can be faster than eager code, especially for graphs with many small ops.
# But for graphs with a few expensive ops (like convolutions), you may not see much speedup.



conv_layer = tf.keras.layers.Conv2D(100, 3)

@tf.function
def conv_fn(image):
  return conv_layer(image)

image = tf.zeros([1, 200, 200, 4])
# warm up
print(conv_layer(image).shape)
print(conv_fn(image).shape)
print("Eager conv:", timeit.timeit(lambda: conv_layer(image), number=10))
print("Function conv:", timeit.timeit(lambda: conv_fn(image), number=10))
print("Note how there's not much difference in performance for convolutions")



#######################################################
print('<<=======================   section tracing  =======================>>')
'''A Function runs your program in a TensorFlow Graph. 
However, a tf.Graph cannot represent all the things that you'd write in an eager TensorFlow program. 
For instance, Python supports polymorphism, but tf.Graph requires its inputs to have a specified data type and dimension. 
Or you may perform side tasks like reading command-line arguments, raising an error, or working with a more complex Python object;
none of these things can run in a tf.Graph.

Function bridges this gap by separating your code in two stages:

1) In the first stage, referred to as "tracing", Function creates a new tf.Graph. 
Python code runs normally, but all TensorFlow operations (like adding two Tensors) are deferred: 
they are captured by the tf.Graph and not run.

2) In the second stage, a tf.Graph which contains everything that was deferred in the first stage is run. 
This stage is much faster than the tracing stage.

Depending on its inputs, Function will not always run the first stage when it is called,
Skipping the first stage and only executing the second stage is what gives you TensorFlow's high performance.'''


'''When Function does decide to trace, the tracing stage is immediately followed by the second stage, 
so calling the Function both creates and runs the tf.Graph. 

you can run only the tracing stage with get_concrete_function.'''

@tf.function
def double(a):
  print("Tracing with", a)
  return a + a

print('double(tf.constant(1))',double(tf.constant(1)))
print('double(tf.constant(1.1))',double(tf.constant(1.1)))
print('double(tf.constant("a"))',double(tf.constant("a")))
# This doesn't print 'Tracing with ...'
print(double(tf.constant("b")))
print(double.pretty_printed_concrete_signatures())

'''So far, you've seen that tf.function creates a cached, dynamic dispatch layer over TensorFlow's graph tracing logic. 
To be more specific about the terminology:

1.A tf.Graph is the raw, language-agnostic, portable representation of a TensorFlow computation.
2.A ConcreteFunction wraps a tf.Graph.
3.A Function manages a cache of ConcreteFunctions and picks the right one for your inputs.
4.tf.function wraps a Python function, returning a Function object.
Tracing creates a tf.Graph and wraps it in a ConcreteFunction, also known as a trace.'''

#Cache keys are based on the Function input parameters so changes to global and free variables alone will not create a new trace.

'''Controlling retracing
Retracing, which is when your Function creates more than one trace, 
helps ensures that TensorFlow generates correct graphs for each set of inputs. However, tracing is an expensive operation! 

To control the tracing behavior, you can use input_signature in tf.function to limit tracing.'''


@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.int32),))
def next_collatz(x):
  print("next_collatz(x) Tracing with", x)
  return tf.where(x % 2 == 0, x // 2, 3 * x + 1)
print(tf.constant([1, 2]).shape)
print(tf.constant([2, 3, 4]).shape)
print(next_collatz(tf.constant([1, 2])))
print(next_collatz(tf.constant([2, 3, 4])))
print('next_collatz',next_collatz.pretty_printed_concrete_signatures())
# We specified a 1-D tensor in the input signature, so this should fail.
with assert_raises(ValueError):
  next_collatz(tf.constant([[1, 2], [3, 4]]))

# We specified an int32 dtype in the input signature, so this should fail.
with assert_raises(ValueError):
  next_collatz(tf.constant([1.0, 2.0]))


'''Specify a [None] dimension in tf.TensorSpec to allow for flexibility in trace reuse!reuse!reuse!reuse!reuse!reuse!reuse!reuse!.
not re-trace.

Since TensorFlow matches tensors based on their shape, 
using a None dimension as a wildcard will allow Functions to reuse traces for variably-sized input. 
Variably-sized input can occur if you have sequences of different length, or images of different sizes for each batch.'''

@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.int32),))
def g(x):
  print('g(x) tracing with', x)
  return x

# No retrace!
print('g',g(tf.constant([1, 2, 3])))
print('g',g(tf.constant([1, 2, 3, 4, 5])))


def train_one_step():
  pass

@tf.function
def train(num_steps):
  print("Tracing with num_steps = ", num_steps)
  tf.print("Executing with num_steps = ", num_steps)
  for _ in tf.range(num_steps):
    train_one_step()

print("Retracing occurs for different Python arguments.")
train(num_steps=10)
train(num_steps=20)

print()
print("Traces are reused for Tensor arguments.")
train(num_steps=tf.constant(10))
train(num_steps=tf.constant(20))

def f():
  print('Tracing!')
  tf.print('Executing')

tf.function(f)()
tf.function(f)()


#######################################################
print('<<=======================   Obtaining concrete functions  =======================>>')
'''Every time a function is traced, a new concrete function is created. 
You can directly obtain a concrete function, by using get_concrete_function.'''
@tf.function
def double(a):
  print("Tracing with", a)
  return a + a
print("Obtaining concrete trace")
double_strings = double.get_concrete_function(tf.constant("a"))
print("Executing traced function")
print(double_strings(tf.constant("a")))
print(double_strings(a=tf.constant("b")))
print(double_strings(a=tf.constant("c")))
# print(double_strings(a=tf.constant(2)))
# print('double_strings',double_strings.pretty_printed_concrete_signatures())

# You can also call get_concrete_function on an InputSpec
double_strings_from_inputspec = double.get_concrete_function(tf.TensorSpec(shape=[], dtype=tf.string))
print(double_strings_from_inputspec(tf.constant("c")))
print('double_strings',double_strings)
print('double_strings_from_inputspec',double_strings_from_inputspec)

# You can also directly retrieve a concrete function's signature.
print('double_strings.structured_input_signature',double_strings.structured_input_signature)
print('double_strings.structured_outputs',double_strings.structured_outputs)
with assert_raises(tf.errors.InvalidArgumentError):
  double_strings(tf.constant(1)) #Using a concrete trace with incompatible types will throw an error


'''You may notice that Python arguments are given special treatment in a concrete function's input signature. 
Prior to TensorFlow 2.3, Python arguments were simply removed from the concrete function's signature. 
Starting with TensorFlow 2.3, Python arguments remain in the signature, 
but are constrained to take the value set during tracing.'''


@tf.function
def pow(a, b):
  return a ** b

square = pow.get_concrete_function(a=tf.TensorSpec(shape=[None], dtype=tf.float32), b=2)
print(square)
print(square(tf.constant(10.0)) == 100)

with assert_raises(TypeError):
  square(tf.constant(10.0), b=3)


#Each concrete function is a callable wrapper around a tf.Graph.
# Although retrieving the actual tf.Graph object is not something you'll normally need to do,
# you can obtain it easily from any concrete function.

graph = double_strings.graph
for node in graph.as_graph_def().node:
  print(f'{node.input} -> {node.name}')


#######################################################
print('<<=======================   AutoGraph Transformations  =======================>>')
'''AutoGraph is a library that is on by default in tf.function, 
and transforms a subset of Python eager code into graph-compatible TensorFlow ops. 
This includes control flow like if, for, while.

TensorFlow ops like tf.cond and tf.while_loop continue to work, 
but control flow is often easier to write and understand when written in Python.'''
@tf.function
def f(x):
  while tf.reduce_sum(x) > 1:
    tf.print(x)
    x = tf.tanh(x)
  return x

f(tf.random.uniform([5]))

print(tf.autograph.to_code(f.python_function))#If you're curious you can inspect the code autograph generates.

'''AutoGraph will convert some if <condition> statements into the equivalent tf.cond calls. 
This substitution is made if <condition> is a Tensor. Otherwise, the if statement is executed as a Python conditional.'''

# Conditionals: if
@tf.function
def fizzbuzz(n):
  for i in tf.range(1, n + 1): # n here is a tensor
    print('Tracing for loop')
    if i % 15 == 0:
      print('Tracing fizzbuzz branch')
      tf.print('fizzbuzz')
    elif i % 3 == 0:
      print('Tracing fizz branch')
      tf.print('fizz')
    elif i % 5 == 0:
      print('Tracing buzz branch')
      tf.print('buzz')
    else:
      print('Tracing default branch')
      tf.print(i)

# fizzbuzz(5)
# fizzbuzz(30)
# print(fizzbuzz.pretty_printed_concrete_signatures())
fizzbuzz(tf.constant(5))
fizzbuzz(tf.constant(20))
print(fizzbuzz.pretty_printed_concrete_signatures())



# AutoGraph will convert some for and while statements into the equivalent TensorFlow looping ops, like tf.while_loop.
# If not converted, the for or while loop is executed as a Python loop.

'''This substitution is made in the following situations:

for x in y: if y is a Tensor, convert to tf.while_loop. In the special case where y is a tf.data.Dataset, 
a combination of tf.data.Dataset ops are generated.

while <condition>: if <condition> is a Tensor, convert to tf.while_loop.'''

#A Python loop executes during tracing, adding additional ops to the tf.Graph for every iteration of the loop.

#A TensorFlow loop traces the body of the loop, and dynamically selects how many iterations to run at execution time.
# The loop body only appears once in the generated tf.Graph


'''Looping over Python data
A common pitfall is to loop over Python/Numpy data within a tf.function. 
This loop will execute during the tracing process, adding a copy of your model to the tf.Graph for each iteration of the loop.'''


def measure_graph_size(f, *args):
  g = f.get_concrete_function(*args).graph
  print("{}({}) contains {} nodes in its graph".format(
      f.__name__, ', '.join(map(str, args)), len(g.as_graph_def().node)))

@tf.function
def train(dataset):
  loss = tf.constant(0)
  for x, y in dataset:
    loss += tf.abs(y - x) # Some dummy computation.
  return loss

small_data = [(1, 1)] * 3
big_data = [(1, 1)] * 10
measure_graph_size(train, small_data)
measure_graph_size(train, big_data)

measure_graph_size(train, tf.data.Dataset.from_generator(lambda: small_data, (tf.int32, tf.int32)))
measure_graph_size(train, tf.data.Dataset.from_generator(lambda: big_data, (tf.int32, tf.int32)))







'''Accumulating values in a loop
A common pattern is to accumulate intermediate values from a loop. 
Normally, this is accomplished by appending to a Python list or adding entries to a Python dictionary. 
However, as these are Python side effects, they will not work as expected in a dynamically unrolled loop. 
Use tf.TensorArray to accumulate results from a dynamically unrolled loop.'''

batch_size = 2
seq_len = 3
feature_size = 4

def rnn_step(inp, state):
  return inp + state

@tf.function
def dynamic_rnn(rnn_step, input_data, initial_state):
  # [batch, time, features] -> [time, batch, features]
  input_data = tf.transpose(input_data, [1, 0, 2])
  max_seq_len = input_data.shape[0]

  states = tf.TensorArray(tf.float32, size=max_seq_len)
  state = initial_state
  for i in tf.range(max_seq_len):
    state = rnn_step(input_data[i], state)
    states = states.write(i, state)
  # return tf.transpose(states.stack(), [1, 0, 2])
  return states.stack()
AA=tf.random.uniform([batch_size, seq_len, feature_size])
print(AA)
BB=dynamic_rnn(rnn_step, AA,
            tf.zeros([batch_size, feature_size]))
print(BB)




#######################################################
print('<<=======================   Limitations:1  naive python code =======================>>')
'''
Executing Python side effects:
Side effects, like printing, appending to lists, and mutating globals, can behave unexpectedly inside a Function, 
sometimes executing twice or not all. Pure untransfered python code won't be traced.
They only happen the first time you call a Function with a set of inputs. 
Afterwards, the traced tf.Graph is reexecuted, without executing the Python code (which not be traced).

The general rule of thumb is to avoid relying on Python side effects in your logic and only use them to debug your traces. Otherwise, TensorFlow APIs like tf.data, tf.print, tf.summary, tf.Variable.assign, and tf.TensorArray are the best way to ensure your code will be executed by the TensorFlow runtime with each call.'''

@tf.function
def f(x):
  print("Traced with", x)
  tf.print("Executed with", x)

f(1)
f(1)
f(2)
#If you would like to execute Python code during each invocation of a Function, tf.py_function is an exit hatch.
# The drawback of tf.py_function is that it's not portable or particularly performant, cannot be saved with SavedModel,
# and does not work well in distributed (multi-GPU, TPU) setups.
# Also, since tf.py_function has to be wired into the graph, it casts all inputs/outputs to tensors.


#######################################################
print('<<=======================   Limitations:2  append =======================>>')
'''Changing Python global and free variables counts as a Python side effect, so it only happens during tracing:'''

external_list = []

@tf.function
def side_effect(x):
  print('Python side effect')
  external_list.append(x)
  tf.print(external_list)

side_effect(tf.constant(1))
side_effect(tf.constant(2))
side_effect(tf.constant(3))
# side_effect(1)
# side_effect(2)
# side_effect(3)
# The list append only happened once!
print('external_list',external_list)

#You should avoid mutating containers like lists, dicts, other objects that live outside the Function.
# Instead, use arguments and TF objects.
# For example, the section "Accumulating values in a loop" has one example of how list-like operations can be implemented.

# You can, in some cases, capture and manipulate state if it is a tf.Variable.
# This is how the weights of Keras models are updated with repeated calls to the same ConcreteFunction.

#######################################################
print('<<=======================   Limitations:3   iterators and generators   =======================>>')
'''Using Python iterators and generators
Many Python features, such as generators and iterators, rely on the Python runtime to keep track of state. 
In general, while these constructs work as expected in eager mode, 
they are examples of Python side effects and therefore only happen during tracing.'''



@tf.function
def buggy_consume_next(iterator):
  tf.print("Value:", next(iterator))

iterator = iter([1, 2, 3])
buggy_consume_next(iterator)
# This reuses the first value from the iterator, rather than consuming the next value.
buggy_consume_next(iterator)
buggy_consume_next(iterator)




@tf.function
def good_consume_next(iterator):
  # This is ok, iterator is a tf.data.Iterator
  # This is ok, iterator is a tf.data.Iterator
  # This is ok, iterator is a tf.data.Iterator
  tf.print("Value:", next(iterator))

ds = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
iterator = iter(ds)
good_consume_next(iterator)
good_consume_next(iterator)
good_consume_next(iterator)
good_consume_next(iterator)
good_consume_next(iterator)



#######################################################
print('<<=======================   Limitations:4   Deleting tf.Variables between Function calls   =======================>>')

'''Another error you may encounter is a garbage-collected variable. 
ConcreteFunctions only retain WeakRefs to the variables they close over, so you must retain a reference to any variables.'''

# external_var = tf.Variable(3)
# @tf.function
# def f(x):
#   return x * external_var
#
# traced_f = f.get_concrete_function(tf.TensorSpec(shape=[None], dtype=tf.int32, name="ECGInput"))
# print("Calling concrete function...",traced_f(tf.constant(5)))


external_var = tf.Variable(3)
@tf.function
def f(x):
  return x * external_var

traced_f = f.get_concrete_function(4)
print("Calling concrete function...")
print(traced_f(4))

# The original variable object gets garbage collected, since there are no more
# references to it.
external_var = tf.Variable(4)

print("Calling concrete function after garbage collecting its closed Variable...")
# print(traced_f(4))
with assert_raises(tf.errors.FailedPreconditionError):
  traced_f(4)


#######################################################
print('<<=======================   Limitations:4   Known Issues   =======================>>')

'''Depending on Python global and free variables
Function creates a new ConcreteFunction when called with a new value of a Python argument. 
However, it does not do that for the Python closure, globals, or nonlocals of that Function. 
If their value changes in between calls to the Function, the Function will still use the values they had when it was traced. 
This is different from how regular Python functions work.

For that reason, we recommend a functional programming style that uses arguments instead of closing over outer names.'''

@tf.function
def buggy_add():
  return tf.constant(10) + foo

@tf.function
def recommended_add(foo):
  return tf.constant(12) + foo

foo = 1
print("Buggy:", buggy_add())
print("Correct:", recommended_add(foo))

print("Updating the value of `foo` to 100!")
foo = 100
print("Buggy:", buggy_add())  # Did not change!
print("Correct:", recommended_add(foo))

'''Creating tf.Variables
Function only supports creating variables once, when first called, and then reusing them. 
You cannot create tf.Variables in new traces. Creating new variables in subsequent calls is currently not allowed, 
but will be in the future.'''

@tf.function
def f(x):
  v = tf.Variable(1.0)
  return v

with assert_raises(ValueError):
  f(1.0)


# You can create variables inside a Function as long as those variables are only created the first time the function is executed.
class Count(tf.Module):
  def __init__(self):
    self.count = None

  @tf.function
  def __call__(self):
    if self.count is None:
      self.count = tf.Variable(0)
    return self.count.assign_add(1)

c = Count()
print(c())
print(c())


'''Using with multiple Keras optimizers
You may encounter ValueError: tf.function-decorated function tried to create variables on non-first call. 
when using more than one Keras optimizer with a tf.function. 
This error occurs because optimizers internally create tf.Variables when they apply gradients for the first time.'''

opt1 = tf.keras.optimizers.Adam(learning_rate = 1e-2)
opt2 = tf.keras.optimizers.Adam(learning_rate = 1e-3)

@tf.function
def train_step(w, x, y, optimizer):
   with tf.GradientTape() as tape:
       L = tf.reduce_sum(tf.square(w*x - y))
   gradients = tape.gradient(L, [w])
   optimizer.apply_gradients(zip(gradients, [w]))

w = tf.Variable(2.)
x = tf.constant([-1.])
y = tf.constant([2.])

train_step(w, x, y, opt1)
print("Calling `train_step` with different optimizer...")
with assert_raises(ValueError):
  train_step(w, x, y, opt2)


'''If you need to change the optimizer during training, 
a workaround is to create a new Function for each optimizer, calling the ConcreteFunction directly.'''

opt1 = tf.keras.optimizers.Adam(learning_rate = 1e-2)
opt2 = tf.keras.optimizers.Adam(learning_rate = 1e-3)

# Not a tf.function.
def train_step(w, x, y, optimizer):
   with tf.GradientTape() as tape:
       L = tf.reduce_sum(tf.square(w*x - y))
   gradients = tape.gradient(L, [w])
   optimizer.apply_gradients(zip(gradients, [w]))

w = tf.Variable(2.)
x = tf.constant([-1.])
y = tf.constant([2.])

# Make a new Function and ConcreteFunction for each optimizer.
train_step_1 = tf.function(train_step).get_concrete_function(w, x, y, opt1)
train_step_2 = tf.function(train_step).get_concrete_function(w, x, y, opt2)
for i in range(10):
  if i % 2 == 0:
    train_step_1(w, x, y) # `opt1` is not used as a parameter. w is weight
  else:
    train_step_2(w, x, y) # `opt2` is not used as a parameter.



# Using with multiple Keras models:
# You may also encounter ValueError: tf.function-decorated function tried to create variables on non-first call.
# when passing different model instances to the same Function.
#
# This error occurs because Keras models (which do not have their input shape defined)
# and Keras layers create tf.Variabless when they are first called.
# You may be attempting to initialize those variables inside a Function, which has already been called.
# To avoid this error, try calling model.build(input_shape) to initialize all the weights before training the model.










