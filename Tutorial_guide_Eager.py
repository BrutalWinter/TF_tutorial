############################################################################################################
                #   package   #
#TensorFlow's eager execution is an imperative programming environment
#that evaluates operations immediately, without building graphs:
#While eager execution makes development and debugging more interactive,
# TensorFlow 1.x style graph execution has advantages for distributed training, performance optimizations, and production deployment.
# To bridge this gap, TensorFlow 2.0 introduces functions via the tf.function API. For more information, see the tf.function guide.
############################################################################################################
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from functools import wraps
from functools import reduce
# '''for each section to learn, just remove # mark'''
############################################################################################################
                #   Train a model   #
############################################################################################################
# Fetch and format the mnist data
(mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data()

dataset = tf.data.Dataset.from_tensor_slices(
  (tf.cast(mnist_images[...,tf.newaxis]/255, tf.float32),
   tf.cast(mnist_labels,tf.int64)))
dataset = dataset.shuffle(1000).batch(32)


mnist_model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(16,[3,3], activation='relu',
                         input_shape=(None, None, 1)),
  tf.keras.layers.Conv2D(16,[3,3], activation='relu'),
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(10)
])


for images,labels in dataset.take(1):
  print("Logits: ", mnist_model(images[0:1]).numpy())


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


loss_history = []
def train_step(images, labels):
    with tf.GradientTape() as tape:
        logits = mnist_model(images, training=True)
        # Add asserts to check the shape of the output.
        tf.debugging.assert_equal(logits.shape, (32, 10))
        loss_value = loss_object(labels, logits)

    loss_history.append(loss_value.numpy().mean())
    #Computes the gradient using operations recorded in context of this tape.
    grads = tape.gradient(loss_value, mnist_model.trainable_variables) # target and target will be differentiated against elements in sources.
    # apply_gradients(grads_and_vars)
    optimizer.apply_gradients(zip(grads, mnist_model.trainable_variables))


def train(epochs):
  for epoch in range(epochs):
    for (batch, (images, labels)) in enumerate(dataset):
      # print('batch',batch)
      train_step(images, labels)
    print ('Epoch {} finished'.format(epoch))

train(epochs = 3)
plt.plot(loss_history)
plt.xlabel('Batch #')
plt.ylabel('Loss [entropy]')
plt.show()
############################################################################################################
                #   Variables and optimizers   #
###########################################################################################################
class Linear(tf.keras.Model):
  def __init__(self):
    super(Linear, self).__init__()
    self.W = tf.Variable(5., name='weight')
    self.B = tf.Variable(10., name='bias')

  def call(self, inputs):

    def add_on(x):
      return x * 2

    return add_on(inputs) * self.W + self.B

print('model initial complete')
# A toy dataset of points around 3 * x + 2
NUM_EXAMPLES = 2000
training_inputs = tf.random.normal([NUM_EXAMPLES])
noise = tf.random.normal([NUM_EXAMPLES])
training_outputs = training_inputs * 3 + 2 + noise

# The loss function to be optimized
def loss(model, inputs, targets):
  error = model(inputs) - targets
  return tf.reduce_mean(tf.square(error))

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return tape.gradient(loss_value, [model.W, model.B])

'''
1 Create the model.
2 The Derivatives of a loss function with respect to model parameters.
3 A strategy for updating the variables based on the derivatives.
'''
model = Linear()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
print("Initial loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))


steps = 300
for i in range(steps):
  grads = grad(model, training_inputs, training_outputs)
  optimizer.apply_gradients(zip(grads, [model.W, model.B]))
  if i % 20 == 0:
    print("Loss at step {:03d}: {:.3f}".format(i, loss(model, training_inputs, training_outputs)))

print("Final loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))
print("W = {}, B = {}".format(model.W.numpy(), model.B.numpy()))
# Note: Variables persist until the last reference to the python object is removed, and is the variable is deleted.
#
#########################################################################################################
                #   Object-based saving   #
'''
A tf.keras.Model includes a convenient save_weights method allowing you to easily create a checkpoint:
'''
model.save_weights('weights')
status = model.load_weights('weights')
# https://www.tensorflow.org/guide/checkpoint
x = tf.Variable(10.)
checkpoint = tf.train.Checkpoint(x=x)
x.assign(2.)   # Assign a new value to the variables and save.
checkpoint_path = './ckpt/'
checkpoint.save('./ckpt/')
x.assign(11.)  # Change the variable after saving.

# Restore values from the checkpoint
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))
print(x)  # => 2.0


# To save and load models, tf.train.Checkpoint stores the internal state of objects, without requiring hidden variables.
# To record the state of a model, an optimizer, and a global step, pass them to a tf.train.Checkpoint:
model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(16,[3,3], activation='relu'),
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(10)
])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
checkpoint_dir = 'path/to/model_dir'
if not os.path.exists(checkpoint_dir):
  os.makedirs(checkpoint_dir)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
root = tf.train.Checkpoint(optimizer=optimizer,
                           model=model)

root.save(checkpoint_prefix)
root.restore(tf.train.latest_checkpoint(checkpoint_dir))





############################################################################################################
                #   Object-oriented metrics   #
'''
tf.keras.metrics are stored as objects. Update a metric by passing the new data to the callable, 
and retrieve the result using the tf.keras.metrics.result method
'''
############################################################################################################
# m = tf.keras.metrics.Mean("loss") # name =loss
# m(0)
# m(5)
# print(m.result().numpy())  # => 2.5
# m([8, 9])
# print(m.result())  # => 5.5
# m.update_state([1, 3, 5]) # => 4.428571
# print(m.result())
############################################################################################################
                #   Advanced automatic differentiation topics   #
# '''Dynamic control flow:
# This has conditionals that depend on tensor values and it prints these values at runtime.
# data-dependent control flow, including if, for, while break, continue and return statements:
############################################################################################################
# e.g. as below shows:
def fizzbuzz(max_num):
  counter = tf.constant(0)
  max_num = tf.convert_to_tensor(max_num)
  for num in range(1, max_num.numpy()+1):
    num = tf.constant(num)
    if int(num % 3) == 0 and int(num % 5) == 0:
      print('FizzBuzz')
    elif int(num % 3) == 0:
      print('Fizz')
    elif int(num % 5) == 0:
      print('Buzz')
    else:
      print(num.numpy())
    counter += 1
fizzbuzz(15)
# ###########
#tf.GradientTape can also be used in dynamic models.
'''the example below can not be runned'''
def line_search_step(fn, init_x, rate=1.0):
  with tf.GradientTape() as tape:
    # Variables are automatically tracked.
    # But to calculate a gradient from a tensor, you must `watch` it.
    tape.watch(init_x) #Ensures that tensor is being traced by this tape.
    value = fn(init_x)
  grad = tape.gradient(value, init_x)
  grad_norm = tf.reduce_sum(grad * grad)
  init_value = value
  while value > init_value - rate * grad_norm:
    x = init_x - rate * grad
    value = fn(x)
    rate /= 2.0
  return x, value
############################################################################################################
                #   Custom gradients   #
############################################################################################################
# tf.custom_gradient: This !decorator! allows fine grained control over the gradients of a sequence for operations.

# @tf.custom_gradient
# def clip_gradient_by_norm(x, norm):
#   y = tf.identity(x)
#   def grad_fn(dresult):
#     return [tf.clip_by_norm(dresult, norm), None]
#   return y, grad_fn
#
# # Custom gradients are commonly used to provide a numerically stable gradient for a sequence of operations:
# def log1pexp(x):
#   return tf.math.log(1 + tf.exp(x))
#
# @tf.custom_gradient
# def log1pexp_custom(x):
#   e = tf.exp(x)
#   def grad(dy):
#     return dy * (1 - 1 / (1 + e))
#   return tf.math.log(1 + e), grad
#
# def grad_log1pexp(x):
#   with tf.GradientTape() as tape:
#     tape.watch(x)
#     # value = log1pexp(x)
#     value = log1pexp_custom(x)
#   return tape.gradient(value, x)
# # The gradient computation works fine at x = 0. for both
# print(grad_log1pexp(tf.constant(0.)).numpy())
# # However, x = 100 fails because of numerical instability. for log1pexp
# print(grad_log1pexp(tf.constant(100.)).numpy())
############################################################################################################
                #   Performance   #
'''Computation is automatically offloaded to GPUs during eager execution. 
If you want control over where a computation runs you can enclose it in a tf.device('/gpu:0') block (or the CPU equivalent):'''
############################################################################################################

def measure(x, steps):
  # TensorFlow initializes a GPU the first time it's used, exclude from timing.
  tf.matmul(x, x)
  start = time.time()
  for i in range(steps):
    x = tf.matmul(x, x)
  # tf.matmul can return before completing the matrix multiplication
  # (e.g., can return after enqueing the operation on a CUDA stream).
  # The x.numpy() call below will ensure that all enqueued operations
  # have completed (and will also copy the result to host memory,
  # so we're including a little more than just the matmul operation
  # time).
  _ = x.numpy()
  end = time.time()
  return end - start

shape = (1000, 1000)
steps = 200
print("Time to multiply a {} matrix by itself {} times:".format(shape, steps))

# Run on CPU:
with tf.device("/cpu:0"):
  print("CPU: {} secs".format(measure(tf.random.normal(shape), steps)))

# Run on GPU, if available:
if tf.config.experimental.list_physical_devices("GPU"):
  with tf.device("/gpu:0"):
    print("GPU: {} secs".format(measure(tf.random.normal(shape), steps)))
else:
  print("GPU: not found")



# A tf.Tensor object can be copied to a different device to execute its operations:
if tf.config.experimental.list_physical_devices("GPU"):
  x = tf.random.normal([10, 10])

  x_gpu0 = x.gpu()
  x_cpu = x.cpu()

  _ = tf.matmul(x_cpu, x_cpu)    # Runs on CPU
  _ = tf.matmul(x_gpu0, x_gpu0)  # Runs on GPU:0
else:
  print("GPU: not found")








############################################################################################################
                #   quickstart for experts   #
'''Key Point: Any Python side-effects (appending to a list, printing with print, etc) will only happen once, when func is traced. 
To have side-effects executed into your tf.function they need to be written as TF ops.
Caution: Passing python scalars or lists as arguments to tf.function will always build a new graph. 
To avoid this, pass numeric arguments as Tensors whenever possible:'''
'@tf.function: reference:https://www.tensorflow.org/api_docs/python/tf/function '
############################################################################################################
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
# Add a channels dimensionself.layer1
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


def layer(x):
  x = x * 2
  return x

class MyModel(tf.keras.Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu')
    self.flatten = tf.keras.layers.Flatten()
    self.d1 = tf.keras.layers.Dense(128, activation='relu')
    self.d2 = tf.keras.layers.Dense(10)
    self.d3 = tf.keras.layers.Dense(10)

    # self.layer1=DarknetConv2D_BN_Leaky(128//2, (1,1)),

  def call(self, x):
    # x = tf.keras.layers.Conv2D(128//2, (1,1))(x)
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)

    # print('call')
    x1 = self.d2(x)
    x2=x1+2 # it does every loop.
    # tf.print(x2.shape)
    # x2 = self.d3(x2) # produce error, Input 0 of layer dense_1 is incompatible with the layer:
    # expected axis -1 of input shape to have value 128 but received input with shape [32, 10], because on the first call
    # the self.de layers has been instantized

    x2=self.d3 (x2)
    x3=layer(x1)
    return x1,x3

# Create an instance of the model
model = MyModel()

def loss(model, inputs, targets):
  error = model(inputs) - targets
  return tf.reduce_mean(tf.square(error))

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
# '@tf.function'+'with tf.GradientTape() as tape:' makes a graph in tensorflow, no longer a eager execution
# it is graph execution
@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions1, predictions2= model(images, training=True)
    tf.print(predictions1.shape, predictions1[0])
    tf.print(predictions2.shape,predictions2[0])
    predictions=predictions1+predictions2
    # when model.trainable_variables changes. the loss would also changed with it
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)


with tf.device("/cpu:0"):
    EPOCHS = 5
    for epoch in range(EPOCHS):
      # Reset the metrics at the start of the next epoch
      train_loss.reset_states()
      train_accuracy.reset_states()
      test_loss.reset_states()
      test_accuracy.reset_states()

      i=0
      for images, labels in train_ds:
        print('batch number',i)
        train_step(images, labels)
        i=i+1

      for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

      print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result()}, '
        f'Accuracy: {train_accuracy.result() * 100}, '
        f'Test Loss: {test_loss.result()}, '
        f'Test Accuracy: {test_accuracy.result() * 100}'
      )

############################################################################################################
                #   tf.keras.Model   #
'''There are two ways to instantiate a Model:
1 - With the "Functional API", where you start from Input, you chain layer calls to specify the model's forward pass, 
and finally you create your model from inputs and outputs.
2 - By subclassing the Model class: in that case, you should define your layers in __init__ and you should implement the model's forward pass in call.
'''
############################################################################################################
# ## 1 - With the "Functional API" example:
# inputs = tf.keras.Input(shape=(3,))
# x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
# outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
# model = tf.keras.Model(inputs=inputs, outputs=outputs)
#
# # 2 - By subclassing the Model class:
# class MyModel(tf.keras.Model):
#
#   def __init__(self):
#     super(MyModel, self).__init__()
#     self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
#     self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
#
#   def call(self, inputs):
#     x = self.dense1(inputs)
#     return self.dense2(x)
#
# model = MyModel()
#
# # If you subclass Model, you can optionally have a training argument (boolean) in call,
# # which you can use to specify a different behavior in training and inference:
# class MyModel(tf.keras.Model):
#
#   def __init__(self):
#     super(MyModel, self).__init__()
#     self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
#     self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
#     self.dropout = tf.keras.layers.Dropout(0.5)
#
#   def call(self, inputs, training=False):
#     x = self.dense1(inputs)
#     if training:
#       x = self.dropout(x, training=training)
#     return self.dense2(x)
#
# model = MyModel()
# # Once the model is created, you can config the model with losses and metrics with model.compile(),
# # train the model with model.fit(), or use the model to do prediction with model.predict().









