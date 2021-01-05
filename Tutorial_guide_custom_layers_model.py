import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import time
############################################################################################################
            ###          Putting it all together: an end-to-end example               ###
'''
Here's what you've learned so far:
1. A Layer encapsulate a state (created in __init__() or build()) and some computation (defined in call()).
2. Layers can be recursively nested to create new, bigger computation blocks.
3. Layers can create and track losses (typically regularization losses) as well as metrics, via add_loss() and add_metric()
4. The outer container, the thing you want to train, is a Model. A Model is just like a Layer, but with added training and serialization utilities.
Let's put all of these things together into an end-to-end example: we're going to implement a Variational AutoEncoder (VAE). We'll train it on MNIST digits.

Our VAE will be a subclass of Model, built as a nested composition of layers that subclass Layer. It will feature a regularization loss (KL divergence).
'''
############################################################################################################
class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(tf.keras.layers.Layer):
    """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""

    def __init__(self, latent_dim=32, intermediate_dim=64, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.dense_proj = tf.keras.layers.Dense(intermediate_dim, activation="relu")
        self.dense_mean = tf.keras.layers.Dense(latent_dim)
        self.dense_log_var = tf.keras.layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.dense_proj(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(tf.keras.layers.Layer):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(self, original_dim, intermediate_dim=64, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense_proj = tf.keras.layers.Dense(intermediate_dim, activation="relu")
        self.dense_output = tf.keras.layers.Dense(original_dim, activation="sigmoid")

    def call(self, inputs):
        x = self.dense_proj(inputs)
        return self.dense_output(x)


class VariationalAutoEncoder(tf.keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(
        self,
        original_dim,
        intermediate_dim=64,
        latent_dim=32,
        name="autoencoder",
        **kwargs
    ):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim, intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss.
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        self.add_loss(kl_loss)
        return reconstructed
########################    dataset pipeline  ###############################
(x_train, x_label), _ = tf.keras.datasets.mnist.load_data()
print(x_train.shape)
print(x_label.shape)
x_train = x_train.reshape(60000, 784).astype("float32") / 255
print(x_train.shape)
train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
##############################################################################
#1 Let's write a simple training loop on MNIST:
original_dim = 784
vae = VariationalAutoEncoder(original_dim, 64, 32)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
mse_loss_fn = tf.keras.losses.MeanSquaredError()

loss_metric = tf.keras.metrics.Mean()

# Iterate over epochs.
epochs = 2
for epoch in range(epochs):
    print("Start of epoch %d" % (epoch,))

    # Iterate over the batches of the dataset.
    for step, x_batch_train in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            reconstructed = vae(x_batch_train)
            # Compute reconstruction loss
            loss = mse_loss_fn(x_batch_train, reconstructed)
            loss += sum(vae.losses)  # Add KLD regularization loss

        grads = tape.gradient(loss, vae.trainable_weights)
        optimizer.apply_gradients(zip(grads, vae.trainable_weights))

        loss_metric(loss)

        if step % 100 == 0:
            print("step %d: mean loss = %.4f" % (step, loss_metric.result()))

#
# #2 Note that since the VAE is subclassing Model, it features built-in training loops. So you could also have trained it like this:
# vae = VariationalAutoEncoder(784, 64, 32)
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
#
# vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError(), metrics=tf.keras.metrics.Mean())
# vae.fit(x_train, x_train, epochs=2, batch_size=64)
print('----------------'*5,'Next part','----------------'*5)

## Subclass model can have subclass model usually be layers
class ResNetBlock(tf.keras.Model):
# class ResNetBlock(tf.keras.layers.Layer):
  def __init__(self, kernel_size, filters):
    super(ResNetBlock, self).__init__(name='')
    filters1, filters2, filters3 = filters

    self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
    self.bn2a = tf.keras.layers.BatchNormalization()

    self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
    self.bn2b = tf.keras.layers.BatchNormalization()

    self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
    self.bn2c = tf.keras.layers.BatchNormalization()

  def call(self, input_tensor, training=False):
    x = self.conv2a(input_tensor)
    x = self.bn2a(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2b(x)
    x = self.bn2b(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2c(x)
    x = self.bn2c(x, training=training)

    x += input_tensor
    x=tf.nn.relu(x)
    return x
block = ResNetBlock(1, [1, 2, 3])
_ = block(tf.zeros([1, 2, 3, 3]))
# print(block.layers)
block.summary()
print("weights:", len(block.weights))

class MLPBlock(tf.keras.Model):
    def __init__(self):
        super(MLPBlock, self).__init__()
        self.layer_1 = ResNetBlock(1, [1, 2, 3])


    def call(self, inputs):
        x = self.layer_1(inputs)
        return x

AA=MLPBlock()
y = AA(tf.ones(shape=(10,64, 64,3)))
AA.summary()
print("weights:", len(AA.weights))
print('----------------'*5,'Next part','----------------'*5)
############################################################################################################
            ###   Another Way:   Beyond object-oriented development: using the Functional API               ###
'''
Was this example too much object-oriented development for you? 
You can also build models using the Functional API. 
Importantly, choosing one style or another does not prevent you from leveraging components written in the other style: you can always mix-and-match.

For instance, the Functional API example below reuses the same Sampling layer we defined in the example above:
'''
############################################################################################################
# original_dim = 784
# intermediate_dim = 64
# latent_dim = 32
# ########################    dataset pipeline  ###############################
# (x_train, _), _ = tf.keras.datasets.mnist.load_data()
# x_train = x_train.reshape(60000, 784).astype("float32") / 255
#
# train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
# train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
# # ##############################################################################
# class Sampling(tf.keras.layers.Layer):
#     """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
#
#     def call(self, inputs):
#         z_mean, z_log_var = inputs
#         batch = tf.shape(z_mean)[0]
#         dim = tf.shape(z_mean)[1]
#         epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
#         return z_mean + tf.exp(0.5 * z_log_var) * epsilon
#
# # Define encoder model.
# original_inputs = tf.keras.Input(shape=(original_dim,), name="encoder_input")
# x = tf.keras.layers.Dense(intermediate_dim, activation="relu")(original_inputs)
# z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(x)
# z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x)
# z = Sampling()((z_mean, z_log_var))
# encoder = tf.keras.Model(inputs=original_inputs, outputs=z, name="encoder")
#
# # Define decoder model.
# latent_inputs = tf.keras.Input(shape=(latent_dim,), name="z_sampling")
# x = tf.keras.layers.Dense(intermediate_dim, activation="relu")(latent_inputs)
# outputs = tf.keras.layers.Dense(original_dim, activation="sigmoid")(x)
# decoder = tf.keras.Model(inputs=latent_inputs, outputs=outputs, name="decoder")
#
# # Define VAE model.
# outputs = decoder(z)
# vae = tf.keras.Model(inputs=original_inputs, outputs=outputs, name="vae")
#
# # Add KL divergence regularization loss.
# kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
# vae.add_loss(kl_loss)
#
# # Train.
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
# vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
# vae.fit(x_train, x_train, epochs=3, batch_size=64)



############################################################################################################
            ###          Beyond object-oriented development: the Functional API               ###
############################################################################################################
# One of the central abstraction in Keras is the Layer class.
# A layer encapsulates both a state (the layer's "weights") and a transformation from inputs to outputs (a "call", the layer's forward pass).
# Here's a densely-connected layer. It has a state: the variables w and b.

class Linear(tf.keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype="float32"),
            trainable=True,
        )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
        )

    def call(self, inputs, **kwargs):
        result=tf.matmul(inputs, self.w) + self.b
        return result
#You would use a layer by calling it on some tensor input(s), much like a Python function:
x = tf.ones((2, 2))
linear_layer = Linear(4, 2)
y = linear_layer(x)
print(y)
assert linear_layer.weights == [linear_layer.w, linear_layer.b]
print('assert done')

# Note you also have access to a quicker shortcut for adding weight to a layer:
# the add_weight() method:Adds a new variable to the layer.
class Linear1(tf.keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(Linear1, self).__init__()
        self.w = self.add_weight(
            shape=(input_dim, units), initializer="random_normal", trainable=True
        )
        self.b = self.add_weight(shape=(units,), initializer="zeros", trainable=True)

    def call(self, inputs,**kwargs):
        return tf.matmul(inputs, self.w) + self.b
x = tf.ones((2, 2))
linear_layer = Linear1(4, 2)
y = linear_layer(x)
print(y)

# Layers can have non-trainable weights:
# Besides trainable weights, you can add non-trainable weights to a layer as well.
# Such weights are meant not to be taken into account during backpropagation, when you are training the layer.

class ComputeSum(tf.keras.layers.Layer):
    def __init__(self, input_dim):
        super(ComputeSum, self).__init__()
        self.total = tf.Variable(initial_value=tf.zeros((input_dim,)), trainable=False)

    def call(self, inputs, **kwargs):
        self.total.assign_add(tf.reduce_sum(inputs, axis=0))
        return self.total

x = tf.ones((2, 2))
my_sum = ComputeSum(2)
y = my_sum(x)
print(y.numpy())
y = my_sum(x)
print(y.numpy())
y = my_sum(x)
print(y.numpy())
# It's part of layer.weights, but it gets categorized as a non-trainable weight:
print("weights:", len(my_sum.weights))
print("non-trainable weights:", len(my_sum.non_trainable_weights))
# It's not included in the trainable weights:
print("trainable_weights:", my_sum.trainable_weights)
print("trainable_weights:", my_sum.non_trainable_weights)

# Best practice: deferring weight creation until the shape of the inputs is known:
# Linear layer below took an 'input_dim' argument that was used to compute the shape of the weights w and b in __init__():
class Linear_with(tf.keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(Linear_with, self).__init__()
        self.w = self.add_weight(
            shape=(input_dim, units), initializer="random_normal", trainable=True
        )
        self.b = self.add_weight(shape=(units,), initializer="zeros", trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

#In many cases, you may not know in advance the size of your inputs, and you would like to lazily create weights when that value becomes known,
# some time after instantiating the layer.
# In the Keras API, we recommend creating layer weights in the build(self, inputs_shape) method of your layer. Like this:

class Linear_with_weightUnknown(tf.keras.layers.Layer):
    def __init__(self, units=32):
        super(Linear_with_weightUnknown, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs): #signature of method: call(inputs, **kwargs)
        return tf.matmul(inputs, self.w) + self.b

# The __call__() method of your layer will automatically run build the first time it is called. You now have a layer that's lazy and thus easier to use:

# At instantiation, we don't know on what inputs this is going to get called
linear_layer = Linear_with_weightUnknown(32)
x = tf.ones((2, 2))
y = linear_layer(x)# The layer's weights are created dynamically the first time the layer is called
print(y)
# x = tf.ones((4, 4))
# y = linear_layer(x)# so second time produce error
# print(y)

print('----------------'*5,'Next part','----------------'*5)
############################################################################################################
            ###          Layers are recursively composable               ###
'''
If you assign a Layer instance as an attribute of another Layer, the outer layer will start tracking the weights of the inner layer.
We recommend creating such sublayers in the __init__() method 
(since the sublayers will typically have a build method, they will be built when the outer layer gets built).
'''
############################################################################################################
# Let's reuse the Linear class above
# with a `build` method that we defined above.


class MLPBlock(tf.keras.layers.Layer):
    def __init__(self):
        super(MLPBlock, self).__init__()
        self.linear_1 = Linear_with_weightUnknown(32)
        self.linear_2 = Linear_with_weightUnknown(32)
        self.linear_3 = Linear_with_weightUnknown(1)

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = tf.nn.relu(x)
        x = self.linear_2(x)
        x = tf.nn.relu(x)
        result=self.linear_3(x)
        return result


mlp = MLPBlock()
x=tf.ones(shape=(3, 64))
y = mlp(x)  # The first call to the `mlp` will create the weights
print("weights:", len(mlp.weights))
print("trainable weights:", len(mlp.trainable_weights))


print('----------------'*5,'Next part','----------------'*5)
#The add_loss() method:
# When writing the call() method of a layer, you can create loss tensors that you will want to use later,
# when writing your training loop. This is doable by calling 'self.add_loss(value):Add loss tensor(s), potentially dependent on layer inputs.'

# A layer that creates an activity regularization loss
class ActivityRegularizationLayer(tf.keras.layers.Layer):
    def __init__(self, rate=1e-2):
        super(ActivityRegularizationLayer, self).__init__()
        self.rate = rate

    def call(self, inputs):
        self.add_loss(self.rate * tf.reduce_sum(inputs))
        return inputs

class OuterLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(OuterLayer, self).__init__()
        self.activity_reg = ActivityRegularizationLayer(1)

    def call(self, inputs):
        return self.activity_reg(inputs)

layer = OuterLayer()
assert len(layer.losses) == 0  # No losses yet since the layer has never been called
print('assert done')
print(layer.losses)

_ = layer(tf.zeros(1, 1))
assert len(layer.losses) == 1  # We created one loss value
print('assert done')
print(layer.losses)

# `layer.losses` gets reset at the start of each __call__
_ = layer(tf.zeros(1, 1))
assert len(layer.losses) == 1  # This is the loss created during the call above
print('assert done')
print(layer.losses)

# In addition, the loss property also contains regularization losses created for the weights of any inner layer:
class OuterLayerWithKernelRegularizer(tf.keras.layers.Layer):
    def __init__(self):
        super(OuterLayerWithKernelRegularizer, self).__init__()
        self.dense = tf.keras.layers.Dense(
            32, kernel_regularizer=tf.keras.regularizers.l2(1e-3)
        )

    def call(self, inputs):
        return self.dense(inputs)


layer = OuterLayerWithKernelRegularizer()
_ = layer(tf.zeros((1, 1)))

# This is `1e-3 * sum(layer.dense.kernel ** 2)`,
# created by the `kernel_regularizer` above.
print(layer.losses)
# These losses are meant to be taken into account when writing training loops, like this:


# # Instantiate an optimizer.
# optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#
# # Iterate over the batches of a dataset.
# for x_batch_train, y_batch_train in train_dataset:
#   with tf.GradientTape() as tape:
#     logits = layer(x_batch_train)  # Logits for this minibatch
#     # Loss value for this minibatch
#     loss_value = loss_fn(y_batch_train, logits)
#     # Add extra losses created during this forward pass:
#     loss_value += sum(model.losses)
#
#   grads = tape.gradient(loss_value, model.trainable_weights)
#   optimizer.apply_gradients(zip(grads, model.trainable_weights))
# #For a detailed guide about writing training loops, see the guide to writing a training loop from scratch
# # https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch/.

inputs = tf.keras.Input(shape=(3,))
outputs = ActivityRegularizationLayer()(inputs)
model = tf.keras.Model(inputs, outputs)

# If there is a loss passed in `compile`, the regularization losses get added to it
model.compile(optimizer="adam", loss="mse")
model.fit(np.random.random((2, 3)), np.random.random((2, 3)))

# It's also possible not to pass any loss in `compile`,
# since the model already has a loss to minimize, via the `add_loss`
# call during the forward pass!
model.compile(optimizer="adam")
model.fit(np.random.random((2, 3)), np.random.random((2, 3)))

print('----------------'*5,'Next part','----------------'*5)
#The  add_metric() method:
# Similarly to add_loss(), layers also have an add_metric() method for tracking the moving average of a quantity during training.

# Consider the following layer: a "logistic endpoint" layer.
# It takes as inputs predictions & targets, it computes a loss which it tracks via add_loss(),
# and it computes an accuracy scalar, which it tracks via add_metric().
class LogisticEndpoint(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super(LogisticEndpoint, self).__init__(name=name)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.accuracy_fn = tf.keras.metrics.BinaryAccuracy()
        self.accuracy_fn2= tf.keras.metrics.BinaryCrossentropy()

    def call(self, targets, logits, sample_weights=None):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        loss = self.loss_fn(targets, logits, sample_weights)
        self.add_loss(loss)

        # Log accuracy as a metric and add it
        # to the layer using `self.add_metric()`.
        acc1 = self.accuracy_fn(targets, logits, sample_weights)
        self.add_metric(acc1, name="accuracy")
        acc2 = self.accuracy_fn2(targets, logits, sample_weights)
        self.add_metric(acc2, name="accuracy2")

        # Return the inference-time prediction tensor (for `.predict()`).
        return tf.nn.softmax(logits)

# Metrics tracked in this way are accessible via layer.metrics:
layer = LogisticEndpoint()

targets = tf.ones((2, 2))
logits = tf.ones((2, 2))
y = layer(targets, logits)

print("layer.metrics:", layer.metrics)
print("current accuracy value:", layer.metrics[0].result(),layer.metrics[1].result())

# Just like for add_loss(), these metrics are tracked by fit():
inputs = tf.keras.Input(shape=(3,), name="inputs")
targets = tf.keras.Input(shape=(10,), name="targets")
logits = tf.keras.layers.Dense(10)(inputs)
predictions = LogisticEndpoint(name="predictions")(logits, targets)

model = tf.keras.Model(inputs=[inputs, targets], outputs=predictions)
model.compile(optimizer="adam")

data = {
    "inputs": np.random.random((3, 3)),
    "targets": np.random.random((3, 10)),
}
model.fit(data)

print('----------------'*5,'Next part','----------------'*5)
############################################################################################################
            ###          You can optionally enable serialization on your layers               ###
'''
If you need your custom layers to be serializable as part of a Functional model, you can optionally implement a get_config() method
'''
############################################################################################################
class Linear3(tf.keras.layers.Layer):
    def __init__(self, units=32):
        super(Linear3, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        return {"units": self.units}
# Now you can recreate the layer from its config:
layer = Linear3(64)
config = layer.get_config()
print(config)
new_layer = Linear3.from_config(config)
#Note that the __init__() method of the base Layer class takes some keyword arguments, in particular a name and a dtype.
# It's good practice to pass these arguments to the parent class in __init__() and to include them in the layer config:
class Linear4(tf.keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super(Linear4, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        config = super(Linear4, self).get_config()
        config.update({"units": self.units})
        return config


layer = Linear4(64)
config = layer.get_config()
print(config)
new_layer = Linear4.from_config(config)
#If you need more flexibility when deserializing the layer from its config,
# you can also override the from_config() class method. This is the base implementation of from_config():

print('----------------'*5,'Next part','----------------'*5)

############################################################################################################
            ###          Privileged training argument in the call() method               ###
'''
Some layers, in particular the BatchNormalization layer and the Dropout layer, have different behaviors during training and inference. 
For such layers, it is standard practice to expose a training (boolean) argument in the call() method.
'''
############################################################################################################
class CustomDropout(tf.keras.layers.Layer):
    def __init__(self, rate, **kwargs):
        super(CustomDropout, self).__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, training=None):
        if training:
            return tf.nn.dropout(inputs, rate=self.rate)
        return inputs





















