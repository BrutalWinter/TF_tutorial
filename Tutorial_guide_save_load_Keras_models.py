import numpy as np
import tensorflow as tf
import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
############################################################################################################
            ###   Save and load Keras models       ###
print('''
A Keras model consists of multiple components:
    1.An architecture, or configuration, which specifies what layers the model contain, and how they're connected.
    2.A set of weights values (the "state of the model").
    3.An optimizer (defined by compiling the model).
    4.A set of losses and metrics (defined by compiling the model or calling add_loss() or add_metric()).

The Keras API makes it possible to save all of these pieces to disk at once, or to only selectively save some of them:
    1.Saving everything into a single archive in the TensorFlow SavedModel format (or in the older Keras H5 format). This is the standard practice. 
    2.Saving the architecture / configuration only, typically as a JSON file.
    3.Saving the weights values only. This is generally used when training the model.
Let's take a look at each of these options: when would you use one or the other? How do they work?''')
############################################################################################################
print('##########      Whole-model saving & loading briefly     ##########\n'*2)
############################################################################################################
###   Whole-model saving & loading      ###
'''
You can save an entire model to a single artifact. It will include:
        1.The model's architecture/config
        2.The model's weight values (which were learned during training)
        3.The model's compilation information (if compile()) was called
        4.The optimizer and its state, if any (this enables you to restart training where you left)
APIs:
    1.model.save() or tf.keras.models.save_model()
    2.tf.keras.models.load_model()

There are two formats you can use to save an entire model to disk: 
    1.the TensorFlow SavedModel format. The recommended format is SavedModel. It is the default when you use model.save().
    2.and the older Keras H5 format. 

You can switch to the H5 format by:
    1.Passing [save_format='h5'] to [save()].
    2.Passing a filename that ends in [.h5] or [.keras] to [save()].'''
############################################################################################################
def get_model():
    # Create a simple model.
    inputs = keras.Input(shape=(32,))
    outputs = keras.layers.Dense(1)(inputs)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

model = get_model()

# Train the model.
test_input = np.random.random((128, 32))
test_target = np.random.random((128, 1))
model.fit(test_input, test_target)

# Calling `save('my_model')` creates a SavedModel folder `my_model`.
model.save('model_save/Keras_model1')

# It can be used to reconstruct the model identically.
reconstructed_model = keras.models.load_model('model_save/Keras_model1')

# Let's check:, !!!!!!! If there is a difference the exe will stop here  !!!!!!!!!!!!'
np.testing.assert_allclose(
    model.predict(test_input), reconstructed_model.predict(test_input)
)

# The reconstructed model is already compiled and has retained the optimizer state, so training can resume:
reconstructed_model.fit(test_input, test_target)
'''
Calling model.save(\'my_model\') creates a folder named my_model, containing the following:
1.The model architecture, and training configuration (including the optimizer, losses, and metrics) are stored in saved_model.pb. 
2.The weights are saved in the variables/ directory.
'''





############################################################################################################
print('##########      How SavedModel handles custom objects       ##########\n'*2)
###   How SavedModel handles custom objects      ###
'''In the absence of the model/layer config, the call function is used to create a model that exists like the original model 
which can be trained, evaluated, and used for inference.

Nevertheless, it is always a good practice to define the get_config and from_config methods when writing a custom model or layer class. 
This allows you to easily update the computation later if needed.'''

#Below is an example of what happens when loading custom layers from he SavedModel format without overwriting the config methods

class CustomModel(keras.Model):
    def __init__(self, hidden_units):
        super(CustomModel, self).__init__()
        self.hidden_units = hidden_units
        self.dense_layers = [keras.layers.Dense(u) for u in hidden_units]

    def call(self, inputs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        return x

    def get_config(self):
        return {"hidden_units": self.hidden_units}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

model = CustomModel([16, 16, 10])
# Build the model by calling it
input_arr = tf.random.uniform((1, 5))
outputs = model(input_arr)
model.save("model_save/Keras_model2")

# Option 1: Load with the custom_object argument.
loaded_1 = keras.models.load_model(
    "model_save/Keras_model2", custom_objects={"CustomModel": CustomModel}
)

# Delete the custom-defined model class to ensure that the loader does not have access to it.
del CustomModel
# Option 2: Load without the CustomModel class.
loaded_2 = keras.models.load_model("model_save/Keras_model2")
np.testing.assert_allclose(loaded_1(input_arr), outputs)#('!!!!!!! If there is a difference the exe will stop here  !!!!!!!!!!!!')
np.testing.assert_allclose(loaded_2(input_arr), outputs)#('!!!!!!! If there is a difference the exe will stop here  !!!!!!!!!!!!')


print("Original model:", model)
print("Model Loaded with custom objects:", loaded_1)
print("Model loaded without the custom object class:", loaded_2)
'''
1.The first loaded model is loaded using the config and CustomModel class. 
2.The second model is loaded by dynamically creating the model class that acts like the original model.
'''





############################################################################################################
print('##########      Keras H5 format       ##########\n'*2)
###   Keras H5 format    ###
print('''Keras also supports saving a single HDF5 file containing the models architecture, weights values, and compile() information. 
It is a light-weight alternative to SavedModel.
Compared to the SavedModel format, there are two things that don't get included in the H5 file:
1.External losses & metrics added via model.add_loss() & model.add_metric() are not saved.
2.The computation graph of custom objects such as custom layers is not included in the saved file.
''')
# model = get_model()
#
# # Train the model.
# test_input = np.random.random((128, 32))
# test_target = np.random.random((128, 1))
# model.fit(test_input, test_target)
#
# # Calling `save('my_model.h5')` creates a h5 file `my_model.h5`.
# model.save("model_save/my_h5_model.h5")
#
# # It can be used to reconstruct the model identically.
# reconstructed_model = keras.models.load_model("model_save/my_h5_model.h5")
#
# # Let's check:
# np.testing.assert_allclose(
#     model.predict(test_input), reconstructed_model.predict(test_input)
# )
#
# # The reconstructed model is already compiled and has retained the optimizer
# # state, so training can resume:
# reconstructed_model.fit(test_input, test_target)



############################################################################################################
print('##########      Saving the architecture not. it is not working for subclassed models      ##########\n'*2)
###   Saving the architecture    ###
'''
#The model's configuration (or architecture) specifies what layers the model contains, and how these layers are connected*.
# If you have the configuration of a model, then the model can be created with a freshly initialized state for the weights and no compilation information.
#* Note this only applies to models defined using the functional or Sequential APIs not subclassed models.'''

# Configuration of a Sequential model or Functional API model:
# These types of models are explicit graphs of layers: their configuration is always available in a structured form.

#       APIs:
#            1.get_config() and from_config()
#            2.tf.keras.models.model_to_json() and tf.keras.models.model_from_json()''')


# Calling config = model.get_config() will return a Python dict containing the configuration of the model.
# The same model can then be reconstructed via Sequential.from_config(config) (for a Sequential model) or Model.from_config(config) (for a Functional API model).
# 1-1 Layer example:
layer = keras.layers.Dense(3, activation="relu")
layer_config = layer.get_config()
new_layer = keras.layers.Dense.from_config(layer_config)

# 1-2 Sequential model example:
model = keras.Sequential([
    keras.Input((32,)),
    keras.layers.Dense(1)])

config = model.get_config()
new_model = keras.Sequential.from_config(config)

# 2-2 Functional model example:
inputs = keras.Input((32,))
outputs = keras.layers.Dense(1)(inputs)
model2 = keras.Model(inputs, outputs)

config = model2.get_config()
new_model2 = keras.Model.from_config(config)

# to_json() and tf.keras.models.model_from_json()
# This is similar to get_config / from_config, except it turns the model into a JSON string,
# which can then be loaded without the original model class. It is also specific to models, it isn't meant for layers.
# Example:
model = keras.Sequential([keras.Input((32,)), keras.layers.Dense(1)])
json_config = model.to_json()
new_model = keras.models.model_from_json(json_config)






############################################################################################################
print('##########      Custom objects       ##########\n'*5)
###   Custom objects    ###
'''
Models and layers:
    1.The architecture of subclassed models and layers are defined in the methods __init__ and call. They are considered 
Python bytecode, which cannot be serialized into a JSON-compatible config. -- you could try serializing the bytecode (e.g. via pickle), 
but it's completely unsafe and means your model cannot be loaded on a different system.

    2.In order to save/load a model with custom-defined layers, or a subclassed model, you should overwrite the get_config and optionally from_config methods. 
Additionally, you should use register the custom object so that Keras is aware of it.

Custom functions:
    1.Custom-defined functions (e.g. activation loss or initialization) do not need a get_config method. 
    2.The function name is sufficient for loading as long as it is registered as a custom object.
'''
############################################################################################################
# Loading the TensorFlow graph only
    #It's possible to load the TensorFlow graph generated by the Keras. If you do so, you won't need to provide any custom_objects. You can do so like this:

# model.save("model_save/my_model")
# tensorflow_graph = tf.saved_model.load("model_save/my_model")
# x = np.random.uniform(size=(4, 32)).astype(np.float32)
# predicted = tensorflow_graph(x).numpy()
'''
Note that above method has several drawbacks:
    1.For traceability reasons, you should always have access to the custom objects that were used. You wouldn't want to put 
in production a model that you cannot re-create.
    2.The object returned by 'tf.saved_model.load' isn't a Keras model. So it's not as easy to use. For example, you won't have access to .predict() or .fit()
    
Even if its use is discouraged, it can help you if you're in a tight spot, for example:
if you lost the code of your custom objects or have issues loading the model with tf.keras.models.load_model().
You can find out more in   https://www.tensorflow.org/api_docs/python/tf/saved_model/load
'''
################ python @classmethod 的使用场合
class Data_test2(object):
    day=0
    month=0
    year=0
    def __init__(self,year=0,month=0,day=0):
        self.day=day
        self.month=month
        self.year=year

    @classmethod
    def get_date(cls,data_as_string):
        #这里第一个参数是cls， 表示调用当前的类名!!!!!!
        year,month,day=map(int,data_as_string.split('-'))
        date1=cls(year,month,day)
        #返回的是一个初始化后的类
        return date1

    def out_date(self):
        print("year :")
        print(self.year)
        print("month :")
        print(self.month)
        print("day :")
        print(self.day)

r=Data_test2.get_date("2016-8-6")
r.out_date()
# 在Date_test类里面创建一个成员函数， 前面用了@classmethod装饰。 它的作用就是有点像静态类，比静态类不一样的就是它可以传进来一个当前类作为第一个参数。
################
'''
Defining the config methods, Specifications:
    1. get_config should return a JSON-serializable dictionary in order to be compatible with the Keras architecture- and model-saving APIs.
    2.from_config(config) (@classmethod) should return a new layer or model object that is created from the config. The default implementation returns cls(**config).
'''

class CustomLayer1(keras.layers.Layer):
    def __init__(self, a):
        self.var = tf.Variable(a, name="var_a")

    def call(self, inputs, training=False):
        if training:
            return inputs * self.var
        else:
            return inputs

    def get_config(self):
        return {"a": self.var.numpy()}

    # There's actually no need to define `from_config` here, since returning
    # `cls(**config)` is the default behavior.
    @classmethod
    def from_config(cls, config):
        return cls(**config)

layer = CustomLayer1(5)
layer.var.assign(2)

serialized_layer = keras.layers.serialize(layer)
new_layer = keras.layers.deserialize(serialized_layer, custom_objects={"CustomLayer1": CustomLayer1})

# Registering the custom object
# Keras keeps a note of which class generated the config. From the example above, tf.keras.layers.serialize generates a serialized form of the custom layer:

# Custom layer and function example:
class CustomLayer(keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super(CustomLayer, self).__init__(**kwargs)
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
        config = super(CustomLayer, self).get_config()
        config.update({"units": self.units})
        return config


def custom_activation(x):
    return tf.nn.tanh(x) ** 2


# Make a model with the CustomLayer and custom_activation
inputs = keras.Input((32,))
x = CustomLayer(32)(inputs)
outputs = keras.layers.Activation(custom_activation)(x)
model = keras.Model(inputs, outputs)

# Retrieve the config
config = model.get_config()

# At loading time, register the custom objects with a `custom_object_scope`:
custom_objects = {"CustomLayer": CustomLayer, "custom_activation": custom_activation}
with keras.utils.custom_object_scope(custom_objects):
    new_model = keras.Model.from_config(config)

















