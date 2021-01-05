import tensorflow as tf
import os
import numpy as np

############################## Example:   ##############################
# define layers and model:
class layer(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super(layer, self).__init__(**kwargs)
    self.layer1 = tf.keras.layers.Multiply()

  def call(self, inputs_set):
    x = self.layer1(inputs_set)
    return x
# layer1=layer()
# B1=np.arange(10).reshape(10, 1)
# B2=np.arange(10,20).reshape(10, 1)
# print(B1)
# print(B2)
# A=layer1([B1,B2])
# print(A)

class MyModel(tf.keras.Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu')
    self.flatten = tf.keras.layers.Flatten()
    self.d1 = tf.keras.layers.Dense(128, activation='relu')
    self.d2 = tf.keras.layers.Dense(10)
    self.d3 = tf.keras.layers.Dense(10)
    self.layer = layer()

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    x2 = self.d2(x)
    x3 = self.d3(x)
    x=self.layer([x2,x3])
    # x=x2+x3
    return x

def loss_object(Crossentropy_fun, inputs, targets):
  input = inputs+2
  loss = Crossentropy_fun(targets, input)
  return loss

# '@tf.function'+'with tf.GradientTape() as tape:' makes a graph in tensorflow, no longer a eager execution, it is graph execution
@tf.function
def train_step(model, loss_object, Crossentropy_fun, optimizer, images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different behavior during training versus inference (e.g. Dropout).
    predictions= model(images, training=True)
    loss = loss_object(Crossentropy_fun, predictions, labels) # when model.trainable_variables changes. the loss would also changed with it

  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  # print('loss',loss)
  return loss


@tf.function
def test_step(model, loss_object, Crossentropy_fun, images, labels):
  # training=False is only needed if there are layers with different behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  loss = loss_object(Crossentropy_fun, predictions, labels)
  # print('loss', loss)
  return loss

#########################################################
'''Using  model.save_weights(filepath=checkpoint_prefix, overwrite=True, save_format=None, options=None)
which is exactly the tf.train.checkpoint without using checkpoint manager'''
# if __name__ == '__main__':
#   # prepare data:
#   mnist = tf.keras.datasets.mnist
#   (x_train, y_train), (x_test, y_test) = mnist.load_data()
#   x_train, x_test = x_train / 255.0, x_test / 255.0
#   print(x_train.shape,y_train.shape)
#   print(x_test.shape,y_test.shape)
#   # Add a channels dimensionself.layer1
#   x_train = x_train[..., tf.newaxis].astype("float32")
#   x_test = x_test[..., tf.newaxis].astype("float32")
#
#   train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(60)
#   test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(10)
#   print(train_ds)
#   print(test_ds)
#   # for step, data in enumerate(train_ds):
#   #   print('The step {} data is {}'.format(step,data[1]))
#   #   if step==50:
#   #     break
#
#   # Create an instance of the model
#   model = MyModel()
#   # loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#   optimizer = tf.keras.optimizers.Adam()
#   Crossentropy_fun = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#   #########################################3
#   checkpoint_directory = "/home/brutal/PycharmProjects/Project-YOLOv3/model_save/my_model"
#   checkpoint_prefix = os.path.join(checkpoint_directory, 'Alternative_test')
#
#
#   with tf.device("/cpu:0"):
# ################## Training and Testing
#       EPOCHS = 1
#       loss1 = 2000
#
#       for epoch in range(EPOCHS):
#         print('The current epoch={}'.format(epoch))
#
#         for step, data in enumerate(train_ds):
#           loss=train_step(model, loss_object, Crossentropy_fun, optimizer, data[0], data[1])
#           if loss<loss1:
#             loss1=loss
#             ## just like check point without ckpt manager, more simpler
#             model.save_weights(filepath=checkpoint_prefix, overwrite=True, save_format=None, options=None)
#             print("In Training step {:d}: total_loss ={} ".format(step, loss))
#
#         model.load_weights(filepath=checkpoint_prefix)
#         for step, data in enumerate(test_ds):
#           loss=test_step(model, loss_object, Crossentropy_fun, data[0], data[1])
#           if step % 100 == 0:
#             print("In Testing step {:d}: total_loss ={} ".format(step, loss))
#
# ########################     after restore         ################################################################
#       print("Restored from {}..................".format(checkpoint_prefix))
#       model.load_weights(filepath=checkpoint_prefix)
#       for step, data in enumerate(test_ds):
#         loss = test_step(model, loss_object, Crossentropy_fun, data[0], data[1])
#         if step % 100 == 0:
#           print("In Testing step {:d}: total_loss ={} ".format(step, loss))


# ########################     after restore         ################################################################
'''Using tf.keras.models.save_model(model,filepath=checkpoint_prefix'''
if __name__ == '__main__':
  # prepare data:
  mnist = tf.keras.datasets.mnist
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0
  print(x_train.shape,y_train.shape)
  print(x_test.shape,y_test.shape)
  # Add a channels dimensionself.layer1
  x_train = x_train[..., tf.newaxis].astype("float32")
  x_test = x_test[..., tf.newaxis].astype("float32")

  train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(60)
  test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(10)
  print(train_ds)
  print(test_ds)
  # for step, data in enumerate(train_ds):
  #   print('The step {} data is {}'.format(step,data[1]))
  #   if step==50:
  #     break

  ## Create an instance of the model
  # model = MyModel()
  # optimizer = tf.keras.optimizers.Adam()
  Crossentropy_fun = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  #########################################3
  checkpoint_directory = "/home/brutal/PycharmProjects/Project-YOLOv3/model_save/my_model0"
  checkpoint_prefix = os.path.join(checkpoint_directory, 'Alternative_test')

  ################## Training and Testing
  with tf.device("/cpu:0"):
      EPOCHS = 1
      loss1 = 2000

      for epoch in range(EPOCHS):
        print('The current epoch={}'.format(epoch))

        for step, data in enumerate(train_ds):
          loss=train_step(model, loss_object, Crossentropy_fun, optimizer, data[0], data[1])
          if loss<loss1:
            loss1=loss
            ## just like check point without ckpt manager, more simpler
            # tf.keras.models.save_model(model,filepath=checkpoint_prefix)
            # model.save(filepath=checkpoint_prefix)
            print("In Training step {:d}: total_loss ={} ".format(step, loss))

        tf.keras.models.save_model(model, filepath=checkpoint_prefix)
        # reconstructed_model = tf.keras.models.load_model(filepath=checkpoint_prefix)
        for step, data in enumerate(test_ds):
          loss=test_step(model, loss_object, Crossentropy_fun, data[0], data[1])
          if step % 100 == 0:
            print("In Testing step {:d}: total_loss ={} ".format(step, loss))

#######################     after restore         ################################################################
  with tf.device("/cpu:0"):
      print("Restored from {}..................".format(checkpoint_prefix))
      reconstructed_model = tf.keras.models.load_model(filepath=checkpoint_prefix)
      for step, data in enumerate(test_ds):
        loss = test_step(reconstructed_model, loss_object, Crossentropy_fun, data[0], data[1])
        if step % 100 == 0:
          print("In Testing step {:d}: total_loss ={} ".format(step, loss))


















