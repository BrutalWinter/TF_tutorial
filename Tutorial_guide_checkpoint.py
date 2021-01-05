import tensorflow as tf
import os
import numpy as np
##############################    Ctrl+/ this part if run       #######################################
print(r'''#################
Checkpoint's constructor accepts keyword arguments whose values are types that contain trackable state,
such as tf.keras.optimizers.Optimizer implementations, tf.Variables, tf.data.Dataset iterators, tf.keras.Layer implementations, or tf.keras.Model implementations.
It saves these values with a checkpoint, and maintains a save_counter for numbering checkpoints.
#################''')

# checkpoint_directory = "/tmp/training_checkpoints"
# checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
# print('checkpoint_prefix=====>',checkpoint_prefix)
#
# # Create a Checkpoint that will manage two objects with trackable state,
# # one we name "optimizer" and the other we name "model".
# checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
# status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))
# for _ in range(num_training_steps):
#   optimizer.minimize( ... )  # Variables will be restored on creation.
# status.assert_consumed()  # Optional sanity checks.
# checkpoint.save(file_prefix=checkpoint_prefix)
# '''
# 1.Checkpoint.save() and Checkpoint.restore() write and read object-based checkpoints.
# 2.Checkpoint objects have dependencies on the objects passed as keyword arguments to their constructors,
# and each dependency is given a name that is identical to the name of the keyword argument for which it was created.
# 3.TensorFlow classes like Layers and Optimizers will automatically add dependencies on their own variables'''
# ############################################################################################################


##############################
#The phrase "Saving a TensorFlow model" typically means one of two things:
  #1 Checkpoints,
  #2 SavedModel.
print("Checkpoints capture the exact value of all parameters (tf.Variable objects) used by a model. "
      "Checkpoints do not contain any description of the computation defined by the mode")
print("only useful when source code that [will use the saved parameter values] is available.")


class Net(tf.keras.Model):
  """A simple linear model."""

  def __init__(self, **kwargs):
    super(Net, self).__init__(**kwargs)
    self.l1 = tf.keras.layers.Dense(5)

  def call(self, x):
    return self.l1(x)

net = Net()
net.save_weights('model_save/easy_checkpoint/easy_checkpoint') #tf.keras.Model.save_weights saves a TensorFlow checkpoint


# The following example constructs a simple linear model, then writes checkpoints which contain values for all of the model's variables.
# You can easily save a model-checkpoint with Model.save_weights
# Manual checkpointing:
def toy_dataset():
  inputs = tf.range(10.)[:, None]
  labels = inputs * 5. + tf.range(5.)[None, :]
  return tf.data.Dataset.from_tensor_slices(
    dict(x=inputs, y=labels)).repeat(500).batch(2)


def train_step(net, example, optimizer):
  """Trains `net` on `example` using `optimizer`."""
  with tf.GradientTape() as tape:
    output = net(example['x'],training=True)
    loss = tf.reduce_mean(tf.abs(output - example['y']))
  variables = net.trainable_variables
  gradients = tape.gradient(loss, variables)
  optimizer.apply_gradients(zip(gradients, variables))
  return loss

def test_step(net, example):
  for data in example:
    # print(data)
    output = net(data['x'],training=False)
    loss = tf.reduce_mean(tf.abs(output - data['y']))
    print('loss of dataset.take(1)={}'.format(loss))


# #####################  data and label test:
# print(tf.range(10.)[:, None]) # shape (10,1)
# print(tf.range(10.))# shape (10,)
# inputs = tf.range(10.)[:, None]
# labels = inputs * 5. + tf.range(5.)[None, :] #braodcast trick be(10,5)
# print(labels)
# #####################
#Create the checkpoint objects:
# To manually make a checkpoint you will need a tf.train.Checkpoint object. Where the objects you want to checkpoint are set as attributes on the object.
print('A tf.train.CheckpointManager can also be helpful for managing multiple checkpoints.')
opt = tf.keras.optimizers.Adam(0.1)
dataset = toy_dataset()
print(dataset)

# for step, data in enumerate(dataset):
#   print('The step {} data is {}'.format(step,data))
#   if step==50:
#     break

A=dataset.take(1) # for testing
# print('list is =',list(A.as_numpy_iterator()))
# for step, data in enumerate(A):
#   print('First:The step {} data in dataset.take(2) is {}'.format(step,data))
# for step, data in enumerate(A):
#   print('Second The step {} data in dataset.take(2) is {}'.format(step, data))



iterator = iter(dataset)
# print('iterator is=',iterator)
# print('next(iterator)=',next(iterator))
# print('next(iterator)=',next(iterator))

# ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=net, iterator=iterator)
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=net)
checkpoint_directory = "/home/brutal/PycharmProjects/Project-YOLOv3/model_save"
checkpoint_prefix = os.path.join(checkpoint_directory, "tf_ckpts")
manager = tf.train.CheckpointManager(checkpoint=ckpt,  directory=checkpoint_prefix, max_to_keep=3)
# manager = tf.train.CheckpointManager(checkpoint=ckpt,  directory='./model_save/tf_ckpts', max_to_keep=3)
#./当前目录 /home/brutal/PycharmProjects/Project-YOLOv3

# Train and checkpoint the model:
def train_and_checkpoint(net, manager, ckpt):
  if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
    ckpt.restore(manager.latest_checkpoint)
  else:
    print("Initializing from scratch.")

  for _ in range(50):
    example = next(iterator)
    # print('example',example)
    loss = train_step(net, example, opt)
    ckpt.batch_run.assign_add(1)
    # print(ckpt.step)
    if int(ckpt.batch_run) % 10 == 0:
      save_path = manager.save()
      print("Saved checkpoint for step {}: {}".format(int(ckpt.batch_run), save_path))
      print("loss {:1.2f}".format(loss.numpy()))
      print('\n')
#
# print("Before Training =========> its loss:")
# test_step(net, A)
# test_step(net, A)
# print('\n')

train_and_checkpoint(net, manager, ckpt)
print("After Training =========> its loss:")
test_step(net, A)
test_step(net, A)
test_step(net, A)
print('\n')

# print(manager.checkpoint)
# print(manager.checkpoints)
print("Starting Restoring =================>:")
#Restore and continue training, After the first you can pass a new model and manager, but pickup training exactly where you left off:
opt1 = tf.keras.optimizers.Adam(0.1)
net1 = Net()
ckpt1 = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt1, net=net1)
manager1 = tf.train.CheckpointManager(ckpt1, '/home/brutal/PycharmProjects/Project-YOLOv3/model_save/tf_ckpts', max_to_keep=3)
# print("Before Restore =========> its loss:")
# test_step(net1, A)
# test_step(net1, A)
# print('\n')

print("Restored from {}".format(manager1.latest_checkpoint))
ckpt1.restore(manager1.latest_checkpoint)
print("After Restore =========> its loss:")
test_step(net1, A)
test_step(net1, A)
test_step(net1, A)
print('\n')

# print("Continue Training")
# train_and_checkpoint(net1, manager1, ckpt1)
# print("After Training Again =========> its loss:")
# test_step(net1, A)
# test_step(net1, A)
# print('\n')



# ##############################
# #Loading mechanics:
# '''Calling restore() on a tf.train.Checkpoint object queues the requested restorations,
# restoring variable values as soon as there's a matching path from the Checkpoint object.
# For example: you can load just the bias from the model we defined above by reconstructing one path to it through the network and the layer.'''
# to_restore = tf.Variable(tf.zeros([5]))
# print(to_restore.numpy())  # All zeros
# fake_layer = tf.train.Checkpoint(bias=to_restore)
# fake_net = tf.train.Checkpoint(l1=fake_layer)
# new_root = tf.train.Checkpoint(net=fake_net)
# status = new_root.restore(tf.train.latest_checkpoint('./model_save/tf_ckpts/'))
# print(to_restore.numpy())  # This gets the restored value.
#
# print(status.assert_existing_objects_matched())
# '''There are many objects in the checkpoint which haven't matched, including the layer's kernel and the optimizer's variables.
# status.assert_consumed() only passes if the checkpoint and the program match exactly, and would throw an exception here.'''
#
# # Manually inspecting checkpoints:
# # tf.train.list_variables lists the checkpoint keys and shapes of variables in a checkpoint. Checkpoint keys are paths in the graph displayed above.
# print(tf.train.list_variables(tf.train.latest_checkpoint('./model_save/tf_ckpts/')))
#
# # Summary:TensorFlow objects provide an easy automatic mechanism for saving and restoring the values of variables they use.



############################## Example:   ##############################
# define layers and model:
class layer(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super(layer, self).__init__(**kwargs)
    self.layer1 = tf.keras.layers.Multiply()

  def call(self, *inputs):
    x = self.layer1([*inputs])
    return x
# layer1=layer()
# B1=np.arange(10).reshape(10, 1)
# B2=np.arange(10,20).reshape(10, 1)
# print(B1)
# print(B2)
# A=layer1(B1,B2)
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
    x=self.layer(x2,x3)
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

  # Create an instance of the model
  model = MyModel()
  # loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  optimizer = tf.keras.optimizers.Adam()
  Crossentropy_fun = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  #########################################3
  # tmpdir='./model_save'
  # module_no_signatures_path = os.path.join(tmpdir, 'module_test')

  ckpt = tf.train.Checkpoint(optimizer=optimizer, net=model)
  checkpoint_directory = "/home/brutal/PycharmProjects/Project-YOLOv3/model_save"
  checkpoint_prefix = os.path.join(checkpoint_directory, 'module_test')
  manager = tf.train.CheckpointManager(checkpoint=ckpt, directory=checkpoint_prefix, max_to_keep=3)

  # print('reconstructed model reslt:')
  # test_step(imported, A)
  # tf.saved_model.save(model, module_no_signatures_path)
  # print('Trained model reslt:')
  # test_step(net, A)
  #


  with tf.device("/cpu:0"):
# ################## Training and Testing
#       EPOCHS = 1
#       loss1 = 2000
#
#       if manager.latest_checkpoint:
#         print("Restored from {}".format(manager.latest_checkpoint))
#         ckpt.restore(manager.latest_checkpoint)
#       else:
#         print("Initializing from scratch.")
#
#       for epoch in range(EPOCHS):
#         print('The current epoch={}'.format(epoch))
#
#         for step, data in enumerate(train_ds):
#           loss=train_step(model, loss_object, Crossentropy_fun, optimizer, data[0], data[1])
#           if loss<loss1:
#             save_path = manager.save()
#             print("Saved checkpoint step={} and its path in {}".format(step,save_path))
#             loss1=loss
#             print("In Training step {:d}: total_loss ={} ".format(step, loss))
#
#           # if step % 100 == 0:
#           #   print("In Training step {:d}: total_loss ={} ".format(step, loss))
#
#         ckpt.restore(manager.latest_checkpoint)
#         for step, data in enumerate(test_ds):
#           loss=test_step(model, loss_object, Crossentropy_fun, data[0], data[1])
#           if step % 100 == 0:
#             print("In Testing step {:d}: total_loss ={} ".format(step, loss))

  # imported = tf.saved_model.load(module_no_signatures_path)
#########################     after restore         ################################################################
      print("Restored from {}..................".format(manager.latest_checkpoint))
      ckpt.restore(manager.latest_checkpoint)
      for step, data in enumerate(test_ds):
        loss = test_step(model, loss_object, Crossentropy_fun, data[0], data[1])
        if step % 100 == 0:
          print("In Testing step {:d}: total_loss ={} ".format(step, loss))




















