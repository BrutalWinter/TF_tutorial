import os
import tempfile
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
print('''
    A SavedModel contains a complete TensorFlow program, including weights and computation. It does not require the original model building code to run, 
which makes it useful for sharing or deploying (with TFLite, TensorFlow.js, TensorFlow Serving, or TensorFlow Hub).

This document dives into some of the details of how to use the low-level tf.saved_model api:
    1.If you are using a tf.keras.Model the keras.Model.save(output_path) method may be all you need: Save and load Keras models
    2.If you just want to save/load weights during training see the guide: Training checkpoints.
    ''')

# ##############################    Creating a SavedModel from Keras       #######################################
print('##########      part1:       ##########\n'*5)
# #For a quick introduction, this section exports a pre-trained Keras model and serves image classification requests with it.
# # The rest of the guide will fill in details and discuss other ways to create SavedModels.
# tmpdir = tempfile.mkdtemp()
# print('tmpdir==============>',tmpdir)
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# for device in physical_devices:
#   tf.config.experimental.set_memory_growth(device, True)
#
# #You'll use an image of Grace Hopper as a running example, and a Keras pre-trained image classification model since it's easy to use.
# # Custom models work too, and are covered in detail later.
# file = tf.keras.utils.get_file(
#     "grace_hopper.jpg",
#     "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg")
# img = tf.keras.preprocessing.image.load_img(file, target_size=[224, 224])
# plt.imshow(img)
# plt.axis('off')
# plt.show()
# x = tf.keras.preprocessing.image.img_to_array(img)
# x = tf.keras.applications.mobilenet.preprocess_input(x[tf.newaxis,...])
#
# labels_path = tf.keras.utils.get_file(
#     'ImageNetLabels.txt',
#     'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
# imagenet_labels = np.array(open(labels_path).read().splitlines())
#
# pretrained_model = tf.keras.applications.MobileNet()
# result_before_save = pretrained_model(x)
# decoded = imagenet_labels[np.argsort(result_before_save)[0,::-1][:5]+1]
# print("Result before saving:\n", decoded)
#
# mobilenet_save_path = os.path.join(tmpdir, "mobilenet/1/")
# tf.saved_model.save(pretrained_model, mobilenet_save_path)
# '''
# The save-path follows a convention used by TensorFlow Serving where the last path component (1/ here) is a version number for your model
# - it allows tools like Tensorflow Serving to reason about the relative freshness.
# You can load the SavedModel back into Python with tf.saved_model.load and see how Admiral Hopper's image is classified.'''
#
# loaded = tf.saved_model.load(mobilenet_save_path)
# print(list(loaded.signatures.keys()))  # ["serving_default"]
# infer = loaded.signatures["serving_default"]
# print(infer.structured_outputs)
#
# # Running inference from the SavedModel gives the same result as the original model.
#
# labeling = infer(tf.constant(x))[pretrained_model.output_names[0]]
# decoded = imagenet_labels[np.argsort(labeling)[0,::-1][:5]+1]
# print("Result after saving and loading:\n", decoded)


print('##########      part2:       ##########\n'*5)
##############################    Saving a custom model      #######################################
# tf.saved_model.save   supports saving   tf.Module   objects and its subclasses, like tf.keras.Layer and tf.keras.Model.
# Let's look at an example of saving and restoring a tf.Module.

class CustomModule(tf.Module):

  def __init__(self):
    super(CustomModule, self).__init__()
    self.v = tf.Variable(1.)

  @tf.function
  def __call__(self, x):
    print('Tracing with', x)
    return x * self.v

  @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
  def mutate(self, new_v):
    self.v.assign(new_v)

module = CustomModule()

tmpdir='./model_save'
module_no_signatures_path = os.path.join(tmpdir, 'module_no_signatures')
Result=module(tf.constant((2.)))
print(Result)
#Warning: Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
#This warning occurs when the shape of the model output does not match with the ground_truth labels. Although it can be broadcasted.
print('Saving model...')
tf.saved_model.save(module, module_no_signatures_path)

print('##########      part3:       ##########\n'*5)
##############################    Saving a custom model      #######################################
'''
When you load a SavedModel in Python, all tf.Variable attributes, tf.function-decorated methods, 
and tf.Modules are restored in the same object structure as the original saved tf.Module.
'''
imported = tf.saved_model.load(module_no_signatures_path)
assert imported(tf.constant(3.)).numpy() == 3
print('assert done')
imported.mutate(tf.constant(2.))
assert imported(tf.constant(3.)).numpy() == 6
print('assert done')
Result=imported(tf.constant((3.))).numpy()
print(Result)
# Because no Python code is saved, calling a tf.function with a new input signature will fail:
# Result=imported(tf.constant([3.]))

############################################################3
# Basic fine-tuning
# Variable objects are available, and you can backprop through imported functions. That is enough to fine-tune (i.e. retrain) a SavedModel in simple cases.
optimizer = tf.optimizers.SGD(0.05)

def train_step(imported,optimizer):
  with tf.GradientTape() as tape:
    loss = (10. - imported(tf.constant(2.))) ** 2
  variables = tape.watched_variables()
  grads = tape.gradient(loss, variables)
  optimizer.apply_gradients(zip(grads, variables))
  return loss

for _ in range(5):
  # "v" approaches 5, "loss" approaches 0
  print("Phase 1: loss={:.2f} v={:.2f}".format(train_step(imported,optimizer), imported.v.numpy()))

tmpdir='./model_save'
module_no_signatures_path = os.path.join(tmpdir, 'module_no_signatures2')
tf.saved_model.save(imported, module_no_signatures_path)
imported = tf.saved_model.load(module_no_signatures_path)
for _ in range(5):
  # "v" approaches 5, "loss" approaches 0
  print("Phase 2: loss={:.2f} v={:.2f}".format(train_step(imported,optimizer), imported.v.numpy()))

############################################################3
# General fine-tuning: