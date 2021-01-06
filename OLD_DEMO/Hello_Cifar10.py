import tensorflow as tf
from tensorflow.keras import datasets, layers, models # FOR SPEEED UP PURPOSE
import matplotlib.pyplot as plt
###################################




def plot_history(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
###################################




(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
# 将像素的值标准化至0到1的区间内。
train_images, test_images = train_images / 255.0, test_images / 255.0
print(train_images.shape)
print(train_images[10000].shape)
print(train_labels.shape)
print(train_labels[10000])
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']

# plt.figure(figsize=(8,8))
# for i in range(16):
#     plt.subplot(4,4,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[4000+i], cmap=plt.cm.binary)
#     # 由于 CIFAR 的标签是 array，
#     # 因此您需要额外的索引（index）。[1] and 0 的区别
#     plt.xlabel(class_names[train_labels[4000+i][0]])
# plt.show()

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2),strides=(2, 2),padding='valid'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2),strides=(2, 2),padding='valid'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu',kernel_regularizer='l2'), #Dense 层的输入为向量（一维）
    tf.keras.layers.Dense(10,kernel_initializer=tf.keras.initializers.GlorotUniform())
])
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
            # metrics=['accuracy',tf.keras.metrics.SparseCategoricalAccuracy(),tf.keras.metrics.CategoricalAccuracy()])

# history = model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
history = model.fit(train_images, train_labels, batch_size=40, epochs=6, validation_data=(test_images, test_labels))

###########################
#A History object. Its History.history attribute is a record of training loss values and metrics values at successive epochs,
# as well as validation loss values and validation metrics values (if applicable).
plot_history(history)
plt.show()
###########################
# Evaluate the model Returns the loss value & metrics values for the model in test mode.
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
################################### test code start here:
# x = tf.constant([[1., 2., 3.,111],
#                  [4., 5., 6.,222],
#                  [7., 8., 9.,333],
#                  [10., 11., 12.,444]])
# x = tf.reshape(x, [1, 4, 4, 1])
# max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(3, 3),
#    strides=None, padding='valid')
# A=max_pool_2d(x)
# print(A)