import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

####################################### MNIST example     ##################################
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0
# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
print('The shape of train_images={}\n The length of train_labels={}'.format(x_train.shape, len(y_train)))
print('The shape of test_images={}\n The length of test_labels={}'.format(x_test.shape, y_test.shape))

# plt.imshow(x_train[2])
# # plt.imshow(x_train[0],cmap=plt.cm.binary)
# # print(x_train[0])
# plt.show()


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax)
])

# model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(encoder.vocab_size, 64),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(1)
# ])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=2, verbose=1)
print('Start Evalulating:')
model.evaluate(x_test,  y_test, verbose=1)


# ############################ Fashion MNIST ##############################################
# def plot_image(i, predictions_array, true_label, img):
#   true_label, img = true_label[i], img[i]
#   plt.grid(False)
#   plt.xticks([])
#   plt.yticks([])
#
#   plt.imshow(img, cmap=plt.cm.binary)
#
#   predicted_label = np.argmax(predictions_array)
#   print(predicted_label)
#   if predicted_label == true_label:
#     color = 'blue'
#   else:
#     color = 'red'
#
#   plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
#                                 100*np.max(predictions_array),
#                                 class_names[true_label]),
#                                 color=color)
#
#
# def plot_value_array(i, predictions_array, true_label):
#   true_label = true_label[i]
#   plt.grid(False)
#   plt.xticks(range(10))
#   plt.yticks([])
#   thisplot = plt.bar(range(10), predictions_array, color="#777777")
#   plt.ylim([0, 1])
#   predicted_label = np.argmax(predictions_array)
#
#   thisplot[predicted_label].set_color('red')
#   thisplot[true_label].set_color('blue')
#
# def model_mnist_fashion():
#   # model = tf.keras.Sequential([
#   #   tf.keras.layers.Flatten(input_shape=(28, 28)),
#   #   tf.keras.layers.Dense(128, activation='relu'),
#   #   tf.keras.layers.Dense(10)
#   #
#   # ])
#   # model = tf.keras.models.Sequential()
#   model = tf.keras.Sequential()
#   model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
#   model.add(tf.keras.layers.Dense(128, activation='relu'))
#   model.add(tf.keras.layers.Dense(10))
#
#
#   return model

########################### Fashion MNIST ##############################################
if __name__ == '__main__':
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    print('The shape of train_images={}\n The length of train_labels={}'.format(train_images.shape, len(train_labels)))
    print('The shape of test_images={}\n The length of test_labels={}'.format(test_images.shape, test_labels.shape))

    # plt.figure()
    # plt.imshow(train_images[5])
    # plt.colorbar()
    # plt.grid(False)
    # plt.show()

    # plt.figure(figsize=(10,10))
    # for i in range(6):
    #     plt.subplot(3,2,i+1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(train_images[i], cmap=plt.cm.binary)
    #     plt.xlabel(class_names[train_labels[i]])
    # plt.show()

    model = model_mnist_fashion()
    model.compile(optimizer='adam',
                  # optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  # from_logits=True:Whether y_pred is expected to be a logits tensor. By false, we assume that y_pred encodes a probability distribution.
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=4)

    print('Start Evalulating:')
    model.evaluate(test_images, test_labels, verbose=1)



    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test_images)
    print(predictions.shape)
    i = 10
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(1, 2, 2)
    plot_value_array(i, predictions[i], test_labels)
    plt.show()

    # Plot the first X test images, their predicted label, and the true label
    # Color correct predictions in blue, incorrect predictions in red
    num_rows = 6
    num_cols = 4
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions, test_labels, test_images)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions, test_labels)
    plt.show()

    j_test=1
    img = test_images[j_test]
    print(img.shape)
    img = (np.expand_dims(img, 0))
    print(img.shape)
    predictions_single = probability_model.predict(img) # Since there is only one 28*28 its dimension has to be extended
    print(predictions_single)
    print(predictions_single[0])

    plot_value_array(j_test, predictions_single[0], test_labels)
    _ = plt.xticks(range(10), class_names, rotation=45)
    plt.show()
    print(np.argmax(predictions_single[0]))






















