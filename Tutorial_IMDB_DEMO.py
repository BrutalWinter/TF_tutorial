import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
####################################
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
# str.join(sequence) 用于将序列中的元素以指定的字符连接生成一个新的字符串。
# dict.get(key, default=None) 函数返回指定键的值，如果键不在字典中返回默认值。
####################################
def plot_history_all(history):
    history_dict = history.history
    print(history_dict.keys())  # dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']


    epochs = range(1, len(acc) + 1)
    fig = plt.figure(2,figsize=(15, 10),dpi=200) #开启一个窗口，同时设置大小，分辨率
    axes1 = fig.add_subplot(2, 1, 1) #通过fig添加子图，参数：行数，列数，第几个。
    axes2 = fig.add_subplot(2, 1, 2)

    # “bo”代表 "蓝点"    # b代表“蓝色实线”
    axes1.plot(epochs, loss, 'bo', label='Training loss')
    axes1.plot(epochs, val_loss, 'b', label='Validation loss')
    axes2.plot(epochs, acc, 'ro', label='Training accuracy')
    axes2.plot(epochs, val_acc, 'r', label='Validation accuracy')

    axes1.set_ylabel('Loss')
    axes1.set_xlabel('Epochs')
    axes1.set_title('Training and validation loss')
    axes1.legend(loc='upper left')

    axes2.set_ylabel('accuracy')
    axes2.set_xlabel('Epochs')
    axes2.set_title('Training and validation accuracy')
    axes2.legend(loc='upper left')
####################################







####################################
#####该数据集是经过预处理的：每个样本都是一个表示影评中词汇的整数数组。每个标签都是一个值为 0 或 1 的整数值，其中 0 代表消极评论，1 代表积极评论。
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
index=0
print(train_data.shape)
print(len(train_data[index]))
print(train_labels.shape)
print(train_labels[index])
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

#将整数转换回单词: 一个映射单词到整数索引的词典
word_index = imdb.get_word_index()
# 保留第一个索引
word_index = {k:(v+3) for k,v in word_index.items()} #item()方法把字典中每对key和value组成一个元组，并把这些元组放在列表中返回。
word_index["<PAD>"] = 0 ## dict[key] = value
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

print(decode_review(train_data[3]))
##############################################################
# 准备数据
# 影评——即整数数组必须在输入神经网络之前转换为张量。这种转换可以通过以下两种方式来完成：
#
# 将数组转换为表示单词出现与否的由 0 和 1 组成的向量，类似于 one-hot 编码。例如，序列[3, 5]将转换为一个 10,000 维的向量，
# 该向量除了索引为 3 和 5 的位置是 1 以外，其他都为 0。然后，将其作为网络的首层——一个可以处理浮点型向量数据的稠密层。
# 不过，这种方法需要大量的内存，需要一个大小为 num_words * num_reviews 的矩阵。
#
# 或者，我们可以填充数组来保证输入数据具有相同的长度，然后创建一个大小为 max_length * num_reviews 的整型张量。我们可以使用能够处理此形状数据的嵌入层作为网络中的第一层。
train_data = keras.preprocessing.sequence.pad_sequences(train_data,value=word_index["<PAD>"],padding='post',maxlen=256)
print('data shape now={}'.format(train_data.shape))
test_data = keras.preprocessing.sequence.pad_sequences(test_data,value=word_index["<PAD>"],padding='post',maxlen=256)
print(test_data.shape)
# 输入形状是用于电影评论的词汇数目（10,000 词）
vocab_size = 10000
model = tf.keras.models.Sequential([
    keras.layers.Embedding(vocab_size, 16),#This layer can only be used as the first layer in a model.
    keras.layers.GlobalAveragePooling1D(data_format='channels_last'),
    #input_shape = (2, 3, 4)
    # x = tf.random.normal(input_shape)
    # y = tf.keras.layers.GlobalAveragePooling1D()(x)
    # print(y.shape) (2,4)

    keras.layers.Dense(16, activation='relu',kernel_regularizer='l2'), #Dense 层的输入为向量（一维）
    keras.layers.Dense(1,activation='sigmoid',kernel_initializer=tf.keras.initializers.GlorotUniform())
])
model.summary()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
########## crosss validation
x_val = train_data[:8000]
x_train = train_data[8000:]

y_val = train_labels[:8000]
y_train = train_labels[8000:]
#model.fit() 返回一个 History 对象，该对象包含一个字典，其中包含训练阶段所发生的一切事件：
history = model.fit(x_train,
                    y_train,
                    epochs=50,
                    batch_size=500,
                    validation_data=(x_val, y_val),
                    verbose=1)
loss, metric = model.evaluate(test_data,  test_labels, verbose=1)

plot_history_all(history)
plt.show()



#################### test code here:
# person = {'name': 'lizhong', 'age': '26', 'city': 'BeiJing', 'blog': 'www.jb51.net'}
# for key, value in person.items():
#     print('key=', key, '，value=', value)
# print(person['name'])
# eNum_person=enumerate(person.items())
# print(list(eNum_person))
# for i, (k, v) in eNum_person:
#     if i in range(0, 3):
#         print(k, v)
####################