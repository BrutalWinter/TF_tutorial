'''
The tf.data.Dataset API supports writing descriptive and efficient input pipelines. Dataset usage follows a common pattern:
Create a source dataset from your input data.
Apply dataset transformations to preprocess the data.
Iterate over the dataset and process the elements.

Iteration happens in a streaming fashion, so the full dataset does not need to fit into memory.
'''

import os
import tensorflow as tf
from tensorflow import keras
# import tensorflow_datasets as tfds
import numpy as np
import functools
import pandas as pd


################### Numpy data  #################################################################
# def create_model():
#   model = tf.keras.models.Sequential([
#     keras.layers.Flatten(input_shape=(28, 28)),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dense(64, activation='relu'),
#     keras.layers.Dropout(0.2),
#     keras.layers.Dense(10, activation='softmax'),
#   ])
#
#   model.compile(optimizer=tf.keras.optimizers.RMSprop(),
#                 loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
#                 metrics=['accuracy'])
#   return model
##########################################


# DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
#
# path = tf.keras.utils.get_file('mnist.npz', DATA_URL) #Downloads a file from a URL if it not already in the cache.
# # Return:Path to the downloaded file
# with np.load(path) as data:
#   train_examples = data['x_train']
#   train_labels = data['y_train']
#   test_examples = data['x_test']
#   test_labels = data['y_test']
#
# # 使用 tf.data.Dataset 加载 NumPy 数组
# train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
# test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))
#
# # 打乱和批次化数据集
# BATCH_SIZE = 64
# SHUFFLE_BUFFER_SIZE = 100
#
# train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
# # For perfect shuffling, a buffer size greater than or equal to the full size of the dataset is required.!!!!!!!!!
# #1 Randomly shuffles the elements of this dataset.2 Combines consecutive elements of this dataset into batches.
# test_dataset = test_dataset.batch(BATCH_SIZE)
#
# model1 = create_model()
# model1.fit(train_dataset, epochs=10)
# model1.evaluate(test_dataset)
####################################################################################


################### CSV data  #################################################################
TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"
train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)
########################################################################################################################
# pd 读取CSV 文件: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
NUMERIC_FEATURES = ['age','n_siblings_spouses','parch', 'fare']
desc = pd.read_csv(train_file_path)[NUMERIC_FEATURES].describe()
#pandas.DataFrame.describe: Descriptive statistics include those that summarize the central tendency, dispersion and shape of a dataset’s distribution
# By default the lower percentile is 25 and the upper percentile is 75. The 50 percentile is the same as the median.
print(desc)
print(desc.T)
MEAN = np.array(desc.T['mean'])
STD = np.array(desc.T['std'])
print(MEAN)
print(STD)


pd.set_option('display.max_rows', 10)#WebSite'pandas中关于DataFrame行，列显示不完全（省略）的解决办法 https://blog.csdn.net/weekdawn/article/details/81389865'
# pd.set_option('display.max_columns', None)
pd.set_option('display.max_columns', 10)
data=pd.read_csv(train_file_path)
print(type(data))
# print(data.head())
# print(data['survived'])
# print(data['sex'][0:20])
########################################################################################################################


'''正如你看到的那样，CSV 文件的每列都会有一个列名。dataset 的构造函数会自动识别这些列名。
如果你使用的文件的第一行不包含列名，那么需要将列名通过字符串列表传给 make_csv_dataset 函数的 column_names 参数。
CSV_COLUMNS = ['survived', 'sex', 'age', 'n_siblings_spouses', 'parch', 'fare', 'class', 'deck', 'embark_town', 'alone']
dataset = tf.data.experimental.make_csv_dataset(...,...)
这个示例使用了所有的列。如果你需要忽略数据集中的某些列，创建一个包含你需要使用的列的列表，然后传给构造器的（可选）参数 select_columns。
dataset = tf.data.experimental.make_csv_dataset(...,select_columns = columns_to_use,...)
'''
# 现在从文件中读取 CSV 数据并且创建 dataset:
np.set_printoptions(precision=4, suppress=True)#控制输出方式
# precision：控制输出的小数点个数，默认是8
# np.array([1.123456789])             [ 1.1235]
# threshold：控制输出的值的个数，其余以…代替；
# np.arange(10)        [0 1 2 3 4 5 6 7 8 9]
# 当设置打印显示方式threshold=np.nan，意思是输出数组的时候完全输出，不需要省略号将中间数据省略
# suppress： 当suppress=True，表示小数不需要以科学计数法的形式输出

LABEL_COLUMN = 'survived'# 对于包含模型需要预测的值的列是你需要显式指定的。
LABELS = [0, 1]
def get_dataset(file_path,**kwargs):
  dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=5, # 为了示例更容易展示，手动设置较小的值 `label_name` provided must be one of the columns.
      label_name=LABEL_COLUMN,#A optional string corresponding to the label column.
    # If provided, the data for this column is returned as a separate Tensor from the features dictionary,
      na_value="?",#Additional string to recognize as NA/NaN.
      num_epochs=1,#An int specifying the number of times this dataset is repeated. If None, cycles through the dataset forever.
      ignore_errors=True,
      **kwargs)# If True, ignores errors with CSV file parsing, such as malformed data or empty lines, and moves on to the next valid CSV record.
  # Otherwise, the dataset raises an error and stops processing when encountering any invalid records. Defaults to False
  return dataset

def show_batch(dataset):
  for batch, label in dataset.take(1):#dataset.take() Creates a Dataset with at most count elements from this dataset(random).
    print("batch={},\nLABEL_COLUMN ={}".format(batch,label))
    for key, value in batch.items():
      print("{:20s} and {}".format(key,value))


def normalize_numeric_data(data, mean, std):
  # Center the data
  return (data-mean)/std






def show_batch_without_label(dataset):
  for batch in dataset.take(1):#dataset.take() Creates a Dataset with at most count elements from this dataset(random).
    for key, value in batch.items():
      print("{:20s} and {}".format(key,value))



raw_train_data = get_dataset(train_file_path)
raw_test_data = get_dataset(test_file_path)
print('raw_train_data:')
show_batch(raw_train_data)
print('\n\n')

CSV_COLUMNS =['survived', 'sex2', 'age3', 'n_siblings_spouses4', 'parch5', 'fare6', 'class7', 'deck8', 'embark_town9', 'alone10']
temp_dataset = get_dataset(train_file_path, column_names=CSV_COLUMNS)
print('temp_dataset with different name:')
show_batch(temp_dataset)
# show_batch(temp_dataset.shape)
print('\n\n')

#A CSV file can contain a variety of data types.
# Typically you want to convert from those mixed types to a fixed length vector before feeding the data into your model.
# The primary advantage of doing the preprocessing inside your model is that when you export the model it includes the preprocessing.
# This way you can pass the raw data directly to your model.
SELECT_COLUMNS = ['survived', 'age', 'n_siblings_spouses', 'parch', 'fare']# When this argument is specified, only a subset of CSV columns will be parsed and returned,
# corresponding to the columns specified.
DEFAULTS = [0, 0.0, 0.0, 0.0, 0.0]#defaulting to 0 for numeric values and "" for string values.
temp_dataset = get_dataset(train_file_path, select_columns=SELECT_COLUMNS, column_defaults = DEFAULTS)
print('temp_dataset:')
show_batch(temp_dataset)
example_batch, labels_batch = next(iter(temp_dataset))
print('exapmle_batch={}'.format(example_batch))
print('labels_batch={}'.format(labels_batch))
print('\n\n')



# ###########################################################
# def pack(features, label):
#   F_list=list(features.values())
#   # print('the length of list(features.values())={}'.format(len(F_list)))
#   return tf.stack(F_list, axis=-1), label
#
# packed_dataset = temp_dataset.map(pack) #Maps map_func across the elements of this dataset.
# # and returns a new dataset containing the transformed elements, in the same order as they appeared in the input.
# for features, labels in packed_dataset.take(2):
#   print('###{}\n'.format(features.numpy()))
#   print(labels.numpy())
# example_batch, labels_batch = next(iter(temp_dataset))
# print('\n\n')
# ###########################################################
#So define a more general preprocessor that selects a list of numeric features and packs them into a single column:
class PackNumericFeatures(object):
  def __init__(self, names):
    self.names = names

  def __call__(self, features, labels):
    numeric_features = [features.pop(name) for name in self.names]
    numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
    numeric_features = tf.stack(numeric_features, axis=-1)
    features['numeric'] = numeric_features
    return features, labels


NUMERIC_FEATURES = ['age','n_siblings_spouses','parch', 'fare']
packed_train_data = raw_train_data.map(PackNumericFeatures(NUMERIC_FEATURES))
packed_test_data = raw_test_data.map(PackNumericFeatures(NUMERIC_FEATURES))
print('packed_train_data shows like:{}')
show_batch(packed_train_data)
example_batch, labels_batch = next(iter(packed_train_data))
print('\n\n')


#返回一个新的部分对象，该对象在被调用时的行为类似于使用位置参数args和关键字arguments关键字调用的func。
# 如果将更多参数提供给调用，则将它们附加到args。 如果提供了其他关键字参数，它们将扩展并覆盖关键字。
normalizer = functools.partial(normalize_numeric_data, mean=MEAN, std=STD)
numeric_column = tf.feature_column.numeric_column('numeric', normalizer_fn=normalizer, shape=[len(NUMERIC_FEATURES)])
numeric_columns = [numeric_column]
print('example_batch[numeric]={}'.format(example_batch))
print('example_batch[numeric]={}'.format(example_batch['numeric']))
numeric_layer = tf.keras.layers.DenseFeatures(numeric_columns)
print('numeric_layer:{}'.format(numeric_layer(example_batch).numpy()))





###########################################################
# Categories column generator
# ###########################################################
# Some of the columns in the CSV data are categorical columns. That is, the content should be one of a limited set of options.
CATEGORIES = {
    'sex': ['male', 'female'],
    'class' : ['First', 'Second', 'Third'],
    'deck' : ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    'embark_town' : ['Cherbourg', 'Southhampton', 'Queenstown'],
    'alone' : ['y', 'n']
}
categorical_columns = []
for feature, vocab in CATEGORIES.items():
  print('feature is {}\n vocab is {}'.format(feature,vocab))
  cat_col = tf.feature_column.categorical_column_with_vocabulary_list(key=feature, vocabulary_list=vocab)
  print('cat_col is {}'.format(cat_col))
  categorical_columns.append(tf.feature_column.indicator_column(cat_col))

print(categorical_columns)
categorical_layer = tf.keras.layers.DenseFeatures(categorical_columns)
print('categorical_layer demo:{}'.format(categorical_layer(example_batch).numpy()))

preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns+numeric_columns)
print(preprocessing_layer(example_batch).numpy()[0:2])



###########################################################
################ model, train, evalulating:
###########################################################
def create_model1():
  model = tf.keras.Sequential([
    preprocessing_layer,
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1),
  ])

  model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy'])
  return model


model = create_model1()
# model.summary() It must has input shape in order to manifest


train_data = raw_train_data.shuffle(500)
packed_train_data = raw_train_data.map(PackNumericFeatures(NUMERIC_FEATURES))
test_data = raw_test_data
packed_test_data = raw_test_data.map(PackNumericFeatures(NUMERIC_FEATURES))
model.fit(packed_train_data, epochs=20)

test_loss, test_accuracy = model.evaluate(packed_test_data)
predictions = model.predict(packed_test_data)

print(list(packed_test_data)[0][1])
for prediction, survived in zip(predictions[:20], list(packed_test_data)[0][1][:20]):
  prediction = tf.sigmoid(prediction).numpy()
  print("Predicted survival: {:.2%}".format(prediction[0]),
        " | Actual outcome: ",
        ("SURVIVED" if bool(survived) else "DIED"))
###########################################################
# ###########################################################