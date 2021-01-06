import os
import time
import sys
import random
import numpy as np
from numpy import array

import tensorflow as tf

# 最早使用的数据l格式load，用于对比评估新方法有效性
def load_data(path, batch_size = 64, input_shape=None, subset = None, train = True):
    print('load_data:', path, input_shape,subset)
    data_pattern = os.path.join(path, subset)
    data_files = tf.io.gfile.glob(data_pattern)
    if len(data_files) < 1:
        print('[ERROR] No train data files found in %s' % path)
        exit(-1)

    total_samples = sum([os.path.getsize(f) / 4 / 2040 for f in data_files])
    dataset = tf.data.FixedLengthRecordDataset(filenames = data_files,
                                               record_bytes = 4 * 2040)

    def transfer(value):
        value = tf.io.decode_raw(value, tf.float32)
        label = tf.cast(tf.slice(value, [0], [40]), tf.int32)
        data = tf.slice(value, [40], [2000])

        if input_shape==3:
            data = tf.reshape(data, [40,50])
            label = tf.reshape(label,[40,1])
        
        elif input_shape==4:
            data = tf.reshape(data, [40,50,1])
            label = tf.reshape(label,[40,1,1])
            
        elif input_shape==5:
            data = tf.reshape(data, [40,50,1,1])
            label = tf.reshape(label,[40,1,1,1])
        
        else:
            print('[ERROR] invalid input_shape %s' % input_shape)
            exit(-1)
            
        return data,label

    dataset = dataset.map(transfer)

    if train:
        #batched_dataset = dataset.batch(batch_size).prefetch(buffer_size = tf.data.experimental.AUTOTUNE).repeat()
        batched_dataset=dataset.shuffle(buffer_size=10000,reshuffle_each_iteration=True)
    else:
        batched_dataset=dataset.shuffle(buffer_size=1000,reshuffle_each_iteration=True)
    
    batched_dataset = batched_dataset.batch(batch_size)
    return batched_dataset, total_samples



def load3(path, batch_size,leader='II', input_shape=3, train=False):
    """
    :param path: 数据的路径
    :param batch_size: 批量大小
    :param train:是否是训练
    :return: tf.dataset
    """
    
    sub=''
    if train == True:
        sub=''
    
    timesteps = 125
    n_features= 16

    # 以下是不全是N的数据
    # path = '/deeplearn_data2/experimental_data/191204/fragment_not_all_N'  # 全部是8个文件
    # 基础数据，片段ID+8sI,II,V1,V5 ，总计8001个float32
    data_path = os.path.join(path, 'fragment_datas',sub+'data*')
    # 心搏数据，片段ID+心搏数量+心搏类型（12分类）+padding(-1),最长40个数，总计42个数
    heart_beat_path = os.path.join(path, 'fragment_labels',sub+'label*')
    # 干扰数据，片段ID+I导干扰段数+I导干扰起止段点+pading[-1]最长40个数20对，所以总点数是1+41*4=165
    #gr_info_path = os.path.join(path, 'fragment_Gr_segments',sub+'segment*')
    # R点位置，片段ID+40个小格[-1,-1,...,位置,...]这种的非零数，总数41
    #rpos_info_path = os.path.join(path, 'attr_pos_' + str(timesteps),sub+'attr_position*')
    # R点掩码，片段ID+40格，有R点的是1，无R点的是0，总数41
    rmask_info_path = os.path.join(path, 'attr_mask_' + str(timesteps),sub+'attr_mask*')
    # 40个小格分类，片段ID+40个小格各个小格的,分类，总计41
    rclass_info_path = os.path.join(path, 'attr_label_' + str(timesteps),sub+'attr_label*')
    # 获取文件
    data_files = sorted(tf.io.gfile.glob(data_path))
    heart_beat_files = sorted(tf.io.gfile.glob(heart_beat_path))
    #gr_info_files = sorted(tf.io.gfile.glob(gr_info_path))
    #rpos_info_files = sorted(tf.io.gfile.glob(rpos_info_path))
    rmask_info_files = sorted(tf.io.gfile.glob(rmask_info_path))
    rclass_info_files = sorted(tf.io.gfile.glob(rclass_info_path))

    
    def print_info(_path, data_files, size):
        print("[INFO] files:", _path, data_files)
        if len(data_files) < 1:
            print('[ERROR] No train data files found in %s' % _path)
            exit(-1)
        total_samples = sum([os.path.getsize(f) / 4 / size for f in data_files])
        return total_samples

    total_data_size = print_info(data_path, data_files, 8001)
    total_heart_beat_size = print_info(heart_beat_path, heart_beat_files, timesteps +2)
    #total_gr_size = print_info(gr_info_path, gr_info_files, 165)
    #total_rpos_size = print_info(rpos_info_path,rpos_info_files, timesteps+1)
    total_rmask_size = print_info(rmask_info_path, rmask_info_files, timesteps+1)
    total_rclass_size = print_info(rclass_info_path,rclass_info_files, timesteps+1)
    #assert total_data_size == total_heart_beat_size == total_gr_size == total_rpos_size == total_rmask_size == total_rclass_size
    # 生成dataset
    data_dataset = tf.data.FixedLengthRecordDataset(filenames = data_files, record_bytes = 4 * 8001)
    heart_beat_dataset = tf.data.FixedLengthRecordDataset(filenames = heart_beat_files, record_bytes = 4 * (timesteps+2))
    #gr_info_dataset = tf.data.FixedLengthRecordDataset(filenames = gr_info_files, record_bytes = 4 * 165)
    #rpos_info_dataset = tf.data.FixedLengthRecordDataset(filenames = rpos_info_files, record_bytes = 4 * (timesteps+1))
    rmask_info_dataset = tf.data.FixedLengthRecordDataset(filenames = rmask_info_files, record_bytes = 4 * (timesteps+1))
    rclass_info_dataset = tf.data.FixedLengthRecordDataset(filenames = rclass_info_files, record_bytes = 4 * (timesteps+1))


    def transfer_dataset(value):
        raw_data = tf.io.decode_raw(value, tf.float32)
        id = tf.cast(tf.slice(raw_data, [0], [1]), tf.int 32)
        data = tf.reshape(tf.slice(raw_data, [1], [8000]), (4, 2000, 1))  # 四个导联的数据

        if leader.lower()=='i':
            return id,data[0,...] #shape=(2000,1)
        elif leader.lower()=='ii':
            return id, data[1,...]
        elif leader.lower()=='v1':
            return id, data[2, ...]
        elif leader.lower()=='v5':
            return id, data[3, ...]
        elif leader.lower()=='all':
            return id, data #shape=(4,2000,1)


    def transfer_heart_beat(value):
        raw_data = tf.io.decode_raw(value, tf.float32)
        id = tf.cast(tf.slice(raw_data, [0], [1]), tf.int32)
        heart_beat_num = tf.cast(tf.slice(raw_data, [1], [1]), tf.int32)
        heart_beat = tf.slice(raw_data, [2], [40])  # 心搏分类
        return id, heart_beat_num, heart_beat

    def transfer_gr_info(value):
        raw_data = tf.io.decode_raw(value, tf.float32)
        id = tf.cast(tf.slice(raw_data, [0], [1]), tf.int32)
        # 四个导联各导联干扰情况,四行分别写四个导的干扰段数以及起始点
        gr_data = tf.reshape(tf.cast(tf.slice(raw_data, [1], [164]), tf.int32), (4, timesteps+1))  # 四个导联各导联干扰情况
        return id, gr_data

    def transfer_rpos_info(value):
        raw_data = tf.io.decode_raw(value, tf.float32)
        id = tf.cast(tf.slice(raw_data, [0], [1]), tf.int32)
        # 40个格，在有R点的小格写入r点位置（0-1999），其它的写-1
        rpos_data = tf.cast(tf.slice(raw_data, [1], [timesteps]), tf.int32)
        return id, rpos_data

    def transfer_rmask_info(value):
        raw_data = tf.io.decode_raw(value, tf.float32)
        id = tf.cast(tf.slice(raw_data, [0], [1]), tf.int32)
        # 40个格，在有R点的小格写入1,其它是0
        rmask_data = tf.cast(tf.slice(raw_data, [1], [timesteps]), tf.int32)
        return id, rmask_data

    def transfer_rclass_info(value):
        raw_data = tf.io.decode_raw(value, tf.float32)
        id = tf.cast(tf.slice(raw_data, [0], [1]), tf.int32)
        # 40个格，每个小格的分类
        rclass_data = tf.cast(tf.slice(raw_data, [1], [timesteps]), tf.int32)
        return id, rclass_data

    data_dataset = data_dataset.map(transfer_dataset,num_parallel_calls = tf.data.experimental.AUTOTUNE)
    heart_beat_dataset = heart_beat_dataset.map(transfer_heart_beat,num_parallel_calls = tf.data.experimental.AUTOTUNE)
    #gr_info_dataset = gr_info_dataset.map(transfer_gr_info,num_parallel_calls = tf.data.experimental.AUTOTUNE)
    #rpos_info_dataset = rpos_info_dataset.map(transfer_rpos_info,num_parallel_calls = tf.data.experimental.AUTOTUNE)
    rmask_info_dataset = rmask_info_dataset.map(transfer_rmask_info,num_parallel_calls = tf.data.experimental.AUTOTUNE)
    rclass_info_dataset = rclass_info_dataset.map(transfer_rclass_info,num_parallel_calls = tf.data.experimental.AUTOTUNE)

    #dataset = tf.data.Dataset.zip((data_dataset,heart_beat_dataset,gr_info_dataset,rpos_info_dataset,
    #                               rmask_info_dataset,rclass_info_dataset))#获取所有的数据和数据信息
    dataset = tf.data.Dataset.zip((data_dataset,heart_beat_dataset,
                                   rmask_info_dataset,rclass_info_dataset))#获取所有的数据和数据信息

    def all_transfer(data_dataset,heart_beat_dataset,rmask_info_dataset,rclass_info_dataset):
        id=data_dataset[0]
        data=data_dataset[1]
        label=rclass_info_dataset[1]
        mask=rmask_info_dataset[1]
        heart_beat=heart_beat_dataset[2]
        #return id,data,label,mask,heart_beat
        
        if input_shape==5:
            data = tf.reshape(data, [timesteps,n_features,1,1])
            label = tf.reshape(label,[timesteps,1,1,1])
        elif input_shape==4:
            data = tf.reshape(data, [timesteps,n_features,1])
            label = tf.reshape(label,[timesteps,1,1])
        elif input_shape==3:
            data = tf.reshape(data, [timesteps,n_features])
            label = tf.reshape(label,[timesteps,1])
        else:
            print('[ERROR] invalid input_shape %s' % input_shape)
            exit(-1)
        
        return id,data,label,mask,heart_beat
        
    dataset=dataset.map(all_transfer,num_parallel_calls = tf.data.experimental.AUTOTUNE)

    if train:
        dataset=dataset.shuffle(buffer_size = 10000,
         reshuffle_each_iteration = True).batch(batch_size,drop_remainder=True).prefetch(buffer_size = tf.data.experimental.AUTOTUNE).repeat(1)
    else:
        dataset=dataset.shuffle(buffer_size = 1000,
         reshuffle_each_iteration = True).batch(batch_size, drop_remainder=True).prefetch(buffer_size = tf.data.experimental.AUTOTUNE).repeat(1)

    return dataset,total_data_size


def load_II(path, batch_size,train=False,num=1):
    """
    :param path: 数据的路径
    :param batch_size: 批量大小
    :param train:是否是训练
    :return: tf.dataset
    """
    
    paths=['/NonPureN/']
    if num>0:
        paths=['/PureN','/NonPureN/']
    
    paths = [path + p for p in paths]
    print('load_II。。。。',path)
    all_data_files=[]
    all_heart_beat_files=[]
    #all_gr_info_files=[]
    all_rpos_info_files=[]
    all_rmask_info_files=[]
    all_rclass_info_files=[]
    for i,path in enumerate(paths):
        # 以下是不全是N的数据
        # path = '/deeplearn_data2/experimental_data/191210/Channel_II/NotContainNoise/NonPureN/Train'  # 全部是8个文件
        # 基础数据，片段ID+8sII ，总计2001个float32
        data_path = os.path.join(path, 'Raw/*')
        # 心搏数据，片段ID+心搏数量+心搏类型（12分类）+padding(-1),最长40个数，总计42个数
        heart_beat_path = os.path.join(path, 'Attr/hb_labels*')
        # 干扰数据，片段ID+II导干扰掩码1正常0干扰，所以总点数是1+2000=2001
        gr_info_path = os.path.join(path, 'Attr/hb_noise*')
        # R点位置，片段ID+40个小格[-1,-1,...,位置,...]这种的非零数，总数41
        rpos_info_path = os.path.join(path, 'Attr/Segment40/seq40_RR*')
        # R点掩码，片段ID+40格，有R点的是1，无R点的是0，总数41
        rmask_info_path = os.path.join(path, 'Attr/Segment40/seq40_mask*')
        # 40个小格分类，片段ID+40个小格各个小格的,分类，总计41
        rclass_info_path = os.path.join(path, 'Attr/Segment40/seq40_labels*')
        # 获取文件
        data_files = sorted(tf.io.gfile.glob(data_path))
        heart_beat_files = sorted(tf.io.gfile.glob(heart_beat_path))
        #gr_info_files = sorted(tf.io.gfile.glob(gr_info_path))
        rpos_info_files = sorted(tf.io.gfile.glob(rpos_info_path))
        rmask_info_files = sorted(tf.io.gfile.glob(rmask_info_path))
        rclass_info_files = sorted(tf.io.gfile.glob(rclass_info_path))
        if i==1:
            data_files=data_files[0:num]
            heart_beat_files=heart_beat_files[0:num]
            #gr_info_files=gr_info_files[0:num]
            rpos_info_files=rpos_info_files[0:num]
            rmask_info_files=rmask_info_files[0:num]
            rclass_info_files=rclass_info_files[0:num]
        all_data_files.extend(data_files)
        all_heart_beat_files.extend(heart_beat_files)
        #all_gr_info_files.extend(rpos_info_files)
        all_rpos_info_files.extend(rpos_info_files)
        all_rmask_info_files.extend(rmask_info_files)
        all_rclass_info_files.extend(rclass_info_files)

    def print_info(data_files, size):
        print("[INFO] files:", data_files)
        if len(data_files) < 1:
            print('[ERROR] No train data files found in %s' % path)
            exit(-1)
        total_samples = sum([os.path.getsize(f) / 4 / size for f in data_files])
        return total_samples

    total_data_size = print_info(all_data_files, 2001)
    total_heart_beat_size = print_info(all_heart_beat_files, 42)
    #total_gr_size = print_info(all_gr_info_files, 2001)
    total_rpos_size = print_info(all_rpos_info_files, 41)
    total_rmask_size = print_info(all_rmask_info_files, 41)
    total_rclass_size = print_info(all_rclass_info_files, 41)
    assert total_data_size == total_heart_beat_size == total_rpos_size == total_rmask_size == total_rclass_size
    # 生成dataset
    data_dataset = tf.data.FixedLengthRecordDataset(filenames = all_data_files, record_bytes = 4 * 2001)
    heart_beat_dataset = tf.data.FixedLengthRecordDataset(filenames = all_heart_beat_files, record_bytes = 4 * 42)
    #gr_info_dataset = tf.data.FixedLengthRecordDataset(filenames = all_gr_info_files, record_bytes = 4 * 2001)
    rpos_info_dataset = tf.data.FixedLengthRecordDataset(filenames = all_rpos_info_files, record_bytes = 4 * 41)
    rmask_info_dataset = tf.data.FixedLengthRecordDataset(filenames = all_rmask_info_files, record_bytes = 4 * 41)
    rclass_info_dataset = tf.data.FixedLengthRecordDataset(filenames = all_rclass_info_files, record_bytes = 4 * 41)


    def transfer_dataset(value):
        raw_data = tf.io.decode_raw(value, tf.float32)
        id = tf.cast(tf.slice(raw_data, [0], [1]), tf.int32)
        data = tf.reshape(tf.slice(raw_data, [1], [2000]), (1, 2000, 1))  # 四个导联的数据


        return id, data #shape=(1,2000,1)


    def transfer_heart_beat(value):
        raw_data = tf.io.decode_raw(value, tf.float32)
        id = tf.cast(tf.slice(raw_data, [0], [1]), tf.int32)
        heart_beat_num = tf.cast(tf.slice(raw_data, [1], [1]), tf.int32)
        heart_beat = tf.cast(tf.slice(raw_data, [2], [40]),tf.int32)  # 心搏分类

        # heart_beat = tf.reshape(tf.gather(heart_beat, tf.where(tf.math.greater(heart_beat, tf.constant(0)))), [-1])
        # heart_label = tf.reshape(tf.gather(heart_beat, tf.where(tf.math.greater(heart_beat, tf.constant(0)))), [-1])
        # negative_value=tf.reshape(tf.gather(heart_beat, tf.where(tf.math.less(heart_beat, tf.constant(0)))), [-1])
        # heart_beat = tf.concat((tf.constant([12]), heart_label, tf.constant([13]),negative_value),axis = -1)
        return id, heart_beat_num, heart_beat

    def transfer_gr_info(value):
        raw_data = tf.io.decode_raw(value, tf.float32)
        id = tf.cast(tf.slice(raw_data, [0], [1]), tf.int32)
        # 四个导联各导联干扰情况,四行分别写四个导的干扰段数以及起始点
        gr_data = tf.cast(tf.slice(raw_data, [1], [2000]), tf.int32) # 四个导联各导联干扰情况
        return id, gr_data

    def transfer_rpos_info(value):
        raw_data = tf.io.decode_raw(value, tf.float32)
        id = tf.cast(tf.slice(raw_data, [0], [1]), tf.int32)
        # 40个格，在有R点的小格写入r点位置（0-1999），其它的写-1
        rpos_data = tf.cast(tf.slice(raw_data, [1], [40]), tf.int32)
        return id, rpos_data

    def transfer_rmask_info(value):
        raw_data = tf.io.decode_raw(value, tf.float32)
        id = tf.cast(tf.slice(raw_data, [0], [1]), tf.int32)
        # 40个格，在有R点的小格写入1,其它是0
        rmask_data = tf.cast(tf.slice(raw_data, [1], [40]), tf.int32)
        return id, rmask_data

    def transfer_rclass_info(value):
        raw_data = tf.io.decode_raw(value, tf.float32)
        id = tf.cast(tf.slice(raw_data, [0], [1]), tf.int32)
        # 40个格，每个小格的分类
        rclass_data = tf.cast(tf.slice(raw_data, [1], [40]), tf.int32)
        # rclass_data=tf.concat(([12], rclass_data, [13]), axis = -1)
        return id, rclass_data

    data_dataset = data_dataset.map(transfer_dataset,num_parallel_calls = tf.data.experimental.AUTOTUNE)
    heart_beat_dataset = heart_beat_dataset.map(transfer_heart_beat,num_parallel_calls = tf.data.experimental.AUTOTUNE)
    #gr_info_dataset = gr_info_dataset.map(transfer_gr_info,num_parallel_calls = tf.data.experimental.AUTOTUNE)
    rpos_info_dataset = rpos_info_dataset.map(transfer_rpos_info,num_parallel_calls = tf.data.experimental.AUTOTUNE)
    rmask_info_dataset = rmask_info_dataset.map(transfer_rmask_info,num_parallel_calls = tf.data.experimental.AUTOTUNE)
    rclass_info_dataset = rclass_info_dataset.map(transfer_rclass_info,num_parallel_calls = tf.data.experimental.AUTOTUNE)

    dataset = tf.data.Dataset.zip((data_dataset,heart_beat_dataset,rpos_info_dataset,
                                   rmask_info_dataset,rclass_info_dataset))#获取所有的数据和数据信息


    def all_transfer(data_dataset,heart_beat_dataset,rpos_info_dataset,rmask_info_dataset,rclass_info_dataset):
        id=data_dataset[0]
        data=data_dataset[1]
        label=rclass_info_dataset[1]
        mask=rmask_info_dataset[1]
        heart_beat=heart_beat_dataset[2]
        heart_beat_nums=heart_beat_dataset[1]
        rpos_info = rpos_info_dataset[1]
        # dataset=DS(id=id,features=data,labels=label,hb_labels = heart_beat,hb_nums =heart_beat_nums,rmask=mask )
        # dataset={'features':data,'labels':label,'hb_labels':heart_beat}
        # dataset={'inputs':data,'outputs':label}

        return id,data,label,mask,heart_beat,heart_beat_nums,rpos_info

    dataset=dataset.map(all_transfer,num_parallel_calls = tf.data.experimental.AUTOTUNE)

    if train:
        dataset=dataset.shuffle(buffer_size = 10000,
         reshuffle_each_iteration = True).batch(batch_size, drop_remainder=True).prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
    else:
        dataset=dataset.shuffle(buffer_size = 10000,
         reshuffle_each_iteration = True).batch(batch_size,drop_remainder=True).prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

    return dataset,total_data_size
    



def load_new(path, batch_size,steps=40,train=False,num=1):
    """
    :param path: 数据的路径
    :param batch_size: 批量大小
    :param train:是否是训练
    :return: tf.dataset
    """
    
    paths=['/NonPureN/']
    if num>0:
        paths=['/NonPureN','/PureN']
    
    paths = [path + p for p in paths]
    
    print('load_new...',paths)
    
    all_data_files=[]
    all_heart_beat_files=[]
    #all_gr_info_files=[]
    all_rpos_info_files=[]
    all_rmask_info_files=[]
    all_rclass_info_files=[]
    for i,path in enumerate(paths):
        # 以下是不全是N的数据
        # path = '/deeplearn_data2/experimental_data/191210/Channel_II/NotContainNoise/NonPureN/Train'  # 全部是8个文件
        # 基础数据，片段ID+8sII ，总计2001个float32
        data_path = os.path.join(path, 'Raw/*')
        # 心搏数据，片段ID+心搏数量+心搏类型（12分类）+padding(-1),最长40个数，总计42个数
        heart_beat_path = os.path.join(path, 'Attr/hb_labels*')
        # 干扰数据，片段ID+II导干扰掩码1正常0干扰，所以总点数是1+2000=2001
        gr_info_path = os.path.join(path, 'Attr/hb_noise*')
        if steps==40:
            # R点位置，片段ID+40个小格[-1,-1,...,位置,...]这种的非零数，总数41
            rpos_info_path = os.path.join(path, 'Attr/Segment40/seq40_RR*')
            # R点掩码，片段ID+40格，有R点的是1，无R点的是0，总数41
            rmask_info_path = os.path.join(path, 'Attr/Segment40/seq40_mask*')
            # 40个小格分类，片段ID+40个小格各个小格的,分类，总计41
            rclass_info_path = os.path.join(path, 'Attr/Segment40/seq40_labels*')
        else:
            # R点位置，片段ID+40个小格[-1,-1,...,位置,...]这种的非零数，总数41
            rpos_info_path = os.path.join(path, 'Attr/Segment{}/seq{}_RR*'.format(steps,steps))
            # R点掩码，片段ID+40格，有R点的是1，无R点的是0，总数41
            rmask_info_path = os.path.join(path, 'Attr/Segment{}/seq{}_mask*'.format(steps,steps))
            # 40个小格分类，片段ID+40个小格各个小格的,分类，总计41
            rclass_info_path = os.path.join(path, 'Attr/Segment{}/seq{}_labels*'.format(steps,steps))
        
        # 获取文件
        data_files = sorted(tf.io.gfile.glob(data_path))
        heart_beat_files = sorted(tf.io.gfile.glob(heart_beat_path))
        #gr_info_files = sorted(tf.io.gfile.glob(gr_info_path))
        rpos_info_files = sorted(tf.io.gfile.glob(rpos_info_path))
        rmask_info_files = sorted(tf.io.gfile.glob(rmask_info_path))
        rclass_info_files = sorted(tf.io.gfile.glob(rclass_info_path))
        if i==1:
            data_files=data_files[0:num]
            heart_beat_files=heart_beat_files[0:num]
            #gr_info_files=gr_info_files[0:num]
            rpos_info_files=rpos_info_files[0:num]
            rmask_info_files=rmask_info_files[0:num]
            rclass_info_files=rclass_info_files[0:num]
        all_data_files.extend(data_files)
        all_heart_beat_files.extend(heart_beat_files)
        #all_gr_info_files.extend(rpos_info_files)
        all_rpos_info_files.extend(rpos_info_files)
        all_rmask_info_files.extend(rmask_info_files)
        all_rclass_info_files.extend(rclass_info_files)

    def print_info(data_files, size):
        print("[INFO] files:", data_files)
        if len(data_files) < 1:
            print('[ERROR] No train data files found in %s' % path)
            exit(-1)
        total_samples = sum([os.path.getsize(f) / 4 / size for f in data_files])
        return total_samples

    total_data_size = print_info(all_data_files, 2001)
    total_heart_beat_size = print_info(all_heart_beat_files, steps+2)
    #total_gr_size = print_info(all_gr_info_files, 2001)
    total_rpos_size = print_info(all_rpos_info_files, steps+1)
    total_rmask_size = print_info(all_rmask_info_files, steps+1)
    total_rclass_size = print_info(all_rclass_info_files, steps+1)
    print('total_data_size ={} total_heart_beat_size ={} total_rpos_size ={} total_rmask_size ={} total_rclass_size ={}'.
    format(total_data_size,total_heart_beat_size,total_rpos_size,total_rmask_size,total_rclass_size))
    
    #assert total_data_size == total_heart_beat_size == total_rpos_size == total_rmask_size == total_rclass_size
    # 生成dataset
    data_dataset = tf.data.FixedLengthRecordDataset(filenames = all_data_files, record_bytes = 4 * 2001)
    heart_beat_dataset = tf.data.FixedLengthRecordDataset(filenames = all_heart_beat_files, record_bytes = 4 * (40+2))
    #gr_info_dataset = tf.data.FixedLengthRecordDataset(filenames = all_gr_info_files, record_bytes = 4 * 2001)
    rpos_info_dataset = tf.data.FixedLengthRecordDataset(filenames = all_rpos_info_files, record_bytes = 4 * (steps+1))
    rmask_info_dataset = tf.data.FixedLengthRecordDataset(filenames = all_rmask_info_files, record_bytes = 4 * (steps+1))
    rclass_info_dataset = tf.data.FixedLengthRecordDataset(filenames = all_rclass_info_files, record_bytes = 4 * (steps+1))


    def transfer_dataset(value):
        raw_data = tf.io.decode_raw(value, tf.float32)
        id = tf.cast(tf.slice(raw_data, [0], [1]), tf.int32)
        data = tf.reshape(tf.slice(raw_data, [1], [2000]), (1, 2000, 1))  # 四个导联的数据


        return id, data #shape=(1,2000,1)


    def transfer_heart_beat(value):
        raw_data = tf.io.decode_raw(value, tf.float32)
        id = tf.cast(tf.slice(raw_data, [0], [1]), tf.int32)
        heart_beat_num = tf.cast(tf.slice(raw_data, [1], [1]), tf.int32)
        heart_beat = tf.cast(tf.slice(raw_data, [2], [40]),tf.int32)  # 心搏分类

        # heart_beat = tf.reshape(tf.gather(heart_beat, tf.where(tf.math.greater(heart_beat, tf.constant(0)))), [-1])
        # heart_label = tf.reshape(tf.gather(heart_beat, tf.where(tf.math.greater(heart_beat, tf.constant(0)))), [-1])
        # negative_value=tf.reshape(tf.gather(heart_beat, tf.where(tf.math.less(heart_beat, tf.constant(0)))), [-1])
        # heart_beat = tf.concat((tf.constant([12]), heart_label, tf.constant([13]),negative_value),axis = -1)
        return id, heart_beat_num, heart_beat

    def transfer_gr_info(value):
        raw_data = tf.io.decode_raw(value, tf.float32)
        id = tf.cast(tf.slice(raw_data, [0], [1]), tf.int32)
        # 四个导联各导联干扰情况,四行分别写四个导的干扰段数以及起始点
        gr_data = tf.cast(tf.slice(raw_data, [1], [2000]), tf.int32) # 四个导联各导联干扰情况
        return id, gr_data

    def transfer_rpos_info(value):
        raw_data = tf.io.decode_raw(value, tf.float32)
        id = tf.cast(tf.slice(raw_data, [0], [1]), tf.int32)
        # 40个格，在有R点的小格写入r点位置（0-1999），其它的写-1
        rpos_data = tf.cast(tf.slice(raw_data, [1], [steps]), tf.int32)
        return id, rpos_data

    def transfer_rmask_info(value):
        raw_data = tf.io.decode_raw(value, tf.float32)
        id = tf.cast(tf.slice(raw_data, [0], [1]), tf.int32)
        # 40个格，在有R点的小格写入1,其它是0
        rmask_data = tf.cast(tf.slice(raw_data, [1], [steps]), tf.int32)
        return id, rmask_data

    def transfer_rclass_info(value):
        raw_data = tf.io.decode_raw(value, tf.float32)
        id = tf.cast(tf.slice(raw_data, [0], [1]), tf.int32)
        # 40个格，每个小格的分类
        rclass_data = tf.cast(tf.slice(raw_data, [1], [steps]), tf.int32)
        # rclass_data=tf.concat(([12], rclass_data, [13]), axis = -1)
        return id, rclass_data

    data_dataset = data_dataset.map(transfer_dataset,num_parallel_calls = tf.data.experimental.AUTOTUNE)
    heart_beat_dataset = heart_beat_dataset.map(transfer_heart_beat,num_parallel_calls = tf.data.experimental.AUTOTUNE)
    #gr_info_dataset = gr_info_dataset.map(transfer_gr_info,num_parallel_calls = tf.data.experimental.AUTOTUNE)
    rpos_info_dataset = rpos_info_dataset.map(transfer_rpos_info,num_parallel_calls = tf.data.experimental.AUTOTUNE)
    rmask_info_dataset = rmask_info_dataset.map(transfer_rmask_info,num_parallel_calls = tf.data.experimental.AUTOTUNE)
    rclass_info_dataset = rclass_info_dataset.map(transfer_rclass_info,num_parallel_calls = tf.data.experimental.AUTOTUNE)

    dataset = tf.data.Dataset.zip((data_dataset,heart_beat_dataset,rpos_info_dataset,
                                   rmask_info_dataset,rclass_info_dataset))#获取所有的数据和数据信息

    def all_transfer(data_dataset,heart_beat_dataset,rpos_info_dataset,rmask_info_dataset,rclass_info_dataset):
        id=data_dataset[0]
        data=data_dataset[1]
        label=rclass_info_dataset[1]
        mask=rmask_info_dataset[1]
        heart_beat=heart_beat_dataset[2]
        heart_beat_nums=heart_beat_dataset[1]
        rpos_info = rpos_info_dataset[1]
        # dataset=DS(id=id,features=data,labels=label,hb_labels = heart_beat,hb_nums =heart_beat_nums,rmask=mask )
        # dataset={'features':data,'labels':label,'hb_labels':heart_beat}
        # dataset={'inputs':data,'outputs':label}

        #heart beat 125 not right
        #return id,data,label,mask,heart_beat,heart_beat_nums,rpos_info
        return id,data,label,mask,heart_beat,heart_beat_nums,rpos_info

    dataset=dataset.map(all_transfer,num_parallel_calls = tf.data.experimental.AUTOTUNE)

    if train:
        dataset=dataset.shuffle(buffer_size = 10000,
         reshuffle_each_iteration = True).batch(batch_size, drop_remainder=True).prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
    else:
        dataset=dataset.shuffle(buffer_size = 10000,
         reshuffle_each_iteration = True).batch(batch_size,drop_remainder=True).prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

    return dataset,total_data_size
    


def load_train(path, batch_size,weights=[0.9,0.1]):
    """
    :param path: 数据的路径[not_pure_N_path,Pure_N_path]
    :param batch_size: 批量大小
    :param train:是否是训练
    :return: tf.dataset
    """
    
    paths=['/NonPureN','/PureN']
    
    paths = [path + p for p in paths]
    
    def read_one_dataset(path):
        data_path = os.path.join(path, 'Raw/*')
        # 心搏数据，片段ID+心搏数量+心搏类型（12分类）+padding(-1),最长40个数，总计42个数
        heart_beat_path = os.path.join(path, 'Attr/hb_labels*')
        # 干扰数据，片段ID+II导干扰掩码1正常0干扰，所以总点数是1+2000=2001
        #gr_info_path = os.path.join(path, 'Attr/hb_noise*')
        # R点位置，片段ID+40个小格[-1,-1,...,位置,...]这种的非零数，总数41
        rpos_info_path = os.path.join(path, 'Attr/Segment40/seq40_RR*')
        # R点掩码，片段ID+40格，有R点的是1，无R点的是0，总数41
        rmask_info_path = os.path.join(path, 'Attr/Segment40/seq40_mask*')
        # 40个小格分类，片段ID+40个小格各个小格的,分类，总计41
        rclass_info_path = os.path.join(path, 'Attr/Segment40/seq40_labels*')
        # 获取文件
        data_files = sorted(tf.io.gfile.glob(data_path))
        heart_beat_files = sorted(tf.io.gfile.glob(heart_beat_path))
        #gr_info_files = sorted(tf.io.gfile.glob(gr_info_path))
        rpos_info_files = sorted(tf.io.gfile.glob(rpos_info_path))
        rmask_info_files = sorted(tf.io.gfile.glob(rmask_info_path))
        rclass_info_files = sorted(tf.io.gfile.glob(rclass_info_path))
            


        def print_info(files, size):
            print("[INFO] files:", files)
            if len(data_files) < 1:
                print('[ERROR] No train data files found in %s' % path)
                exit(-1)
            total_samples = sum([os.path.getsize(f) / 4 / size for f in files])
            return total_samples

        total_data_size = print_info(data_files, 2001)
        total_heart_beat_size = print_info(heart_beat_files, 42)
        #total_gr_size = print_info(gr_info_files, 2001)
        total_rpos_size = print_info(rpos_info_files, 41)
        total_rmask_size = print_info(rmask_info_files, 41)
        total_rclass_size = print_info(rclass_info_files, 41)
        assert total_data_size == total_heart_beat_size == total_rpos_size == total_rmask_size == total_rclass_size
        # 生成dataset
        data_dataset = tf.data.FixedLengthRecordDataset(filenames = data_files, record_bytes = 4 * 2001)
        heart_beat_dataset = tf.data.FixedLengthRecordDataset(filenames = heart_beat_files, record_bytes = 4 * 42)
        #gr_info_dataset = tf.data.FixedLengthRecordDataset(filenames = gr_info_files, record_bytes = 4 * 2001)
        rpos_info_dataset = tf.data.FixedLengthRecordDataset(filenames = rpos_info_files, record_bytes = 4 * 41)
        rmask_info_dataset = tf.data.FixedLengthRecordDataset(filenames = rmask_info_files, record_bytes = 4 * 41)
        rclass_info_dataset = tf.data.FixedLengthRecordDataset(filenames = rclass_info_files, record_bytes = 4 * 41)


        def transfer_dataset(value):
            raw_data = tf.io.decode_raw(value, tf.float32)
            id = tf.cast(tf.slice(raw_data, [0], [1]), tf.int32)
            data = tf.reshape(tf.slice(raw_data, [1], [2000]), (1, 2000, 1))  # 四个导联的数据


            return id, data #shape=(1,2000,1)


        def transfer_heart_beat(value):
            raw_data = tf.io.decode_raw(value, tf.float32)
            id = tf.cast(tf.slice(raw_data, [0], [1]), tf.int32)
            heart_beat_num = tf.cast(tf.slice(raw_data, [1], [1]), tf.int32)
            heart_beat = tf.cast(tf.slice(raw_data, [2], [40]),tf.int32)  # 心搏分类

            # heart_beat = tf.reshape(tf.gather(heart_beat, tf.where(tf.math.greater(heart_beat, tf.constant(0)))), [-1])
            # heart_label = tf.reshape(tf.gather(heart_beat, tf.where(tf.math.greater(heart_beat, tf.constant(0)))), [-1])
            # negative_value=tf.reshape(tf.gather(heart_beat, tf.where(tf.math.less(heart_beat, tf.constant(0)))), [-1])
            # heart_beat = tf.concat((tf.constant([12]), heart_label, tf.constant([13]),negative_value),axis = -1)
            return id, heart_beat_num, heart_beat

        def transfer_gr_info(value):
            raw_data = tf.io.decode_raw(value, tf.float32)
            id = tf.cast(tf.slice(raw_data, [0], [1]), tf.int32)
            # 四个导联各导联干扰情况,四行分别写四个导的干扰段数以及起始点
            gr_data = tf.cast(tf.slice(raw_data, [1], [2000]), tf.int32) # 四个导联各导联干扰情况
            return id, gr_data

        def transfer_rpos_info(value):
            raw_data = tf.io.decode_raw(value, tf.float32)
            id = tf.cast(tf.slice(raw_data, [0], [1]), tf.int32)
            # 40个格，在有R点的小格写入r点位置（0-1999），其它的写-1
            rpos_data = tf.cast(tf.slice(raw_data, [1], [40]), tf.int32)
            return id, rpos_data

        def transfer_rmask_info(value):
            raw_data = tf.io.decode_raw(value, tf.float32)
            id = tf.cast(tf.slice(raw_data, [0], [1]), tf.int32)
            # 40个格，在有R点的小格写入1,其它是0
            rmask_data = tf.cast(tf.slice(raw_data, [1], [40]), tf.int32)
            return id, rmask_data

        def transfer_rclass_info(value):
            raw_data = tf.io.decode_raw(value, tf.float32)
            id = tf.cast(tf.slice(raw_data, [0], [1]), tf.int32)
            # 40个格，每个小格的分类
            rclass_data = tf.cast(tf.slice(raw_data, [1], [40]), tf.int32)
            # rclass_data=tf.concat(([12], rclass_data, [13]), axis = -1)
            return id, rclass_data

        data_dataset = data_dataset.map(transfer_dataset,num_parallel_calls = tf.data.experimental.AUTOTUNE)
        heart_beat_dataset = heart_beat_dataset.map(transfer_heart_beat,num_parallel_calls = tf.data.experimental.AUTOTUNE)
        #gr_info_dataset = gr_info_dataset.map(transfer_gr_info,num_parallel_calls = tf.data.experimental.AUTOTUNE)
        rpos_info_dataset = rpos_info_dataset.map(transfer_rpos_info,num_parallel_calls = tf.data.experimental.AUTOTUNE)
        rmask_info_dataset = rmask_info_dataset.map(transfer_rmask_info,num_parallel_calls = tf.data.experimental.AUTOTUNE)
        rclass_info_dataset = rclass_info_dataset.map(transfer_rclass_info,num_parallel_calls = tf.data.experimental.AUTOTUNE)

        dataset = tf.data.Dataset.zip((data_dataset,heart_beat_dataset,rpos_info_dataset,
                                       rmask_info_dataset,rclass_info_dataset))#获取所有的数据和数据信息


        def all_transfer(data_dataset,heart_beat_dataset,rpos_info_dataset,rmask_info_dataset,rclass_info_dataset):
            id=data_dataset[0]
            data=data_dataset[1]
            label=rclass_info_dataset[1]
            mask=rmask_info_dataset[1]
            heart_beat=heart_beat_dataset[2]
            heart_beat_nums=heart_beat_dataset[1]
            rpos_info = rpos_info_dataset[1]
            
            # dataset=DS(id=id,features=data,labels=label,hb_labels = heart_beat,hb_nums =heart_beat_nums,rmask=mask )
            # dataset={'features':data,'labels':label,'hb_labels':heart_beat}
            # dataset={'inputs':data,'outputs':label}
            #dataset=tf.data.Dataset.zip((id,data,label,mask,heart_beat,heart_beat_nums))
            #return id,data,label,mask,heart_beat,heart_beat_nums
            return id,data,label,mask,heart_beat,heart_beat_nums,rpos_info
            
        dataset=dataset.map(all_transfer,num_parallel_calls = tf.data.experimental.AUTOTUNE)
        return dataset,total_data_size
    ds_NPN,NPN_data_size=read_one_dataset(paths[0])
    ds_PN,PN_data_size=read_one_dataset(paths[1])
    total_data_size=NPN_data_size+PN_data_size
    dataset = tf.data.experimental.sample_from_datasets([ds_NPN, ds_PN], weights=weights, seed=1234)
    dataset=dataset.shuffle(buffer_size = 100000,
         reshuffle_each_iteration = True).batch(batch_size,drop_remainder=True).prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
    

    return dataset,total_data_size

