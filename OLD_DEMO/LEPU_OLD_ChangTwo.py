import os
import time
import sys
import random
import numpy as np
from numpy import array
import csv

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import backend as backend
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix

import LEPU_OLD_EcgDataPipe
import ImageGen
import LEPU_OLD_EcgModels
import EcgLossFn
import EcgConfig


if len(sys.argv)<6:
    print('missing args:')
    print('for Infer/Trail: arg1=N/T,arg2=model_name,arg3=data_path,arg4=lead,arg5=timesteps')
    exit(-1)

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

tf.keras.backend.clear_session()

for_train = False
for_trial = True
for_lead_all = False

print_steps = 1000

classes=12
filters = 32
max_heartbeats=30
timesteps=125
n_features=16

flag=sys.argv[1]
model_name = sys.argv[2]
data_path=sys.argv[3]
lead=sys.argv[4]
timesteps=int(sys.argv[5])
if timesteps==40 or timesteps == 125:
    n_features = int(2000/timesteps)
else:
    print('invalid timesteps:', timesteps)
    exit(-1)
    
    
if lead=='i' or lead=='ii' or lead=='v1' or lead=='v5' or lead=='all':
    print('Lead is ok!')
else:
    print('[ERROR]', 'Invalid lead input!')
    exit(-1)
    
if flag == 'Y' or flag == 'y':
    if len(sys.argv)<7:
        print('missing arg6=epochs')
        print('for train: arg1=Y,arg2=model_name,arg3=data_path,arg4=lead,arg5=timesteps,arg6=epochs')
        exit(-1)

    for_train = True
    for_trial = False
    for_lead_all = False
    epochs=int(sys.argv[6])
    batch_size=64
    if data_path == 'mini3':
        print_steps = 100
        
    print('Configure for train lead ', lead)
    
else:

    if flag == 'T':    
        for_train = True
        for_trial = True
        epochs = 2
        batch_size = 16 
        print_steps = 100
        print('Configure for trail train wiht lead ', lead)
    else:
        for_train = False
        for_trial = False
        for_lead_all = False
        epochs = 1
        batch_size = 16
        print_steps = 100
        run_time  = ''

        search_path = os.path.join('models', model_name)
        for f in sorted(os.listdir(search_path),reverse=True):
            mfile = os.path.join('models', model_name, f,model_name + '_' + lead + '_' + str(timesteps) + '.h5')
            print(mfile)
            if os.path.isfile(mfile):
                run_time = f
                break        
        if run_time=='?':
            print('[ERROR] Cannot find saved model to run.')
            exit()    
        print('Configure for single lead test ', lead)

start=time.time()
if for_train == True:
    run_time = time.strftime('%Y%m%d-%H%M', time.localtime(start))

print(flag,model_name,data_path,lead,epochs,batch_size,run_time, timesteps,n_features)

ctc_encoder_length = tf.ones([batch_size]) * timesteps

train_data_path = os.path.join('data', data_path, 'train')
test_data_path = os.path.join('data', data_path, 'test')
result_path   = os.path.join('results', model_name,run_time) 
train_log_dir = os.path.join('logs',model_name,'train', run_time)
test_log_dir  = os.path.join('logs', model_name,'test', run_time)
model_path    = os.path.join('models', model_name, run_time) 
#model_file = os.path.join(model_path, model_name + '_' + lead + '.h5')
#model_image = os.path.join(model_path, model_name + '_' + lead + '.png')
result_file = os.path.join('results', model_name, run_time + '_' + lead + '_' + str(timesteps) + '.csv') 

if model_name=='two':
    model = LEPU_OLD_EcgModels.model_Two(filters, timesteps, n_features, classes)
    input_shape=3
elif model_name=='seq':
    model = LEPU_OLD_EcgModels.model_Seq(filters, timesteps, n_features, classes)
    input_shape=3
elif model_name=='seq2':
    model = LEPU_OLD_EcgModels.model_Seq2Seq(filters, 12, 12)
    input_shape=3
elif model_name=='tcp':
    model = LEPU_OLD_EcgModels.model_TwoCP(filters, timesteps, n_features, classes)
    input_shape=3
elif model_name=='1dlstm_d':
    model = LEPU_OLD_EcgModels.model_Cnn1DLSTM(filters, timesteps, n_features, classes)
    input_shape=3
elif model_name=='lstm':
    model = LEPU_OLD_EcgModels.model_CnnLSTM(filters, timesteps, n_features, classes)
    input_shape=3
elif model_name=='cnnlstm_d':
    model = LEPU_OLD_EcgModels.model_CnnLSTM_D1(filters, timesteps, n_features, classes)
    input_shape=4
elif model_name=='cnnlstm_d_two':
    model = LEPU_OLD_EcgModels.model_CnnLSTM_D1_Two(filters, timesteps, n_features, classes)
    input_shape=4
elif model_name=='cnnlstm_s':
    model = LEPU_OLD_EcgModels.model_CnnLSTM_S(filters, timesteps, n_features, classes)
    input_shape=4
elif model_name=='cnnlstm_4':  #4 leads train
    model = LEPU_OLD_EcgModels.model_CnnLSTM_4(filters, timesteps, n_features, classes)
    input_shape=4
elif model_name=='ecgseq':
    model = LEPU_OLD_EcgModels.model_EcgSeq1(filters, timesteps, n_features, classes)
    input_shape=3
elif model_name=='ecgatt':
    model = LEPU_OLD_EcgModels.model_EcgAtt(filters, timesteps, n_features, classes)
    input_shape=3
elif model_name=='ecgvgg':
    model = LEPU_OLD_EcgModels.model_VGG16(filters, timesteps, n_features, classes)
    input_shape=3
elif model_name=='vggatt':
    model = LEPU_OLD_EcgModels.model_VGGATT(filters, timesteps, n_features, classes)
    input_shape=3
else:
    print('invalid model name:', model_name)
    exit(-1)

if for_train:
    model.summary()

if os.path.exists(model_path)==False: 
    os.makedirs(model_path)
    print('Create model path:',model_path)

#plot_model(model, to_file= os.path.join(model_path, model_name+'.png'), show_shapes=True)
#f=os.path.join(model_path, model_name +'_' + lead + '_' + str(timesteps) + '.h5') 
#model.save(f)
#print('saving model to:',f)
#exit()

# 转换为40*800的numpy二维数组，padding值为：-1000
def transform_data(ecg_data, attr_rr_data):
    # 通过RR间期6:4获得各R点的起始与结束
    r_begin_and_end_list = get_begin_and_end_by_RR(attr_rr_data)
    _arr = np.zeros(40*800) - 1000
    arr = _arr.reshape(40, 800)
    for k, (r_begin, r_end) in enumerate(r_begin_and_end_list):
        r_len = (r_end - r_begin) if (r_end - r_begin) < 800 else 800
        arr[k, :r_len] = ecg_data[r_begin: r_begin+r_len]
    return arr

def dense_to_sparse(dense_tensor, sequence_length):
  indices = tf.where(tf.sequence_mask(sequence_length))
  values = tf.gather_nd(dense_tensor, indices)
  shape = tf.shape(dense_tensor, out_type=tf.int64)
  return tf.SparseTensor(indices, values, shape)

def mask_nans(x):
  x_zeros = tf.zeros_like(x)
  x_mask = tf.is_finite(x)
  y = tf.where(x_mask, x, x_zeros)
  return y
  
def get_begin_and_end_by_RR(rpos_data):
    #rpos_data = list(rpos_data)
    #print(rpos_data)
    #print(type(rpos_data))
    fragment_rpos = [rpos for rpos in rpos_data if rpos > -1]
    
    fragment_len = 2000
    split_pos_list = []
    for _ind, rpos in enumerate(fragment_rpos):
        if _ind == 0:
            continue
        pre_rpos = fragment_rpos[_ind - 1]
        rr_dif = rpos - pre_rpos
        #print(rr_dif, type(rr_dif))
        split_pos = int(tf.cast(pre_rpos,tf.float32) + tf.cast(rr_dif,tf.float32) * 0.6)
        split_pos_list.append(split_pos)
        if _ind == len(fragment_rpos) - 1:
            # 最后一个心搏的分割点为片段的尾点
            split_pos_list.append(fragment_len)
    r_begin_and_end_list = []
    for k, split_pos in enumerate(split_pos_list):
        if k == 0:
            pre_split_pos = 0
        else:
            pre_split_pos = split_pos_list[k-1]
        r_begin_and_end_list.append((pre_split_pos, split_pos))
    #print(r_begin_and_end_list)
    return r_begin_and_end_list

    
#train_the_model
def train_on_batch(features,labels):
    #print(tf.shape(features))
    #print(tf.shape(labels))

    with tf.GradientTape() as tape:
        logits = model(features)
        #labels = tf.reshape(labels,(batch_size,timesteps))
        
        #print(tf.shape(logits))
        #print(tf.shape(labels))

        #loss=loss_fn(labels,logits)
        loss = tf.keras.backend.sparse_categorical_crossentropy(labels,logits)

        gradients = tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(gradients,model.trainable_variables))
        train_loss(loss)
        train_acc(labels,logits)

def train_on_batch_cnn(features,labels):

    with tf.GradientTape() as tape:
        logits = model(features)
        #labels = tf.reshape(labels,(batch_size,timesteps))
        
        #print(tf.shape(logits))
        #print(tf.shape(labels))

        #loss=loss_fn(labels,logits)
        loss = tf.keras.backend.sparse_categorical_crossentropy(labels,logits)

        gradients = tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(gradients,model.trainable_variables))
        train_loss(loss)
        train_acc(labels,logits)

def train_on_batch_two(features,labels, labels2=None):
    with tf.GradientTape() as tape:
        #print('====Train====')
        logits1 = model(features)
 
        loss1 = tf.keras.backend.sparse_categorical_crossentropy(labels,logits1)
        #loss1 = tf.reduce_mean(loss1)

        #mask = tf.cast((tf.math.greater(labels2,-1)),tf.bool)
        #labels2_mask = tf.multiply(labels2, tf.cast(mask, tf.float32))
        #loss2 = tf.keras.backend.sparse_categorical_crossentropy(labels2_mask,logits2)
        #loss2 = tf.boolean_mask(loss2, mask)

        #loss2 = tf.reduce_mean(loss2)
        loss = loss1 #0.6*loss1 + 0.4*loss2 
        
        gradients = tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(gradients,model.trainable_variables))
        train_loss(loss)
        train_acc(labels,logits1)
        #train_acc2(labels2,logits2)  


def train_on_batch_ecgseq(features,mask,labels,labels2,heart_beat_nums,rr_pos):
    with tf.GradientTape() as tape:
        #print('====Train====')
        #mask += 1
        #print(tf.shape(features))
        #print(tf.shape(mask))
        #print(labels[0])
        #print(mask[0])
        
        ''' #use same 40x50 input
        features2=[]
        for data,rr in zip(features,rr_pos):
            data = tf.reshape(data, [-1])
            rr = tf.reshape(rr,[-1])
            #print(data.shape)
            #print(rr.shape)
            data2 = transform_data(data,rr)
            features2.append(data2)
        
        features2 = tf.constant(features2)
        features2 = tf.reshape(features2, (batch_size,timesteps,800))
        
        
        logits1,logits2 = model({'inputs1':features,'inputs2':features2})
        
        loss1 = tf.keras.backend.sparse_categorical_crossentropy(labels,logits1)
        loss1 = tf.reduce_mean(loss1)

        mask = tf.cast((tf.math.greater(labels2,-1)),tf.bool)
        labels2_mask = tf.multiply(labels2, tf.cast(mask, tf.int32))
        loss2 = tf.keras.backend.sparse_categorical_crossentropy(labels2_mask,logits2)
        loss2 = tf.boolean_mask(loss2, mask)

        loss2 = tf.reduce_mean(loss2)
        '''
        
        logits1,logits2 = model({'inputs1':features,'inputs2':features})
        #print(logits1.shape)
        #print(logits2.shape)
        loss1 = tf.keras.backend.sparse_categorical_crossentropy(labels,logits1)
        loss1 = tf.reduce_mean(loss1)
        loss2 = tf.keras.backend.sparse_categorical_crossentropy(labels,logits2)
        loss2 = tf.reduce_mean(loss2)
        
        loss = 0.5*loss1 + 0.5*loss2 
        
        gradients = tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(gradients,model.trainable_variables))
        train_loss(loss)
        train_acc(labels,logits1)
        train_acc2(labels,logits2)  

def train_on_batch_ecgseq_ctc(features,mask,labels,labels2,heart_beat_nums,rr_pos):
    with tf.GradientTape() as tape:
        #print('====Train====')
        #mask += 1
        #print(mask[0])


        logits2 = model({'inputs1':features})
        mx = tf.cast((tf.math.greater(labels2,-1)),tf.bool)
        labels2_mask = tf.multiply(labels2, tf.cast(mx, tf.int32))
        
        loss2 = tf.keras.backend.sparse_categorical_crossentropy(labels2_mask,logits2)
        loss2 = tf.boolean_mask(loss2, mx)
        loss2 = tf.reduce_mean(loss2)
        
        '''
        features2=[]
        for data,rr in zip(features,rr_pos):
            data = tf.reshape(data, [-1])
            rr = tf.reshape(rr,[-1])
            data2 = transform_data(data,rr)
            features2.append(data2)
        
        features2 = tf.constant(features2)
        features2 = tf.reshape(features2, (batch_size,timesteps,800))
        '''

        '''
        logits1,logits2 = model({'inputs1':features,'inputs2':features})

        loss1 = tf.keras.backend.sparse_categorical_crossentropy(labels,logits1)
        loss1 = tf.reduce_mean(loss1)
        '''
        
        
        '''
        loss1 = tf.nn.ctc_loss(labels=tf.reshape(tf.cast(labels,tf.int32),(batch_size,timesteps)),
                logits=logits1,
                label_length = tf.reshape(ctc_encoder_length,[-1]),
                logit_length = tf.reshape(ctc_encoder_length,[-1]),
                logits_time_major = False,
                blank_index = 12
                )
                
        loss1 = tf.reduce_mean(loss1)
        '''

        #mx = tf.cast((tf.math.greater(labels2,-1)),tf.bool)
        #labels2_mask = tf.multiply(labels2, tf.cast(mx, tf.int32))
        
        #loss2 = tf.keras.backend.sparse_categorical_crossentropy(labels2_mask,logits2)
        #loss2 = tf.boolean_mask(loss2, mx)
        #loss2 = tf.reduce_mean(loss2)

        #print(logits1.shape)
        #print(logits2.shape)
        #print(labels.shape)
        #print(heart_beat_nums)
        #print(labels[0])
        '''
        loss2 = tf.nn.ctc_loss(labels=labels2,
                logits=logits2,
                label_length = tf.cast(tf.reshape(heart_beat_nums,[-1]),tf.int32),
                logit_length = ctc_encoder_length,
                logits_time_major = False,
                blank_index = 12
                )
        #loss2 = mask_nans(loss2)
        loss2 = tf.reduce_mean(loss2)
        '''
        
        loss = loss2 
        #print(loss, loss1, loss2)
        
        gradients = tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(gradients,model.trainable_variables))
        train_loss(loss)
        #train_acc(labels,logits1)
        train_acc2(labels2,logits2)  


def train_on_batch_ecgatt(features,mask,labels,labels2,heart_beat_nums,rr_pos):
    with tf.GradientTape() as tape:
        #print('====Train====')
        #mask += 1
        
        
        logits = model({'inputs':features})
        
        labels = tf.reshape(labels,(batch_size,timesteps))

        #print(tf.shape(features))
        #print(tf.shape(labels))
        #print(tf.shape(labels2))
        #print(tf.shape(logits))

        loss = loss_fn(labels,logits)
        #loss = tf.keras.backend.sparse_categorical_crossentropy(labels,logits)
        #loss = tf.reduce_mean(loss)

       
        gradients = tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(gradients,model.trainable_variables))
        train_loss(loss)
        train_acc(labels,logits)


#test_the_model
def test_on_batch(features,labels):
    logits = model(features)
    loss=loss_fn(labels,logits)
    test_loss(loss)
    test_acc(labels, logits)


def test_on_batch_two(features,labels, labels2=None):
    logits1 = model(features)
    loss1 = tf.keras.backend.sparse_categorical_crossentropy(labels,logits1)
    #loss1 = tf.reduce_mean(loss1)
    
    #mask = tf.cast((tf.math.greater(labels2,-1)),tf.bool)
    #labels2_mask = tf.multiply(labels2, tf.cast(mask, tf.float32))
    #loss2 = tf.keras.backend.sparse_categorical_crossentropy(labels2_mask,logits2)
    #loss2 = tf.boolean_mask(loss2, mask)
    #loss2 = tf.reduce_mean(loss2)
    
    loss = loss1 # 0.6*loss1 + 0.4*loss2 
    test_loss(loss)
    test_acc(labels, logits1)
    #test_acc2(labels2,logits2)

def test_on_batch_ecgatt(features,mask,labels,labels2,heart_beat_nums,rr_pos):

    logits = model({'inputs':features})
    labels = tf.reshape(labels,(batch_size,timesteps))

    loss = loss_fn(labels,logits)
    test_loss(loss)
    test_acc(labels, logits)

def test_on_batch_ecgseq(features,mask,labels,labels2,heart_beat_nums,rr_pos):
    #mask += 1
    #features = np.multiply(features,mask)
    
    '''
    features2=[]
    for data,rr in zip(features,rr_pos):
        data = tf.reshape(data, [-1])
        rr = tf.reshape(rr,[-1])
        data2 = transform_data(data,rr)
        features2.append(data2)
        
    features2 = tf.constant(features2)
    features2 = tf.reshape(features2, (batch_size,timesteps,800))
    logits1,logits2 = model({'inputs1':features,'inputs2':features2})

    loss1 = tf.keras.backend.sparse_categorical_crossentropy(labels,logits1)
    loss1 = tf.reduce_mean(loss1)

    mask = tf.cast((tf.math.greater(labels2,-1)),tf.bool)
    labels2_mask = tf.multiply(labels2, tf.cast(mask, tf.int32))
    loss2 = tf.keras.backend.sparse_categorical_crossentropy(labels2_mask,logits2)
    loss2 = tf.boolean_mask(loss2, mask)

    loss2 = tf.reduce_mean(loss2)
    '''
    
    logits1,logits2 = model({'inputs1':features,'inputs2':features})

    loss1 = tf.keras.backend.sparse_categorical_crossentropy(labels,logits1)
    loss1 = tf.reduce_mean(loss1)

    loss2 = tf.keras.backend.sparse_categorical_crossentropy(labels,logits1)
    loss2 = tf.reduce_mean(loss2)

    loss = 0.5*loss1 + 0.5*loss2 
    
    test_loss(loss)
    test_acc(labels, logits1)
    test_acc2(labels,logits2)

def compare_array(a,b):
    r=[]
    for (c,d) in zip(a,b):
        r.append(0) if c!=d else r.append(1)
    #l = tf.constant(r)
    #print('a   ', a)
    #print('b   ', b)
    #print('list', r)
    #print('sum',sum(r))
    return sum(r)

"""
This python/numpy example is really misleading. It inherently assumes that the generation of words is completely independent, or alternatively – that P(w_{t}|seq_{1:t-1}) = P(w_{t}).
In this scenario, the results of greedy-search will *always* be the same of the best results of the beam-search.
I suggest to fix the
score = best_score[“i prev”] + -log P(word[i]|next)
to –score = best_score[“i prev”] +-log P(next|prev) + -log P(word[i]|next)
with a real RNN decoding example
"""
def beam_search_decoder(data, k):
	sequences = [[list(), 1.0]]
	# walk over each step in sequence
	for row in data:
		all_candidates = list()
		# expand each current candidate
		for i in range(len(sequences)):
			seq, score = sequences[i]
			for j in range(len(row)):
				candidate = [seq + [j], score * - np.log(row[j])]
				all_candidates.append(candidate)
		# order all candidates by score
		ordered = sorted(all_candidates, key=lambda tup:tup[1])
		# select k best
		sequences = ordered[:k]
	return sequences

def access_model(result_file):
    missing_shot=0
    total_count=0
    print('access_model', result_file)
    with open(result_file, 'w') as csvfile:
        str_fmt='{},{},{}\n'
        str_temp = str_fmt.format('id', 'Ref', 'AI')
        csvfile.write(str_temp)
        #fieldnames = ['id', 'Ref', 'AI']
        #writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        #writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        for k,(id,data,labels,mask,heart_label,heart_beat_nums) in enumerate(test_ds):
            
            if input_shape==3:
                data=np.reshape(data,[batch_size,timesteps,n_features])
            elif input_shape == 4:
                data=np.reshape(data,[batch_size,timesteps,n_features,1])

            mask=np.reshape(mask,[batch_size,timesteps])
            pred1 = model.predict(data) 

            for i in range(batch_size):
                total_count+=1
                eid=str(np.array(id[i],np.int)[0])
                
                p = pred1[i]
                m = mask[i]
                q = [] #adjusted predict label wihtout mask, used to compare with hb 
                hb=[] #true label removed mask
                hb_mask = heart_label[i]
                #print(hb_mask, hb_mask.shape)
                
                for j in range(timesteps):
                    c=int(hb_mask[j])
                    if c>=0:
                        hb.append(c)
                
                for j in range(timesteps):
                    if m[j]==1:
                        v=np.argmax(p[j],axis=-1)
                        """
                        ind=np.argpartition(p[j], -3)[-3:]
                        print('j ', v,ind)
                        print(p[j][ind])
                        if j>1:
                            v=np.argmax(p[j-1],axis=-1)
                            ind=np.argpartition(p[j-1], -3)[-3:]
                            print('j-1',v,ind)
                            print(p[j-1][ind])
                        if j<timesteps-1:
                            v=np.argmax(p[j+1],axis=-1)
                            ind=np.argpartition(p[j+1], -3)[-3:]
                            print('j+1',v,ind)
                            print(p[j+1][ind])
                        """
                        q.append(v)
                    
                if len(q)!=len(hb):
                    missing_shot+=1
                    print('Step=', k, 'eid=', eid, int(tf.reduce_sum(m)),q,hb)
                else:
                    for j in range(len(q)):
                        str_temp = str_fmt.format(eid, q[j],hb[j])
                        csvfile.write(str_temp)
                        #writer.writerow({'id':eid, 'Ref':q[j], 'AI':hb[j]})
               
    print('missing_shot and total count:', missing_shot, total_count, (missing_shot/total_count)*100)

def access_model_all(result_file):
    missing_shot=0
    total_count=0

    with open(result_file, 'w') as csvfile:
        str_fmt='{},{},{}\n'
        str_temp = str_fmt.format('id', 'Ref', 'AI')
        csvfile.write(str_temp)
        #fieldnames = ['id', 'Ref', 'AI']
        #writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        #writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        for k,((id2,data2,labels2,mask2,hbl2),(id5,data5,labels5,mask5,hbl5)) in enumerate(zip(test_ds2,test_ds5)):
            
            if input_shape==3:
                data=np.reshape(data,[batch_size,timesteps,n_features])
            elif input_shape == 4:
                data2=np.reshape(data2,[batch_size,timesteps,n_features,1])
                data5=np.reshape(data5,[batch_size,timesteps,n_features,1])

            mask2=np.reshape(mask2,[batch_size,timesteps])
            pred = model.predict([data2,data5]) 

            for i in range(batch_size):
                total_count+=1
                eid=str(np.array(id2[i],np.int)[0])
                
                p = pred[i]
                m = mask2[i]
                q = [] #adjusted predict label wihtout mask, used to compare with hb 
                hb=[] #true label removed mask
                hb_mask = hbl2[i]
                #print(hb_mask, hb_mask.shape)
                
                for j in range(timesteps):
                    c=int(hb_mask[j])
                    if c>=0:
                        hb.append(c)
                
                for j in range(timesteps):
                    if m[j]==1:
                        v=np.argmax(p[j],axis=-1)
                        q.append(v)
                    
                if len(q)!=len(hb):
                    missing_shot+=1
                    print('Step=', k, 'eid=', eid, int(tf.reduce_sum(m)),q,hb)
                else:
                    for j in range(len(q)):
                        str_temp = str_fmt.format(eid, q[j],hb[j])
                        csvfile.write(str_temp)
               
    print('missing_shot and total count:', missing_shot, total_count, (missing_shot/total_count)*100)


def prepare_image_data(n):
    ids=random.sample(range(total_test_steps_per_epoch),n)
    
    _ecgid=[]
    _data=[]
    _label=[]
    _heart_label=[]
    _pred1=[]
    _pred2=[]
    _pred3=[]
    
    for id in ids:
        ecgid,data,label,mask, heart_label=get_data(test_ds,id,1)
        
        if input_shape==3:
            data=np.reshape(data,[batch_size,timesteps,n_features])
        elif input_shape == 4:
            data=np.reshape(data,[batch_size,timesteps,n_features,1])

        #heart_label=np.reshape(heart_label,[batch_size,timesteps])
        mask=np.reshape(mask,[batch_size,timesteps])
        
        pred1 = model.predict(data) 
        _ecgid.append(ecgid)
        _data.append(data)
        _label.append(label)
        _heart_label.append(heart_label)
        _pred1.append(pred1)
        #_pred2.append(pred2)
        
        _pred3_val = []
        
        for i in range(batch_size):
            p = pred1[i]
            m = mask[i]
            q = [] #adjusted predict label wihtout mask, used to compare with hb 
            hb=[] #true label removed mask
            hb_mask = heart_label[0][i]
            #remove hb mask '-1'
            for j in range(timesteps):
                c=int(hb_mask[j])
                if c>=0:
                    hb.append(c)
                    
            for j in range(timesteps):
                if m[j]==1:  #find QRS location
                    k=np.argmax(p[j],axis=-1)
                    #ind=np.argpartition(p[j], -3)[-3:]
                    #print(k,ind)
                    #print(p[j][ind])
                    q.append(k)
            
            
            if len(q)!=len(hb):
                print('Invalid len!', ecgid[i])
 
            _pred3_val.append(q)
        
        _pred3.append(_pred3_val)
        
        """
        win_count=0
        for i in range(batch_size):
            preds = pred1[i]
            preds_val=np.argmax(preds,axis=1)
            print(tf.shape(preds_val))
            true_y=label[0][i].reshape(timesteps)
            
            #print('Labels  :',true_y)
            x=compare_array(true_y,preds_val)
            for (_,t) in beam_search_decoder(preds,1):
                print('Greed   :',preds_val,x)
                
            preds_beam = beam_search_decoder(preds,3)
            win_flag = False
            for (d,p) in preds_beam:
                y = compare_array(true_y,array(d))
                if y>=x and p>t:
                    print('beam WIN:', array(d), y)  
                    win_flag=True        
                else:
                    print('beam LSS:',array(d), y)    
            print('------------------------------------------')
            
            if win_flag==True:
                win_count+=1
            
        print('Win Count:', win_count)
        """

    return _ecgid, _data,_label,_heart_label,_pred1,_pred3
    
    
def get_data(data, index=0, length=10):
    features = []
    labels = []
    heart_labels = []
    masks = []
    ecgids = []
    
    for i,(id,feature,label,mask, heart_label,heart_beat_nums) in enumerate(data):
        #print('get_data: ',i)
        if (i>=index and i<index+length):
            #print('append: ', label.shape)
            ecgids.append(id)
            feature=feature.numpy()
            label=label.numpy()
            features.append(feature)
            labels.append(label)
            heart_labels.append(heart_label)
            masks.append(mask)
        elif (i>=index+length):
            break
    
    if len(features)==0:
        print('[ERROR] No test data feed')
        exit(-1)    
        
    return ecgids,features, labels, masks, heart_labels

print('Loading data.....')

if data_path=='simple':
    print('Loading data.....simple')
    train_ds, total_train_samples = LEPU_OLD_EcgDataPipe.load_data(train_data_path, batch_size = batch_size, input_shape=input_shape, subset='train*', train = True)
    test_ds, total_test_samples = LEPU_OLD_EcgDataPipe.load_data(test_data_path, batch_size = batch_size, input_shape=input_shape, subset='test*', train = False)
else:
    if for_train==True:
        #train_ds, total_train_samples = EcgData.load3(train_data_path, batch_size = batch_size,leader=lead ,input_shape=input_shape, train = True)
        #test_ds, total_test_samples = EcgData.load3(test_data_path, batch_size = batch_size,leader=lead ,input_shape=input_shape,train = False)

        
        train_ds,total_train_samples=LEPU_OLD_EcgDataPipe.load_train(train_data_path, batch_size, weights=[0.9, 0.1])
        #test_ds,total_test_samples=EcgData.load_train(test_data_path,batch_size,weights=[0.9,0.1])
        test_ds,total_test_samples=LEPU_OLD_EcgDataPipe.load_new(test_data_path, batch_size, timesteps, False, 1)

        
        total_train_steps_per_epoch=int(total_train_samples//batch_size)
        total_test_steps_per_epoch=int(total_test_samples//batch_size)
        print('total train, test samples; train_steps/epoch,test_steps/epoch',total_train_samples,total_test_samples,total_train_steps_per_epoch,total_test_steps_per_epoch)
    else:
        test_path='data/mini3/Test/NonPureN'
        test_ds, total_test_samples = LEPU_OLD_EcgDataPipe.load_II(test_data_path, batch_size, False, 1)
        total_test_steps_per_epoch=int(total_test_samples//batch_size)

        print('total test samples',total_test_samples)



#metrics
"""
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_acc = tf.keras.metrics.CategoricalAccuracy(name='train_acc')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_acc = tf.keras.metrics.CategoricalAccuracy(name='test_acc')
"""

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')
train_acc2 = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc2')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='test_acc')
test_acc2 = tf.keras.metrics.SparseCategoricalAccuracy(name='test_acc2')


train_acc_history=[]
train_acc2_history=[]
test_acc_history=[]

train_loss_history=[]
test_loss_history=[]

# loss
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
#loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
#loss_fn = tf.keras.losses.mse()

if for_train:
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps = 0.75*total_train_steps_per_epoch,
        decay_rate = 0.90,
        staircase = True)

    # optimizer
    #optimizer = tf.keras.optimizers.SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)
    #optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule, rho=0.9)


train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

def do_train(epoch):
    for i,(id,data,labels,mask,heart_beat,heart_beat_nums, rr_pos) in enumerate(train_ds):
        
        if model_name=='two':
            #features = np.reshape(features,(batch_size,timesteps,n_features))
            data = np.reshape(data,(batch_size,timesteps,n_features))
            #labels2 = np.reshape(labels2,(batch_size,max_heartbeats))
            train_on_batch(data,labels)
        elif model_name=='tcp':
            data = tf.reshape(data,(-1,timesteps,n_features))
            train_on_batch_tcp(data,labels,labels2)
        elif model_name=='cnnlstm_d_two' or model_name=='cnnlstm_s':
            #data = tf.reshape(data,(-1,timesteps,n_features))
            #train_on_batch(data,labels)
            train_on_batch_two(data,labels,heart_beat)
        elif model_name=='cnnlstm_d':
            data = tf.reshape(data,(-1,timesteps,n_features,1))
            labels = tf.reshape(labels,(-1,timesteps,1))
            train_on_batch_cnn(data,labels)
        elif model_name=='seq':
            data = np.reshape(data,(batch_size,timesteps,n_features))
            labels = np.reshape(labels,(batch_size,timesteps,1))
            train_on_batch_seq(data,labels,labels2)
        elif model_name=='ecgseq':
            data = tf.reshape(data,(-1,timesteps,n_features))
            labels = tf.reshape(labels,(-1,timesteps,1))
            mask = tf.reshape(mask,(-1,timesteps,1))
            rr_pos = tf.reshape(rr_pos,(-1,timesteps,1))
            train_on_batch_ecgseq(data,mask,labels,heart_beat,heart_beat_nums,rr_pos)
        elif model_name=='ecgatt':
            data = tf.reshape(data,(-1,timesteps,n_features))
            labels = tf.reshape(labels,(-1,timesteps,1))
            mask = tf.reshape(mask,(-1,timesteps,1))
            rr_pos = tf.reshape(rr_pos,(-1,timesteps,1))
            train_on_batch_ecgatt(data,mask,labels,heart_beat,heart_beat_nums,rr_pos)
        else:
            data = tf.reshape(data,(-1,timesteps,n_features))
            labels = tf.reshape(labels,(-1,timesteps))
            train_on_batch(data,labels)
        
        end1 = time.time()
        with train_summary_writer.as_default():
            tf.summary.scalar('Loss-Training', train_loss.result(), step=i + epoch*total_train_steps_per_epoch)
            tf.summary.scalar('Acc-Training', train_acc.result(), step=i + epoch*total_train_steps_per_epoch)
            #tf.summary.scalar('Acc2-Training', train_acc2.result(), step=i + epoch*total_train_steps_per_epoch)
            train_summary_writer.flush()
    
        '''
        if for_trial and i>print_steps:        
            print("Exit Training Trail: Loss={:09.6f}, Acc={:09.6f}, Acc2={:09.6f}".format(epoch, i,train_loss.result(),train_acc.result()*100, train_acc2.result()*100))
            break
        '''
        
        if (i%print_steps == 0) and (i>1):
            print("Train Epoch {}, step {}: Loss={:09.6f}, Acc={:09.6f}, Acc2={:09.6f}".format(epoch, i,train_loss.result(),train_acc.result()*100, train_acc2.result()*100))
        


def do_evaluate(epoch):
    for i,(id,data,labels,mask,labels2,heart_beat_nums,rr_pos) in enumerate(test_ds):
        if model_name=='two':
            data = np.reshape(data,(batch_size,timesteps,n_features))
            test_on_batch(data,labels)
        elif model_name=='tcp':
            data = tf.reshape(data,(-1,timesteps,n_features))
            test_on_batch_tcp(data,labels,labels2)
        elif model_name=='seq':
            data = np.reshape(data,(batch_size,timesteps,n_features))
            labels = np.reshape(labels,(batch_size,timesteps,1))
            test_on_batch_seq(data,labels,labels2)
        elif model_name=='cnnlstm_d_two' or model_name=='cnnlstm_s':
            #data = tf.reshape(data,(-1,timesteps,n_features))
            test_on_batch_two(data,labels,labels2)
        elif model_name=='cnnlstm_d':
            data = tf.reshape(data,(batch_size,timesteps,n_features,1))
            labels = tf.reshape(labels,(batch_size,timesteps,1))
            test_on_batch(data,labels)
        elif model_name=='ecgseq':
            data = tf.reshape(data,(-1,timesteps,n_features))
            label = tf.reshape(labels,(-1,timesteps,1))
            mask = tf.reshape(mask,(-1,timesteps,1))
            rr_pos = tf.reshape(rr_pos,(-1,timesteps,1))
            test_on_batch_ecgseq(data,mask,labels,labels2,heart_beat_nums,rr_pos)
        elif model_name=='ecgatt':
            data = tf.reshape(data,(-1,timesteps,n_features))
            label = tf.reshape(labels,(-1,timesteps,1))
            mask = tf.reshape(mask,(-1,timesteps,1))
            rr_pos = tf.reshape(rr_pos,(-1,timesteps,1))
            test_on_batch_ecgatt(data,mask,labels,labels2,heart_beat_nums,rr_pos)
        else:
            data = tf.reshape(data,(-1,timesteps,n_features))
            label = tf.reshape(labels,(-1,timesteps))
            test_on_batch(data,labels)
        
    with test_summary_writer.as_default():
       tf.summary.scalar('Loss-Test', test_loss.result(), step=(epoch+1)*total_test_steps_per_epoch)
       tf.summary.scalar('Acc-Test', test_acc.result(), step=(epoch+1)*total_test_steps_per_epoch)
       #tf.summary.scalar('Acc2-Test', test_acc2.result(), step=i+epoch*total_test_steps_per_epoch)
       test_summary_writer.flush()

    '''
    if for_trial and i>print_steps:        
        print("Exit Testing Trail: Loss={:09.6f}, Acc={:09.6f}, Acc2={:09.6f}".format(epoch, i,test_loss.result(),test_acc.result()*100, test_acc2.result()*100))
        break
    '''
    
    print("Test  Epoch {}, Loss={:09.6f}, Acc={:09.6f}, Acc2={:09.6f}".format(epoch, test_loss.result(),test_acc.result()*100, test_acc2.result()*100))


if for_train==True:
    localtime = time.strftime('%H:%M:%S', time.localtime(time.time()))
    template='#Begin Training at {}#\nTraining Info: epochs={}, train steps={}, test steps={}, lead={}, input_shape={}, print_steps={}'
    print(template.format(localtime,epochs,total_train_steps_per_epoch,total_test_steps_per_epoch,lead,input_shape,print_steps))
    
    #model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    #model.fit(train_ds, epochs=10, validation_data=test_ds)

    if os.path.exists(model_path)==False: 
        os.makedirs(model_path)
        print('Create model path:',model_path)
        
    #plot_model(model, to_file= os.path.join(model_path, model_name+'.png'), show_shapes=True)
    for epoch in range(epochs):
        template='--------------------Training epoch {} {}--------------------'
        print(template.format(epoch,time.strftime('%H:%M:%S', time.localtime(time.time()))))
        if for_lead_all==True:
            do_train_all(epoch)
        else:
            do_train(epoch)
        
        template='--------------------Testing epoch {} {}--------------------'
        print(template.format(epoch,time.strftime('%H:%M:%S', time.localtime(time.time()))))
        if for_lead_all==True:
            do_evaluate_all(epoch)
        else:
            do_evaluate(epoch)
            #print('skip test')
            
        #save model every epoch for specific lead
        f=os.path.join(model_path, model_name +'_' + lead + '_' + str(timesteps) + '.h5') 
        #model.save(f)
        #print('saving model to:',f)
        train_acc_history.append(train_acc.result())
        train_loss_history.append(train_loss.result())
        test_acc_history.append(test_acc.result())
        test_loss_history.append(test_loss.result())


        # Reset the metrics for the next epoch
        train_loss.reset_states()
        train_acc.reset_states()
        train_acc2.reset_states()
        
        test_loss.reset_states()
        test_acc.reset_states()  
        test_acc2.reset_states()  
        
    localtime = time.strftime('%H:%M:%S', time.localtime(time.time()))
    template='#Complete Training at {}#\nTrain Loss={:09.6f}, Acc={:09.6f}, Acc2={:09.6f}; Test Loss={:09.6f}, Acc={:09.6f}, Acc2={:09.6f}'
    print(template.format(localtime,train_loss.result(),train_acc.result()*100,train_acc2.result()*100,
    test_loss.result(),test_acc.result()*100,test_acc2.result()*100))

else:
    localtime = time.strftime('%H:%M:%S', time.localtime(time.time()))
    template='#Begin Inference at {}#'
    print(template.format(localtime))

    model = tf.keras.models.load_model(os.path.join(model_path, model_name +'_' + lead + '_' + str(timesteps) + '.h5'))

    if os.path.exists(result_path)==False: 
        os.makedirs(result_path)
    
    if for_lead_all == True:
        access_model_all(result_file)
    else:
        access_model(result_file)
    
    _ecgid,_data,_label,_heart_label,_pred1, _pred3 = prepare_image_data(1)
    
    ImageGen.run_test_and_save_image(result_path, batch_size, _ecgid, _data,_label,_heart_label,_pred1,_pred3)

    localtime = time.strftime('%H:%M:%S', time.localtime(time.time()))
    template='#Complete Inference at {}#'
    print(template.format(localtime))
    
end=time.time()
duration= (end-start)

template='for-Train={}, is-Trail={}, for-Infer={}.  Model={}, Data={}, Lead={}, Run-time={}  Run-duration={}'
print(template.format(for_train, for_trial, not for_train, model_name, data_path,lead,run_time,duration))
