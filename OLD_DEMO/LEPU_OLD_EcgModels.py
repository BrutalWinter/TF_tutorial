import numpy as np
from numpy import array

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LeakyReLU,PReLU,ELU,ReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed

#from tensorflow.keras.layers import Layer
#from tensorflow.keras import backend as K
#from tensorflow.keras import initializers, regularizers, constraints

'''
class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
        #print('attention build',input_shape)
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")        
        super(attention, self).build(input_shape)

    def call(self,x):
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)
        output=x*at
        #print('attention call x,et,at,output:', tf.shape(x),tf.shape(et),tf.shape(at),tf.shape(output))
        #print('attention K.sum:',K.sum(output,axis=1))
        #return K.sum(output,axis=1)
        return output
        
        
    def compute_output_shape(self,input_shape):
        print('attention compute_output_shape:', tf.shape(input_shape),input_shape[0],input_shape[-1])
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        return super(attention,self).get_config()
        


class Attention1(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention1, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim
'''
        
#1D CNN + BLSTM分类模型
def model_Cnn1DLSTM(filter1,times,dim,classes):
    model = tf.keras.Sequential()
    model.add(layers.Conv1D(filter1,3,activation='relu',input_shape=(times,dim),padding='same'))
    model.add(layers.MaxPooling1D(2,1,padding='same',data_format='channels_last'))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv1D(filter1,3,activation='relu',padding='same'))
    model.add(layers.MaxPooling1D(2,1,padding='same',data_format='channels_last'))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    #print('model.output_shape：', model.output_shape)
    #model.add(layers.Reshape([times,dim*filter1]))
    model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True,kernel_regularizer=keras.regularizers.l2(0.001))))
    model.add(layers.MaxPooling1D(2,1,padding='same',data_format='channels_last'))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True,kernel_regularizer=keras.regularizers.l2(0.001))))
    model.add(layers.MaxPooling1D(2,1,padding='same',data_format='channels_last'))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True,kernel_regularizer=keras.regularizers.l2(0.001))))
    model.add(layers.MaxPooling1D(2,1,padding='same',data_format='channels_last'))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(classes, activation='softmax',kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(layers.Dense(classes, activation='softmax',kernel_regularizer=keras.regularizers.l2(0.001)))
    model.summary()
    return model


def model_CnnLSTM_S(filter1,times,dim,classes):
    inputs = layers.Input(shape=(times,dim,1))
    outputs = layers.Conv2D(filter1,[1,3],activation='relu',padding='same',kernel_regularizer=keras.regularizers.l2(0.001))(inputs)
    outputs = layers.Conv2D(filter1*2,[1,3],activation='relu',padding='same',kernel_regularizer=keras.regularizers.l2(0.001))(outputs)
    outputs = layers.Reshape([times,dim*filter1*2])(outputs)
    outputs = layers.Bidirectional(layers.LSTM(64, return_sequences=True,kernel_regularizer=keras.regularizers.l2(0.001)))(outputs)
    outputs = layers.Bidirectional(layers.LSTM(128, return_sequences=True,kernel_regularizer=keras.regularizers.l2(0.001)))(outputs)
    outputs = layers.Bidirectional(layers.LSTM(128, return_sequences=True,kernel_regularizer=keras.regularizers.l2(0.001)))(outputs)
    outputs = layers.Bidirectional(layers.LSTM(256, return_sequences=True,kernel_regularizer=keras.regularizers.l2(0.001)))(outputs)

    outputs = layers.Dense(classes, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001))(outputs)
    outputs = layers.Dense(classes, activation='softmax',kernel_regularizer=keras.regularizers.l2(0.001))(outputs)
    
    model = Model(inputs, outputs)
    
    #model.summary()
    return model



#deep and wider stacked 2D CNN + BLSTM分类模型LSTM
def model_CnnLSTM_D1(filter1,times,dim,classes):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(filter1,[1,3],activation='relu',input_shape=(times,dim,1),padding='same',kernel_regularizer=keras.regularizers.l2(0.001)))
    #model.add(layers.MaxPooling2D((1,2),padding='same'))
    model.add(layers.Dropout(0.2))
    #print(model.output_shape)
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filter1*2,[1,3],activation='relu',input_shape=(times,dim,1),padding='same',kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(layers.Dropout(0.2))
    print(model.output_shape)
    model.add(layers.BatchNormalization())
    model.add(layers.Reshape([times,dim*filter1*2]))
    model.add(layers.Bidirectional(layers.LSTM(32, return_sequences=True,kernel_regularizer=keras.regularizers.l2(0.001)),input_shape=(times,dim)))
    #model.add(layers.MaxPooling1D(2,1,padding='same',data_format='channels_last'))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True,kernel_regularizer=keras.regularizers.l2(0.001)),input_shape=(times,dim)))
    #model.add(layers.MaxPooling1D(2,1,padding='same',data_format='channels_last'))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True,kernel_regularizer=keras.regularizers.l2(0.001)),input_shape=(times,dim)))
    #model.add(layers.MaxPooling1D(2,1,padding='same',data_format='channels_last'))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())    
    model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True,kernel_regularizer=keras.regularizers.l2(0.001)),input_shape=(times,dim)))
    #model.add(layers.MaxPooling1D(2,1,padding='same',data_format='channels_last'))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True,kernel_regularizer=keras.regularizers.l2(0.001)),input_shape=(times,dim)))
    #model.add(layers.MaxPooling1D(2,1,padding='same',data_format='channels_last'))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(classes*5, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001)))
    #model.add(layers.MaxPooling1D(2,1,padding='same',data_format='channels_last'))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(classes, activation='softmax',kernel_regularizer=keras.regularizers.l2(0.001)))
    model.summary()
    return model



# 在CnnLSTM_D1基础上，考虑two ouputs 训练, logits1 Acc可以到96%, logits2 Acc在23%
#初步结果目测,logitis1的40step 分类效果有改进.
#接下来,在这个模型基础上,进行Seq2Seq的融合 -->model_EcgSequence
def model_CnnLSTM_D1_Two(filter1,times,dim,classes):
    
    
    inputs = layers.Input(shape=(times,dim,1))
    outputs = layers.Masking(mask_value=-1)(inputs)
    outputs = layers.Conv2D(filter1,[1,3],activation='relu',padding='same',kernel_regularizer=keras.regularizers.l2(0.001))(inputs)
    outputs = layers.MaxPooling2D((1,2),padding='same',data_format='channels_last')(outputs)
    outputs = layers.Dropout(0.2)(outputs)
    outputs = layers.BatchNormalization()(outputs)
    outputs = layers.Conv2D(filter1*2,[1,3],activation='relu',padding='same',kernel_regularizer=keras.regularizers.l2(0.001))(outputs)
    outputs = layers.MaxPooling2D((1,2),padding='same',data_format='channels_last')(outputs)
    outputs = layers.Dropout(0.2)(outputs)
    outputs = layers.BatchNormalization()(outputs)
    #outputs = layers.Reshape([times,dim*filter1*2])(outputs)
    outputs = layers.Reshape([times,int(dim/4+0.5)*filter1 *2])(outputs)
    outputs = layers.Bidirectional(layers.LSTM(64, return_sequences=True,kernel_regularizer=keras.regularizers.l2(0.001)))(outputs)
    outputs = layers.Dropout(0.2)(outputs)
    outputs = layers.BatchNormalization()(outputs)
    outputs = layers.Bidirectional(layers.LSTM(64, return_sequences=True,kernel_regularizer=keras.regularizers.l2(0.001)))(outputs)
    outputs = layers.MaxPooling1D(2,1,padding='same',data_format='channels_last')(outputs)
    outputs = layers.Dropout(0.2)(outputs)
    outputs = layers.BatchNormalization()(outputs)
    outputs = layers.Bidirectional(layers.LSTM(128, return_sequences=True,kernel_regularizer=keras.regularizers.l2(0.001)))(outputs)
    outputs = layers.MaxPooling1D(2,1,padding='same',data_format='channels_last')(outputs)
    outputs = layers.Dropout(0.2)(outputs)
    outputs = layers.BatchNormalization()(outputs)    
    outputs = layers.Bidirectional(layers.LSTM(128, return_sequences=True,kernel_regularizer=keras.regularizers.l2(0.001)))(outputs)
    outputs = layers.MaxPooling1D(2,1,padding='same',data_format='channels_last')(outputs)
    outputs = layers.Dropout(0.2)(outputs)
    outputs = layers.BatchNormalization()(outputs)
    outputs = layers.Bidirectional(layers.LSTM(256, return_sequences=True,kernel_regularizer=keras.regularizers.l2(0.001)))(outputs)
    outputs = layers.MaxPooling1D(2,1,padding='same',data_format='channels_last')(outputs)
    outputs = layers.Dropout(0.2)(outputs)
    outputs = layers.BatchNormalization()(outputs)
    
    
    lstm_outputs, state_h, state_c = layers.LSTM(256, return_sequences=True, return_state=True,kernel_regularizer=keras.regularizers.l2(0.001))(outputs)
    encoder_states = ([state_h, state_c])
    """
   
    lstm_outputs,forward_h, forward_c, backward_h, backward_c = layers.Bidirectional(
    layers.LSTM(128, activation='relu', return_sequences=True, return_state=True))(outputs)
    state_h = layers.Concatenate(axis=-1)([forward_h, backward_h])
    state_c = layers.Concatenate(axis=-1)([forward_c, backward_c])   
    encoder_states = ([state_h, state_c])
    """
    
    #-------------- Above the basic -------------------------------------------------------------------------------
    
    #----40 标签的分类输出 output1--------------------------------------------------------
    outputs = layers.Dense(classes*2, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001))(lstm_outputs)
    outputs = layers.MaxPooling1D(2,1,padding='same',data_format='channels_last')(outputs)
    outputs = layers.Dropout(0.2)(outputs)
    outputs = layers.BatchNormalization()(outputs)
    outputs = layers.Dense(classes, activation='softmax',kernel_regularizer=keras.regularizers.l2(0.001))(outputs)
    outputs = layers.MaxPooling1D(2,1,padding='same',data_format='channels_last')(outputs)
    outputs = layers.Dropout(0.2)(outputs)
    outputs = layers.BatchNormalization()(outputs)
    outputs = layers.Dense(classes, activation='softmax',kernel_regularizer=keras.regularizers.l2(0.001))(outputs)

    #----可以变心搏真实分类输出 output2-------------------------------------------------------------------------------------
    
    output2s = layers.LSTM(256, return_sequences=True,name='lstm_out2')(outputs,initial_state=encoder_states)
    #output2s = layers.LSTM(128, return_sequences=True, name='lstm_out2')(output2s)
    output2s = layers.MaxPooling1D(2,1,padding='same',data_format='channels_last')(output2s)
    output2s = layers.Dropout(0.2)(output2s)
    output2s = layers.BatchNormalization()(output2s)
    output2s = layers.Bidirectional(layers.LSTM(256, return_sequences=True,kernel_regularizer=keras.regularizers.l2(0.001)))(output2s)
    output2s = layers.MaxPooling1D(2,1,padding='same',data_format='channels_last')(output2s)
    output2s = layers.Dropout(0.2)(output2s)
    output2s = layers.BatchNormalization()(output2s)
    output2s = layers.Bidirectional(layers.LSTM(128, return_sequences=True,kernel_regularizer=keras.regularizers.l2(0.001)))(output2s)
    output2s = layers.MaxPooling1D(2,1,padding='same',data_format='channels_last')(output2s)
    output2s = layers.Dropout(0.2)(output2s)
    output2s = layers.BatchNormalization()(output2s)

    output2s = layers.Dense(classes, activation='relu')(output2s)
    output2s = layers.MaxPooling1D(2,1,padding='same',data_format='channels_last')(output2s)
    output2s = layers.Dropout(0.2)(output2s)
    output2s = layers.BatchNormalization()(output2s)
    output2s = layers.Dense(classes, activation='softmax')(output2s)
    output2s = layers.MaxPooling1D(2,1,padding='same',data_format='channels_last')(output2s)
    output2s = layers.Dropout(0.2)(output2s)
    output2s = layers.BatchNormalization()(output2s)
    output2s = layers.Dense(classes, activation='softmax')(output2s)

    #mask = layers.Input(shape=(times, classes), dtype='float32', name='mask')
    #output2_with_mask = layers.Multiply()([output2s, mask])
    
    #model = Model(inputs, [outputs,output2s])
    model = Model(inputs, [outputs])
    
    #model.summary()
    return model

def lead_layer(filter1,times,dim,classes,inputs):
    outputs = layers.Masking(mask_value=-1)(inputs)
    outputs = layers.Conv2D(filter1,[1,3],activation='relu',padding='same',kernel_regularizer=keras.regularizers.l2(0.001))(inputs)
    outputs = layers.MaxPooling2D((1,2),padding='same',data_format='channels_last')(outputs)
    outputs = layers.Dropout(0.2)(outputs)
    outputs = layers.BatchNormalization()(outputs)
    outputs = layers.Conv2D(filter1*2,[1,3],activation='relu',padding='same',kernel_regularizer=keras.regularizers.l2(0.001))(outputs)
    outputs = layers.MaxPooling2D((1,2),padding='same',data_format='channels_last')(outputs)
    outputs = layers.Dropout(0.2)(outputs)
    outputs = layers.BatchNormalization()(outputs)
    #outputs = layers.Reshape([times,dim*filter1*2])(outputs)
    outputs = layers.Reshape([times,int(dim/4+0.5)*filter1 *2])(outputs)
    outputs = layers.Bidirectional(layers.LSTM(64, return_sequences=True,kernel_regularizer=keras.regularizers.l2(0.001)))(outputs)
    outputs = layers.Dropout(0.2)(outputs)
    outputs = layers.BatchNormalization()(outputs)
    outputs = layers.Bidirectional(layers.LSTM(64, return_sequences=True,kernel_regularizer=keras.regularizers.l2(0.001)))(outputs)
    outputs = layers.MaxPooling1D(2,1,padding='same',data_format='channels_last')(outputs)
    outputs = layers.Dropout(0.2)(outputs)
    outputs = layers.BatchNormalization()(outputs)
    outputs = layers.Bidirectional(layers.LSTM(128, return_sequences=True,kernel_regularizer=keras.regularizers.l2(0.001)))(outputs)
    outputs = layers.MaxPooling1D(2,1,padding='same',data_format='channels_last')(outputs)
    outputs = layers.Dropout(0.2)(outputs)
    outputs = layers.BatchNormalization()(outputs)    
    outputs = layers.Bidirectional(layers.LSTM(128, return_sequences=True,kernel_regularizer=keras.regularizers.l2(0.001)))(outputs)
    outputs = layers.MaxPooling1D(2,1,padding='same',data_format='channels_last')(outputs)
    outputs = layers.Dropout(0.2)(outputs)
    outputs = layers.BatchNormalization()(outputs)
    outputs = layers.Bidirectional(layers.LSTM(256, return_sequences=True,kernel_regularizer=keras.regularizers.l2(0.001)))(outputs)
    outputs = layers.MaxPooling1D(2,1,padding='same',data_format='channels_last')(outputs)
    outputs = layers.Dropout(0.2)(outputs)
    outputs = layers.BatchNormalization()(outputs)
    
    
    lstm_outputs, state_h, state_c = layers.LSTM(256, return_sequences=True, return_state=True,kernel_regularizer=keras.regularizers.l2(0.001))(outputs)
    encoder_states = ([state_h, state_c])
    
    return outputs, lstm_outputs,encoder_states




#测试custom activation
def model_Seq(filter1,timesteps,input_dim,classes):
    encoder_inputs = layers.Input(shape=(None,input_dim))
    encoder = layers.LSTM(filter1, return_sequences=True,return_state=True)

    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    
    encoder_states = [state_h, state_c]
    #print('encoder_states',np.shape(encoder_states))
    #print('state_h',state_h.shape)
    #print('state_c',state_c.shape)

    decoder_inputs = layers.Input(shape=(None,input_dim))
    decoder_lstm = layers.LSTM(filter1, return_sequences=True,return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    print('decoder_outputs',decoder_outputs.shape)
    #decoder_dense = layers.Dense(input_dim,activation='tanh') #数据幅值太小,没有负值？
    #decoder_dense = layers.Dense(input_dim,activation=myReLU) #数据幅值太小 调整参数，还是太小
    #decoder_dense = layers.Dense(input_dim,activation=mySeLU) #数据幅值太小
    #decoder_dense = layers.Dense(input_dim,activation=myELU) #没有生成数据,1.0 太小，2.0 感觉生成噪声
    #decoder_dense = layers.Dense(input_dim,activation=myLeakyReLU) #数据幅值太小 0.3，0.9
    #decoder_outputs = decoder_dense(decoder_outputs)
    td = layers.TimeDistributed(layers.Dense(input_dim))
    decoder_outputs = td(decoder_outputs)
    
    all_inputs = [encoder_inputs, decoder_inputs]
    
    model = Model(all_inputs, decoder_outputs)
    model.summary()
    return model

#验证j基于NLP characters translation对40->v Heartbeats的转换,效果不佳 
def model_Seq2Seq(filters, num_encoder_tokens, num_decoder_tokens):
    # Define an input sequence and process it.
    encoder_inputs = layers.Input(shape=(None, num_encoder_tokens))
    encoder = layers.LSTM(filters, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = layers.Input(shape=(None, num_decoder_tokens))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = layers.LSTM(filters, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = layers.Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.summary()
    
    encoder_model = Model(encoder_inputs, encoder_states)
    encoder_model.summary()
    
    decoder_state_input_h = layers.Input(shape=(filters,))
    decoder_state_input_c = layers.Input(shape=(filters,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    decoder_model.summary()
    
    return model,encoder_model,decoder_model
          

# 
def model_EcgSeq(filters,times,dim,classes):

    inputs1 = layers.Input(shape=(times, dim))
    encoder_inputs = layers.Conv1D(filters,3,activation='relu',padding='same')(inputs1)
    encoder_inputs = layers.BatchNormalization()(encoder_inputs)
    encoder_inputs = layers.Dense(filters, activation='relu')(encoder_inputs)
    encoder_inputs = layers.Dense(filters)(encoder_inputs)
    encoder_inputs = layers.BatchNormalization()(encoder_inputs)
    
    encoder_inputs = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(encoder_inputs)
    encoder_inputs = layers.BatchNormalization()(encoder_inputs)
    encoder_inputs = layers.Dense(filters*2, activation='relu')(encoder_inputs)
    encoder_inputs = layers.Dense(filters*2)(encoder_inputs)
    encoder_inputs = layers.BatchNormalization()(encoder_inputs)

    encoder_inputs = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(encoder_inputs)
    encoder_inputs = layers.BatchNormalization()(encoder_inputs)
    encoder_inputs = layers.Dense(filters*2, activation='relu')(encoder_inputs)
    encoder_inputs = layers.Dense(filters*2)(encoder_inputs)
    encoder_inputs = layers.BatchNormalization()(encoder_inputs)
    
    encoder_outputs,forward_h, forward_c, backward_h, backward_c = layers.Bidirectional(
    layers.LSTM(64, activation='relu', return_sequences=True, return_state=True))(encoder_inputs)
    
    state_h = layers.Concatenate(axis=-1)([forward_h, backward_h])
    state_c = layers.Concatenate(axis=-1)([forward_c, backward_c])   
    encoder_states = ([state_h, state_c])
    
    encoder_outputs = layers.BatchNormalization()(encoder_outputs)
    encoder_outputs = layers.Dense(classes, activation='relu')(encoder_outputs)
    encoder_outputs = layers.Dense(classes, activation='softmax')(encoder_outputs)
    
    inputs2 = layers.Input(shape=(times, 50))
    decoder_inputs = layers.Masking(mask_value=-1000)(inputs2)
    decoder_inputs = layers.Conv1D(filters,3,activation='relu',padding='same',name='decoder_conv1D')(decoder_inputs)
    decoder_inputs = layers.BatchNormalization()(decoder_inputs)
    decoder_inputs = layers.Dense(filters, activation='relu')(decoder_inputs)
    decoder_inputs = layers.Dense(filters)(decoder_inputs)
    decoder_inputs = layers.BatchNormalization()(decoder_inputs)

    decoder_lstm = layers.LSTM(128, return_sequences=True)
    decoder_inputs = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_inputs = layers.BatchNormalization()(decoder_inputs)
    decoder_inputs = layers.Dense(filters*2, activation='relu')(decoder_inputs)
    decoder_inputs = layers.Dense(filters*2)(decoder_inputs)
    decoder_inputs = layers.BatchNormalization()(decoder_inputs)
    
    decoder_inputs = layers.LSTM(128, return_sequences=True)(decoder_inputs, initial_state=encoder_states)
    decoder_inputs = layers.BatchNormalization()(decoder_inputs)
    decoder_inputs = layers.Dense(filters*2, activation='relu')(decoder_inputs)
    decoder_inputs = layers.Dense(filters*2)(decoder_inputs)
    decoder_inputs = layers.BatchNormalization()(decoder_inputs)

    decoder_inputs = layers.LSTM(128, return_sequences=True)(decoder_inputs, initial_state=encoder_states)
    decoder_inputs = layers.BatchNormalization()(decoder_inputs)
    decoder_inputs = layers.Dense(filters*2, activation='relu')(decoder_inputs)
    decoder_inputs = layers.Dense(filters*2)(decoder_inputs)
    decoder_inputs = layers.BatchNormalization()(decoder_inputs)

    decoder_inputs = layers.Dense(classes, activation='relu')(decoder_inputs)
    decoder_outputs = layers.Dense(classes, activation='softmax')(decoder_inputs)

    model = Model(inputs=[inputs1,inputs2], outputs=[encoder_outputs, decoder_outputs])
    
    return model


def model_EcgSeq1(filters,times,dim,classes):

    #position_inputs = layers.Input(shape=(times, 1))
    #inputs = layers.Multiply()([data_inputs,position_inputs])
    
    inputs1 = layers.Input(shape=(times, dim))
    encoder_inputs = layers.Conv1D(filters,3,activation='relu',padding='same')(inputs1)
    encoder_inputs = layers.Dense(filters,activation='relu')(encoder_inputs)
    encoder_inputs = layers.BatchNormalization()(encoder_inputs)
    encoder_inputs_0 = encoder_inputs
    
    encoder_inputs = layers.Conv1D(filters*2,5,activation='relu',padding='same')(inputs1)
    encoder_inputs = layers.Dense(filters,activation='relu')(encoder_inputs)
    encoder_inputs = layers.BatchNormalization()(encoder_inputs)

    encoder_inputs = layers.Bidirectional(layers.LSTM(filters*2, return_sequences=True))(encoder_inputs)
    encoder_inputs = layers.Dense(filters,activation='relu')(encoder_inputs)
    encoder_inputs = layers.BatchNormalization()(encoder_inputs)

    encoder_inputs = layers.Bidirectional(layers.LSTM(filters*2, return_sequences=True))(encoder_inputs)
    encoder_inputs = layers.Dense(filters,activation='relu')(encoder_inputs)
    encoder_inputs = layers.BatchNormalization()(encoder_inputs)
    
    encoder_inputs += encoder_inputs_0
    encoder_inputs = layers.BatchNormalization()(encoder_inputs)
    
    encoder_outputs,forward_h, forward_c, backward_h, backward_c = layers.Bidirectional(
    layers.LSTM(filters*2, activation='relu', return_sequences=True, return_state=True))(encoder_inputs)
    
    state_h = layers.Concatenate(axis=-1)([forward_h, backward_h])
    state_c = layers.Concatenate(axis=-1)([forward_c, backward_c])   
    encoder_states = ([state_h, state_c])
    
    encoder_outputs = layers.Dense(classes, activation='relu')(encoder_outputs)
    encoder_outputs = layers.Dense(classes, activation='softmax')(encoder_outputs)
    
    inputs2 = layers.Input(shape=(times, dim))
    #decoder_inputs = layers.Masking(mask_value=-1000)(inputs2)
    decoder_inputs = layers.Conv1D(filters,3,activation='relu',padding='same')(inputs2)
    decoder_inputs = layers.Dense(filters,activation='relu')(decoder_inputs)
    decoder_inputs = layers.BatchNormalization()(decoder_inputs)
    decoder_inputs_0 = decoder_inputs
    
    decoder_inputs = layers.Conv1D(filters*2,5,activation='relu',padding='same')(inputs2)
    decoder_inputs = layers.Dense(filters,activation='relu')(decoder_inputs)
    decoder_inputs = layers.BatchNormalization()(decoder_inputs)

    decoder_inputs = layers.LSTM(filters*4, return_sequences=True)(decoder_inputs, initial_state=encoder_states)
    decoder_inputs = layers.Dense(filters,activation='relu')(decoder_inputs)
    decoder_inputs = layers.BatchNormalization()(decoder_inputs)
    
    decoder_inputs = layers.LSTM(filters*4, return_sequences=True)(decoder_inputs, initial_state=encoder_states)
    decoder_inputs = layers.Dense(filters,activation='relu')(decoder_inputs)
    decoder_inputs = layers.BatchNormalization()(decoder_inputs)
    decoder_inputs_2 = decoder_inputs
    
    decoder_inputs = layers.LSTM(filters*4, return_sequences=True)(decoder_inputs, initial_state=encoder_states)
    decoder_inputs = layers.Dense(filters,activation='relu')(decoder_inputs)
    decoder_inputs = layers.BatchNormalization()(decoder_inputs)
    
    
    decoder_inputs += decoder_inputs_0
    decoder_inputs = layers.BatchNormalization()(decoder_inputs)
    
    decoder_att = layers.Attention()([encoder_inputs,decoder_inputs])

    decoder_outputs = layers.Concatenate(axis=-1)([encoder_inputs, decoder_att])
    
    
    decoder_outputs = layers.BatchNormalization()(decoder_outputs)

    decoder_outputs = layers.Dense(classes, activation='relu')(decoder_outputs)
    decoder_outputs = layers.Dense(classes, activation='softmax')(decoder_outputs)

    model = Model(inputs=[inputs1,inputs2], outputs=[encoder_outputs, decoder_outputs])
    
    return model
    
def model_EcgAtt(filters,times,dim,classes):

    #position_inputs = layers.Input(shape=(times, 1))
    #inputs = layers.Multiply()([data_inputs,position_inputs])
    
    inputs = layers.Input(shape=(times, dim))

    encoder_inputs = layers.Conv1D(filters,3,activation='relu',padding='same')(inputs)
    encoder_inputs = layers.Dense(filters,activation='relu')(encoder_inputs)
    encoder_inputs = layers.BatchNormalization()(encoder_inputs)
    encoder_inputs_0 = encoder_inputs

    encoder_inputs = layers.Conv1D(filters*2,5,activation='relu',padding='same')(encoder_inputs)
    encoder_inputs = layers.Dense(filters,activation='relu')(encoder_inputs)
    encoder_inputs = layers.BatchNormalization()(encoder_inputs)
    encoder_inputs_1 = encoder_inputs
    
    encoder_inputs = layers.Conv1D(filters*3,7,activation='relu',padding='same')(encoder_inputs)
    encoder_inputs = layers.Dense(filters,activation='relu')(encoder_inputs)
    encoder_inputs = layers.BatchNormalization()(encoder_inputs)
    encoder_inputs_2 = encoder_inputs

    encoder_inputs = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(encoder_inputs)
    encoder_inputs = layers.Dense(filters,activation='relu')(encoder_inputs)
    encoder_inputs = layers.BatchNormalization()(encoder_inputs)

    encoder_inputs = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(encoder_inputs)
    encoder_inputs = layers.Dense(filters,activation='relu')(encoder_inputs)
    encoder_inputs = layers.BatchNormalization()(encoder_inputs)
    
    encoder_inputs += encoder_inputs_0
    encoder_inputs += encoder_inputs_1
    encoder_inputs += encoder_inputs_2
    
    encoder_inputs = layers.BatchNormalization()(encoder_inputs)
    
    encoder_outputs = layers.Dense(filters*2, activation='relu')(encoder_inputs)
    
    encoder_inputs = layers.Attention()([encoder_inputs,encoder_inputs])
    
    encoder_inputs = layers.Dense(filters, activation='relu')(encoder_inputs)
    encoder_outputs = layers.Dense(classes, activation='softmax')(encoder_inputs)
    
    model = Model(inputs=inputs, outputs=encoder_outputs)
    
    
    return model


def model_VGGATT(filters,times,dim,classes):

    inputs = layers.Input(shape=(times, dim))
    vgg_outputs = layers.Conv1D(filters,3,activation='relu',padding='same')(inputs)
    vgg_outputs = layers.Conv1D(filters,3,activation='relu',padding='same')(vgg_outputs)
    vgg_outputs = layers.MaxPooling1D(2,1,padding='same')(vgg_outputs)
    vgg_outputs = layers.BatchNormalization()(vgg_outputs)

    '''    
    vgg_inputs = layers.Conv1D(filters*2,3,activation='relu',padding='same')(vgg_inputs)
    vgg_inputs = layers.Conv1D(filters*2,3,activation='relu',padding='same')(vgg_inputs)
    vgg_inputs = layers.MaxPooling1D(2,1,padding='same')(vgg_inputs)
    
    vgg_inputs = layers.Conv1D(filters*4,3,activation='relu',padding='same')(vgg_inputs)
    vgg_inputs = layers.Conv1D(filters*4,3,activation='relu',padding='same')(vgg_inputs)
    vgg_inputs = layers.Conv1D(filters*4,3,activation='relu',padding='same')(vgg_inputs)
    vgg_inputs = layers.MaxPooling1D(2,1,padding='same')(vgg_inputs)
    
    vgg_inputs = layers.BatchNormalization()(vgg_inputs)
    
    vgg_inputs = layers.Conv1D(filters*8,3,activation='relu',padding='same')(vgg_inputs)
    vgg_inputs = layers.Conv1D(filters*8,3,activation='relu',padding='same')(vgg_inputs)
    vgg_inputs = layers.Conv1D(filters*8,3,activation='relu',padding='same')(vgg_inputs)
    vgg_inputs = layers.MaxPooling1D(2,1,padding='same')(vgg_inputs)
    
    vgg_inputs = layers.Conv1D(filters*8,3,activation='relu',padding='same')(vgg_inputs)
    vgg_inputs = layers.Conv1D(filters*8,3,activation='relu',padding='same')(vgg_inputs)
    vgg_inputs = layers.Conv1D(filters*8,3,activation='relu',padding='same')(vgg_inputs)
    vgg_inputs = layers.MaxPooling1D(2,1,padding='same')(vgg_inputs)

    vgg_inputs = layers.BatchNormalization()(vgg_inputs)
    #vgg_inputs = layers.Flatten()(vgg_inputs)
    vgg_inputs = Dense(filters*2,activation="relu")(vgg_inputs)
    vgg_inputs = layers.Dropout(0.2)(vgg_inputs)
    vgg_inputs = Dense(filters*2,activation="relu")(vgg_inputs)
    vgg_inputs = layers.Dropout(0.2)(vgg_inputs)
    vgg_inputs = layers.BatchNormalization()(vgg_inputs)
    '''
    
    lstm_outputs = layers.Bidirectional(layers.LSTM(filters*1, return_sequences=True))(vgg_outputs)
    lstm_outputs = Dense(filters*1,activation="relu")(lstm_outputs)
    lstm_outputs = layers.MaxPooling1D(2,1,padding='same')(lstm_outputs)
    lstm_outputs = layers.BatchNormalization()(lstm_outputs)

    '''
    vgg_inputs = layers.Bidirectional(layers.LSTM(filters*2, return_sequences=True))(vgg_inputs)
    vgg_inputs = Dense(filters*2,activation="relu")(vgg_inputs)
    vgg_inputs = layers.MaxPooling1D(2,1,padding='same')(vgg_inputs)
    vgg_inputs = layers.BatchNormalization()(vgg_inputs)

    encoder_outputs,forward_h, forward_c, backward_h, backward_c = layers.Bidirectional(
    layers.LSTM(filters, activation='relu', return_sequences=True, return_state=True))(vgg_inputs)
    
    state_h = layers.Concatenate(axis=-1)([forward_h, backward_h])
    state_c = layers.Concatenate(axis=-1)([forward_c, backward_c])   
    encoder_states = ([state_h, state_c])


    decoder_outputs = layers.LSTM(filters*2, return_sequences=True)(encoder_outputs, initial_state=encoder_states)
    decoder_outputs = Dense(filters*2,activation="relu")(decoder_outputs)
    decoder_outputs = layers.MaxPooling1D(2,1,padding='same')(decoder_outputs)
    decoder_outputs = layers.BatchNormalization()(decoder_outputs)

    decoder_outputs = layers.LSTM(filters*2, return_sequences=True)(decoder_outputs, initial_state=encoder_states)
    decoder_outputs = Dense(filters*2,activation="relu")(decoder_outputs)
    decoder_outputs = layers.MaxPooling1D(2,1,padding='same')(decoder_outputs)
    decoder_outputs = layers.BatchNormalization()(decoder_outputs)

    decoder_outputs = layers.LSTM(filters*2, return_sequences=True)(decoder_outputs, initial_state=encoder_states)
    decoder_outputs = Dense(filters*2,activation="relu")(decoder_outputs)
    decoder_outputs = layers.MaxPooling1D(2,1,padding='same')(decoder_outputs)
    decoder_outputs = layers.BatchNormalization()(decoder_outputs)
    '''
    
    vgg_outputs_att = layers.Attention()([vgg_outputs,lstm_outputs])

    final_outputs = layers.Concatenate(axis=-1)([vgg_outputs, vgg_outputs_att])

    final_outputs = layers.BatchNormalization()(final_outputs)
    final_outputs = layers.MaxPooling1D(2,1,padding='same')(final_outputs)
    final_outputs = Dense(filters,activation="relu")(final_outputs)
    final_outputs = layers.Dropout(0.2)(final_outputs)
    final_outputs = layers.MaxPooling1D(2,1,padding='same')(final_outputs)
    final_outputs = layers.BatchNormalization()(final_outputs)
    final_outputs = layers.Dense(classes, activation='softmax')(final_outputs)
    
    model = Model(inputs=inputs, outputs=final_outputs)

    return model


def model_VGG16(filters,times,dim,classes):

    inputs = layers.Input(shape=(times, dim))
    vgg_inputs = layers.Conv1D(filters,3,activation='relu',padding='same')(inputs)
    vgg_inputs = layers.Conv1D(filters,3,activation='relu',padding='same')(vgg_inputs)
    vgg_inputs = layers.MaxPooling1D(2,1,padding='same')(vgg_inputs)
    
    vgg_inputs = layers.Conv1D(filters*2,3,activation='relu',padding='same')(vgg_inputs)
    vgg_inputs = layers.Conv1D(filters*2,3,activation='relu',padding='same')(vgg_inputs)
    vgg_inputs = layers.MaxPooling1D(2,1,padding='same')(vgg_inputs)
    
    vgg_inputs = layers.Conv1D(filters*4,3,activation='relu',padding='same')(vgg_inputs)
    vgg_inputs = layers.Conv1D(filters*4,3,activation='relu',padding='same')(vgg_inputs)
    vgg_inputs = layers.Conv1D(filters*4,3,activation='relu',padding='same')(vgg_inputs)
    vgg_inputs = layers.MaxPooling1D(2,1,padding='same')(vgg_inputs)
    
    vgg_inputs = layers.Conv1D(filters*8,3,activation='relu',padding='same')(vgg_inputs)
    vgg_inputs = layers.Conv1D(filters*8,3,activation='relu',padding='same')(vgg_inputs)
    vgg_inputs = layers.Conv1D(filters*8,3,activation='relu',padding='same')(vgg_inputs)
    vgg_inputs = layers.MaxPooling1D(2,1,padding='same')(vgg_inputs)
    
    vgg_inputs = layers.Conv1D(filters*8,3,activation='relu',padding='same')(vgg_inputs)
    vgg_inputs = layers.Conv1D(filters*8,3,activation='relu',padding='same')(vgg_inputs)
    vgg_inputs = layers.Conv1D(filters*8,3,activation='relu',padding='same')(vgg_inputs)
    vgg_inputs = layers.MaxPooling1D(2,1,padding='same')(vgg_inputs)

    #vgg_inputs = layers.Flatten()(vgg_inputs)
    vgg_inputs = Dense(filters*2,activation="relu")(vgg_inputs)
    vgg_inputs = layers.Dropout(0.2)(vgg_inputs)
    vgg_inputs = Dense(filters*2,activation="relu")(vgg_inputs)
    vgg_inputs = layers.Dropout(0.2)(vgg_inputs)

    #vgg_inputs = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(vgg_inputs)
    #vgg_inputs = Dense(64,activation="relu")(vgg_inputs)

    vgg_inputs = Dense(classes,activation="softmax")(vgg_inputs)
    
    model = Model(inputs=inputs, outputs=vgg_inputs)
    return model
    
    #model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    #model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    #model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    #model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    #model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    #model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    #model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    #model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    #model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    #model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    #model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    #model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    #model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    #model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    #model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    #model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    #model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    #model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    #model.add(Flatten())
    #model.add(Dense(units=4096,activation="relu"))
    #model.add(Dense(units=4096,activation="relu"))
    #model.add(Dense(units=2, activation="softmax"))

"""
如果第一次看不懂，那么请反复阅读几次，这个代码包含了Keras中实现最一般模型的思路：把目标当成一个输入，构成多输入模型，把loss写成一个层，作为最后的输出，
搭建模型的时候，就只需要将模型的output定义为loss，而compile的时候，直接将loss设置为y_pred（因为模型的输出就是loss，所以y_pred就是loss），
无视y_true，训练的时候，y_true随便扔一个符合形状的数组进去就行了。最后我们得到的是问题和答案的编码器，也就是问题和答案都分别编码出一个向量来，我们只需要比较lcos，就可以选择最优答案了。
"""
def model_Test():
    word_size = 128
    nb_features = 10000
    nb_classes = 10
    encode_size = 64
    margin = 0.1

    embedding = layers.Embedding(nb_features,word_size)
    lstm_encoder = layers.LSTM(encode_size)

    def encode(input):
        return lstm_encoder(embedding(input))

    q_input = layers.Input(shape=(None,))
    a_right = layers.Input(shape=(None,))
    a_wrong = layers.Input(shape=(None,))
    q_encoded = encode(q_input)
    a_right_encoded = encode(a_right)
    a_wrong_encoded = encode(a_wrong)

    q_encoded = layers.Dense(encode_size)(q_encoded) #一般的做法是，直接讲问题和答案用同样的方法encode成向量后直接匹配，但我认为这是不合理的，我认为至少经过某个变换。

    right_cos = layers.dot([q_encoded,a_right_encoded], -1, normalize=True)
    wrong_cos = layers.dot([q_encoded,a_wrong_encoded], -1, normalize=True)

    loss = layers.Lambda(lambda x: backend.relu(margin+x[0]-x[1]))([wrong_cos,right_cos])

    model_train = Model(inputs=[q_input,a_right,a_wrong], outputs=loss)
    model_q_encoder = Model(inputs=q_input, outputs=q_encoded)
    model_a_encoder = Model(inputs=a_right, outputs=a_right_encoded)

    model_train.compile(optimizer='adam', loss=lambda y_true,y_pred: y_pred)
    model_q_encoder.compile(optimizer='adam', loss='mse')
    model_a_encoder.compile(optimizer='adam', loss='mse')
    model_train.summary()
    model_q_encoder.summary()
    model_q_encoder.summary()
    
    return model_q_encoder
    #model_train.fit([q,a1,a2], y, epochs=10)
    #其中q,a1,a2分别是问题、正确答案、错误答案的batch，y是任意形状为(len(q),1)的矩阵
    inputs = tf.keras.Input(shape=(3,))
    x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
    outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)