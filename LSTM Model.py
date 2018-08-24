
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import time
import random

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops

from pandas_ml import ConfusionMatrix


# ## load all npz files from npz_files directory.

# In[2]:


import os

npz_files_directory_train = './npz-files/npz-files/train'
npz_files_directory_test = './npz-files/npz-files/test'
# collect all files from npz directory.
train_files = list()
for f in os.listdir(npz_files_directory_train):
    train_files.append(f)
    
test_files = list()
for f in os.listdir(npz_files_directory_test):
    test_files.append(f)


# In[3]:


train_files = [file for file in train_files if file[-3:] == 'npz']
test_files = [file for file in test_files if file[-3:] == 'npz']


# In[4]:


print('number of train files', len(train_files))
print('number of test files', len(test_files))


# ## load all numpy arrays into training data x and y.
# structure of the npz files = ['c1', 'c2', 'c3', 'Z', 'labels'].
# c1, c2, c3 => numpy array containing all class 1, 2, 3 bounding boxes respectively.
# Z => the raw PSD files.
# labels => pixel wise labels for psd files. same dimensions as the PSD files.

# In[5]:


def load_data(directory, files_list):
    training_data_x = list()
    training_data_y = list()

    count = 0
    for file_name in files_list:
        data1 = np.load(directory + '/' + file_name)
        training_data_x.append(data1['Z'])
        training_data_y.append(data1['labels'])
        count += 1
    print('files loaded', count)
    return training_data_x, training_data_y 


# ## convert training_data_x to shape (size of all timesteps, 512)

# In[6]:


def processFrequencies(freq, num_steps):
    frequencies = np.zeros((512, num_steps), dtype=np.float32)
    assert freq.shape[0] == 512
    freq = np.reshape(freq, (512, 1))
    for i in range(freq.shape[0]):
        if freq[i:i+num_steps, 0].shape[0] == num_steps:
            frequencies[i, :] = freq[i:i+num_steps, 0]
        else:
#             print(freq[i:i+num_steps, 0].shape[0])
            frequencies[i, :] = np.pad(freq[i:i+num_steps, 0], (0, num_steps-freq[i:i+num_steps, 0].shape[0]), 'edge')
    frequencies = np.reshape(frequencies, (512, num_steps, 1))
    return frequencies


# In[7]:


def processLabels(labels):
    labels = np.reshape(labels, (512, 1))
    labels_reshaped = np.zeros((512, 4))
    for i in range(labels.shape[0]):
        labels_reshaped[i, labels[i, 0]] = 1
    return labels_reshaped


# ## Convert all x_train data to serial x_train data.

# In[8]:


def convertToSerialList(x, y, x_copy, y_copy):
    assert len(x) == len(y)
    for i in range(len(x)):
        for j in range(x[i].shape[0]):
            x_copy.append(processFrequencies(x[i][j,:] , num_steps))
            y_copy.append(processLabels(y[i][j,:]))


# In[9]:


def normalizePSD(psd, new_min=-1, new_max=1):
    psd_max = np.max(psd)
    psd_min = np.min(psd)
    scaling_factor = (new_max - new_min) / (psd_max - psd_min)
    psd *= scaling_factor
    psd -= new_min
    return psd


# In[10]:


def plotPSD(psd):
    plt.plot(psd[0])
    plt.show()


# ## Creating the RNN model.

# In[11]:


# RNN Global Parameter models.
num_steps = 8
frequency_shape = [512, num_steps, 1]
labels_shape = [512, 4]

hidden_layer_dimension = 32
number_layers = 2
use_dropout = False
dropout = 0.0
num_layers = 2
batch_size = 128
learning_rate = 0.001
num_epochs = 500
step_size = 512 // batch_size
num_samples_train = 0
num_samples_test = 0


# In[12]:


class Input:
    def __init__(self, is_train=True):
        """
        creates two run able objects -> inputs for feeding inputs and labels.
        """
        self.is_train = is_train
        self.frequencies = None
        self.labels = None
        self.getData()
        self.dataset = tf.data.Dataset.from_tensor_slices((self.frequencies, self.labels))
        if self.is_train:
            self.dataset = self.dataset.shuffle(buffer_size=self.frequencies.shape[0])
        self.dataset = self.dataset.apply(tf.contrib.data.unbatch())
        self.dataset = self.dataset.batch(batch_size)
        self.dataset = self.dataset.prefetch(512 // batch_size)
        
        self.iterator = self.dataset.make_initializable_iterator()
        
    def getOutputType(self):
        return self.dataset.output_types
    
    def getOutputShape(self):
        return self.dataset.output_shapes
    
    def getData(self):
        global num_samples_train, num_samples_test
        if self.is_train:
            training_x, training_y = load_data(npz_files_directory_train, train_files)
#             for index in range(len(training_x)):
#                 plotPSD(training_x[index])
#                 training_x[index] = normalizePSD(training_x[index])
#                 plotPSD(training_x[index])
#                 print(np.max(training_x[index]), np.min(training_x[index]))
            train_x, train_y = list(), list()
            convertToSerialList(training_x, training_y, train_x, train_y)
            self.frequencies = np.array(train_x)
            self.labels = np.array(train_y)
            num_samples_train = self.frequencies.shape[0]
        else:
            testing_x, testing_y = load_data(npz_files_directory_test, test_files)
            test_x, test_y = list(), list()
            convertToSerialList(testing_x, testing_y, test_x, test_y)
            self.frequencies = np.array(test_x)
            self.labels = np.array(test_y)
            num_samples_test = self.frequencies.shape[0]


# In[13]:


# test = Input()


# In[14]:


class Model:
    def __init__(self):
        self.train_dataset = Input(is_train=True)
        self.test_dataset = Input(is_train=False)
        
        self.handle = tf.placeholder(tf.string, [], name='dataset_handler')
        iterator = tf.data.Iterator.from_string_handle(self.handle, 
                                               self.train_dataset.dataset.output_types,
                                               self.train_dataset.dataset.output_shapes)
        input_data, label_data = iterator.get_next()
        self.input_data = input_data
        self.label_data = label_data
        
        self.init_state = tf.placeholder(tf.float32, [number_layers, 2, batch_size, hidden_layer_dimension])
        state_per_layer_list = tf.unstack(self.init_state, axis=0)
        rnn_state_tuples = tuple([tf.nn.rnn_cell.LSTMStateTuple(state[0], state[1]) for state in state_per_layer_list])

        if use_dropout:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout)
            
        self.state = current_state = np.zeros((num_layers, 2, batch_size, hidden_layer_dimension))

        if num_layers > 1:
            cell_list = [self.createLSTMCells() for _ in range(num_layers)]
            cell = tf.nn.rnn_cell.MultiRNNCell(cell_list, state_is_tuple=True)
        elif num_layers == 1:
            cell = createLSTMCells()

        self.output, self.state = tf.nn.dynamic_rnn(cell, 
                                          input_data,
                                          dtype=tf.float32, 
                                          initial_state=rnn_state_tuples)
        
        # extract the last output for time=num_seq from the output.
        self.output = tf.transpose(self.output, [1, 0, 2])
        self.output = tf.gather(self.output, int(self.output.shape[0]-1))
        
        # softmax layer
        dense_layer = tf.layers.dense(self.output, units=4, activation=tf.nn.relu)
        self.logits = tf.layers.dense(dense_layer, units=4, activation=None)
        
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_data,
                                                                        logits=self.logits)
        self.loss = tf.reduce_mean(self.cross_entropy)
        # optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
        
        # measuring accuracy.
        prediction = tf.nn.softmax(self.logits, name='label_prediction')
        self.prediction_class = tf.argmax(prediction, axis=1)
        output_labels_class = tf.argmax(label_data, axis=1)
        matching_prediction = tf.equal(output_labels_class, self.prediction_class)
        self.accuracy = tf.reduce_mean(tf.cast(matching_prediction, tf.float32), name='prediction_accuracy')
        self.accuracy_dict = dict()
        
    def plotAccuracy(self):
        lists = sorted(self.accuracy_dict.items()) # sorted by key, return a list of tuples
        x, y = zip(*lists) # unpack a list of pairs into two tuples
        plt.plot(x, y)
        plt.show()
        
    def saveModel(self, epoch):
        output_node_names = "label_prediction"
        output_graph_definition = graph_util.convert_variables_to_constants(
            sess,
            tf.get_default_graph().as_graph_def(),
            output_node_names.split(",")
        )
        model_save_path = ('./saved_models/model_'+str(epoch)+'.pb')
        with tf.gfile.GFile(model_save_path, "wb") as f:
            f.write(output_graph_definition.SerializeToString())
        
    def createLSTMCells(self, cell_type='LSTM'):
        if cell_type == 'LSTM':
            cell = tf.contrib.rnn.LSTMCell(hidden_layer_dimension, forget_bias=1.0)
            return cell
        elif cell_type == 'GRU':
            cell = tf.contrib.rnn.GRUCell(hidden_layer_dimension)
            return cell
        else:
            print('Invalid Value - createLSTMCells')
            return None
    
    def calculateTestAccuracy(self, epoch):
        sess.run(self.test_dataset.iterator.initializer)
        
        test_accuracy_list = list()
        label_list = list()
        current_state = np.zeros((num_layers, 2, batch_size, hidden_layer_dimension))
        counter = 0
        # looping till all values are consumed
        while True:
            try:
                accuracy, prediction_class, current_state, = sess.run([m.accuracy, m.prediction_class, m.state], 
                                                    feed_dict={m.init_state: current_state, m.handle: testing_handle})
                test_accuracy_list.append(accuracy)
                label_list.append(prediction_class)
                counter += 1
                if counter != 0 and counter % step_size == 0:
                    current_state = np.zeros((num_layers, 2, batch_size, hidden_layer_dimension))
            except tf.errors.OutOfRangeError:
                break
        print('test accuracy at epoch #', epoch, 'is ', (sum(test_accuracy_list) / len(test_accuracy_list)))
        self.accuracy_dict[epoch] = (sum(test_accuracy_list) / len(test_accuracy_list) )
        print(len(label_list))
        return label_list
    
    def printConfusionMatrix(self, prediction):
        prediction_np = np.array(prediction)
        pred_np = np.zeros((num_samples_test, 512))
        row, col = 0, 0
        for i in range(prediction_np.shape[0]):
            pred_np[row, col:col+batch_size] = prediction_np[i]
            col += batch_size
            if col == 512:
                col = 0
                row += 1
        
        y_test_np = np.argmax(self.test_dataset.labels, axis=2)
        y_test_np = np.reshape(y_test_np, [y_test_np.shape[0] * y_test_np.shape[1]])
        pred_np = np.reshape(pred_np, [pred_np.shape[0] * pred_np.shape[1]])
        cm = ConfusionMatrix(y_test_np, pred_np)
        print(cm)


# In[15]:


m = Model()


# In[ ]:


cell_start_time = time.time()
prediction = None
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    counter = 0
    
    training_handle = sess.run(m.train_dataset.iterator.string_handle())
#     print(training_handle)
#     print(training_handle.shape)
    testing_handle = sess.run(m.test_dataset.iterator.string_handle())
    
    for epoch in range(num_epochs):
        print('epoch #', epoch, 'started')
        
        # variables.
        epoch_time = time.time()
        counter = 0
        current_state = np.zeros((num_layers, 2, batch_size, hidden_layer_dimension))
        train_accuracy_list = list()
        
        # loading the dataset
        sess.run(m.train_dataset.iterator.initializer)
        # looping till all values are consumed
        while True:
            try:
                if epoch % 10 == 0:
                    parameter1, accuracy, current_state = sess.run([m.optimizer, m.accuracy, m.state], 
                                                                   feed_dict={m.init_state: current_state, 
                                                                              m.handle: training_handle})
#                     print('step executed ')
                    train_accuracy_list.append(accuracy)
                    counter += 1
                    if counter != 0 and counter % step_size == 0:
                        current_state = np.zeros((num_layers, 2, batch_size, hidden_layer_dimension))   
                else:
                    parameter1, current_state = sess.run([m.optimizer, m.state], 
                                                             feed_dict={m.init_state: current_state, 
                                                                        m.handle: training_handle})
                    counter += 1
                    if counter != 0 and counter % step_size == 0:
                        current_state = np.zeros((num_layers, 2, batch_size, hidden_layer_dimension))
            except tf.errors.OutOfRangeError:
                break
            
        # after training, if epoch number is multiple of 10.
        if epoch % 10 == 0:
            print('train set accuracy at epoch #', epoch, 'is ', (sum(train_accuracy_list) / len(train_accuracy_list)))
            saver.save(sess, './checkpoints/model_checkpoint_'+str(epoch))
            m.saveModel(epoch)
            prediction = m.calculateTestAccuracy(epoch)
            m.printConfusionMatrix(prediction)
        end_time = time.time()
        print('epoch #', epoch, 'ended - ',  end_time - epoch_time)
    saver.save(sess, './checkpoints/model_checkpoint_final')
    prediction = m.calculateTestAccuracy(epoch)
    m.printConfusionMatrix(prediction)
#     m.saveModel(epoch)
cell_end_time = time.time()
print('total time to train the network is ', cell_end_time - cell_start_time)


# In[ ]:


m.printConfusionMatrix(prediction)


# In[ ]:


m.plotAccuracy()


# Predicted      0.0     1.0    2.0    3.0  __all__
# Actual                                           
# 0.0        2591457  122997  23270  11766  2749490
# 1.0         103267   45613   2025   2147   153052
# 2.0          60231    4473   9225   2700    76629
# 3.0          14268    2793   1423   1641    20125
# __all__    2769223  175876  35943  18254  2999296

# Predicted      0.0     1.0    2.0    3.0  __all__
# Actual                                           
# 0.0        2688045   44755  13187   3503  2749490
# 1.0          37150  110823   1467   3612   153052
# 2.0          55988    2004  18186    451    76629
# 3.0           3436   12181    307   4201    20125
# __all__    2784619  169763  33147  11767  2999296
