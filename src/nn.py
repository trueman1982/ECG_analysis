
# coding: utf-8

# In[26]:

import tensorflow as tf
from mit_bih_data_maker import mit_bih_data_maker

def test_label_maker(label):
    label_map = {"N":0, "L":1, "R":2, "/":3, "V":4, "A":5}
    if label in label_map:
        return label_map[label]
    return "Omit"

data_maker = mit_bih_data_maker()
data_maker.config("../database/mit-bih/", "../database/mit-bih", "../database/mit-bih-r-label")
#rec_list = ["100", "109", "118", "107", "208", "232"]
rec_list = []
data_maker.load_data(rec_list)

sample_num = -1
label_percent = {0:0.197, 1:0.220, 2:0.191, 3:0.183, 4:0.087, 5:0.122}
train_percent = 0.7
ecg_data = data_maker.make_by_r_peak(before_r = 90, after_r = 146, sample_num = sample_num, label_maker = test_label_maker,                                      label_percent = label_percent, train_percent = train_percent)
ecg_data.print_summary()


# In[27]:

import numpy as np

def weight_var(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_var(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 236])
y_ = tf.placeholder(tf.float32, [None])
y_onehot = tf.one_hot(tf.cast(y_, tf.int64), 6, 1.0, 0.0)

#隐层
W_h = weight_var([236, 32])
b_h = bias_var([32])
h = tf.nn.relu(tf.matmul(x, W_h) + b_h)

#输出层
W_o = weight_var([32, 6])
b_o = bias_var([6])
o = tf.nn.softmax(tf.matmul(h, W_o) + b_o)

cross_entropy = -tf.reduce_sum(y_onehot * tf.log(o))
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

corr_pred = tf.equal(tf.arg_max(o, 1), tf.arg_max(y_onehot, 1))
accuracy = tf.reduce_mean(tf.cast(corr_pred, tf.float32))

sess.run(tf.initialize_all_variables())

for i in xrange(200000):
    batch = ecg_data.train.next_batch(50)
    
    if i % 1000 == 0:
        train_accuacy = accuracy.eval(feed_dict={x: ecg_data.test.x, y_: ecg_data.test.y})
        print("step %d, training accuracy %g"%(i, train_accuacy))
        
    sess.run(train_step, feed_dict = {x : batch[0], y_ : batch[1]})

