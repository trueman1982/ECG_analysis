# coding: utf-8

get_ipython().magic(u'matplotlib nbagg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
from mit_bih_data_maker import mit_bih_data_maker

zhfont = mpl.font_manager.FontProperties(fname='/usr/share/fonts/truetype/arphic/uming.ttc')

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
points_before = 360
points_after = 360
ecg_data = data_maker.make_by_r_peak(before_r = points_before, after_r = points_after, sample_num = sample_num, label_maker = test_label_maker,                                      label_percent = label_percent, train_percent = train_percent)
ecg_data.print_summary()

tf.reset_default_graph()

def get_label_name_en(l):
    label_map = {0:"Normal", 1:"LBundle", 2:"RBundle", 3:"Pace", 4:"PV", 5:"AP"}
    if l in label_map:
        return label_map[l]
    else:
        return "Unknown"

def get_label_name_cn(l):
    label_map = {0:"正常", 1:"左束肢阻塞", 2:"右束肢阻塞", 3:"起搏心跳", 4:"室性早搏", 5:"房性早搏"}
    if l in label_map:
        return label_map[l]
    else:
        return "Unknown"
    
def weight_var(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_var(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = "SAME")

def max_pool(x, pool_size):
    return tf.nn.max_pool(x, ksize = [1, pool_size[0], pool_size[1], 1], strides = [1, pool_size[0], pool_size[1], 1], padding = "SAME")


x = tf.placeholder(tf.float32, [None, points_before + points_after])
y_ = tf.placeholder(tf.float32, [None])
y_onehot = tf.one_hot(tf.cast(y_, tf.int64), 6, 1.0, 0.0)

#卷积层
conv_out_channel_1 = 4
conv_out_channel_2 = 64
conv_size_1 = 24
conv_size_2 = 5
pooling_rate_1 = 12
pooling_rate_2 = 2
fc_hidden_1 = 32

#第一层卷积
W_conv1 = weight_var([conv_size_1, 1, 1, conv_out_channel_1])
b_conv1 = bias_var([conv_out_channel_1])

x_reshape = tf.reshape(x, [-1, points_before + points_after, 1, 1])

h_conv1 = tf.nn.relu(conv2d(x_reshape, W_conv1) + b_conv1)
h_pool1 = max_pool(h_conv1, (pooling_rate_1, 1))

#隐层
size_after_pool = (points_before + points_after) / pooling_rate_1 * conv_out_channel_1
W_fc1 = weight_var([size_after_pool, fc_hidden_1])
b_fc1 = bias_var([fc_hidden_1])
h_pool1_flat = tf.reshape(h_pool1, [-1, size_after_pool])
h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#输出层
W_o = weight_var([fc_hidden_1, 6])
b_o = bias_var([6])
o = tf.nn.softmax(tf.matmul(h_fc1_drop, W_o) + b_o)

cross_entropy = -tf.reduce_sum(y_onehot * tf.log(tf.clip_by_value(o, 0.001, 1)))
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

corr_pred = tf.equal(tf.arg_max(o, 1), tf.arg_max(y_onehot, 1))
accuracy = tf.reduce_mean(tf.cast(corr_pred, tf.float32))

saver = tf.train.Saver(max_to_keep = 100)

sess = tf.InteractiveSession()
#sess = tf.Session()

train_or_test = "train"
check_point_dir = "../models/nn_720window/"
sess.run(tf.initialize_all_variables())

print h_pool1_flat.eval(feed_dict={x: ecg_data.train.x, y_: ecg_data.train.y}).shape

if train_or_test == "train":
    for i in xrange(200000):
        batch = ecg_data.train.next_batch(50)

    
        if i % 1000 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x: ecg_data.train.x, y_: ecg_data.train.y, keep_prob: 1})
            test_accuracy = sess.run(accuracy, feed_dict={x: ecg_data.test.x, y_: ecg_data.test.y, keep_prob: 1})
            #saver.save(sess, check_point_dir + "model.mdl", global_step = i)
            print("step %d, training accuracy %g, testing accuracy %g"%(i, train_accuracy, test_accuracy))

        sess.run(train_step, feed_dict = {x : batch[0], y_ : batch[1], keep_prob: 0.5})
