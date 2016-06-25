
# coding: utf-8

# In[2]:

from statsmodels.tsa.ar_model import AR
import pywt
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
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
ecg_data = data_maker.make_by_r_peak(before_r = 90, after_r = 146, sample_num = sample_num, label_maker = test_label_maker,                                      label_percent = label_percent, train_percent = train_percent)
ecg_data.print_summary()


# In[5]:
db8 = pywt.Wavelet('db8')
fea_train = []
for i in xrange(ecg_data.train.x.shape[0]):
    cA3, cD3, cD2, cD1 = pywt.wavedec(ecg_data.train.x[i].tolist(), db8, level = 3)
    model = AR(ecg_data.train.x[i].tolist())
    result = model.fit(maxlag = 4)
    fea_train.append(ecg_data.train.x[i].tolist() + cA3.tolist() + result.params[1:].tolist())

fea_test = []
for i in xrange(ecg_data.test.x.shape[0]):
    cA3, cD3, cD2, cD1 = pywt.wavedec(ecg_data.test.x[i].tolist(), db8, level = 3)
    model = AR(ecg_data.test.x[i].tolist())
    result = model.fit(maxlag = 4)
    fea_test.append(ecg_data.test.x[i].tolist() + cA3.tolist() + result.params[1:].tolist())
        
print np.array(fea_train).shape
print np.array(fea_test).shape


# In[111]:



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
    
def weight_var(shape, name):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial, name = name)

def bias_var(shape, name):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial, name = name)

x = tf.placeholder(tf.float32, [None, 282])
y_ = tf.placeholder(tf.float32, [None])
y_onehot = tf.one_hot(tf.cast(y_, tf.int64), 6, 1.0, 0.0)

#隐层
W_h = weight_var([282, 32], "weight_h")
b_h = bias_var([32], "bais_h")
h = tf.nn.relu(tf.matmul(x, W_h) + b_h)

#输出层
W_o = weight_var([32, 6], "weight_o")
b_o = bias_var([6], "bias_o")
o = tf.nn.softmax(tf.matmul(h, W_o) + b_o)

cross_entropy = -tf.reduce_sum(y_onehot * tf.log(o))
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

corr_pred = tf.equal(tf.arg_max(o, 1), tf.arg_max(y_onehot, 1))
accuracy = tf.reduce_mean(tf.cast(corr_pred, tf.float32))

saver = tf.train.Saver(max_to_keep = 100)

#sess = tf.InteractiveSession()
sess = tf.Session()

train_or_test = "test"
check_point_dir = "../models/dwt_ar_nn/"
sess.run(tf.initialize_all_variables())

if train_or_test == "train":
    for i in xrange(200000):
        batch = ecg_data.train.next_batch(50)
    
        if i % 1000 == 0:
            train_accuacy = sess.run(accuracy, feed_dict={x: np.array(fea_test), y_: ecg_data.test.y})
            saver.save(sess, check_point_dir + "model.mdl", global_step = i)
            print("step %d, training accuracy %g"%(i, train_accuacy))
        
        db8 = pywt.Wavelet('db8')
        fea_train = []
        for i in xrange(batch[0].shape[0]):
            cA3, cD3, cD2, cD1 = pywt.wavedec(batch[0][i].tolist(), db8, level = 3)
            model = AR(batch[0][i].tolist())
            result = model.fit(maxlag = 4)
            fea_train.append(batch[0][i].tolist() + cA3.tolist() + result.params[1:].tolist())

        sess.run(train_step, feed_dict = {x : np.array(fea_train), y_ : batch[1]})
elif train_or_test == "test":
    global_step = 28000
    show_before = 1200
    show_after = 1200
    saver.restore(sess, check_point_dir + "model.mdl-" + str(global_step))
    pred = tf.arg_max(o, 1)
    pred_y = sess.run(pred, feed_dict={x: np.array(fea_test)})
    for i in xrange(ecg_data.test.x.shape[0]):
        rec_no = ecg_data.test.src[i]["rec_no"]
        start = ecg_data.test.src[i]["start"]
        
        show_start = start - show_before
        if show_start < 0:
            show_start = 0
        show_end = start + 236 + show_after - 1
        if show_end >= len(data_maker.raw_data[rec_no]["signal"]):
            show_end = len(data_maker.raw_data[rec_no]["signal"]) - 1
        
        figure = plt.figure(figsize=(20,4)) 
        x = range(0, start - show_start)
        plt.plot(x, data_maker.raw_data[rec_no]["signal"][show_start:start])
        x = range(start - show_start, start - show_start + 236)
        plt.plot(x, data_maker.raw_data[rec_no]["signal"][start:start+236], 'r')
        x = range(start - show_start + 236, show_end - show_start + 1)
        plt.plot(x, data_maker.raw_data[rec_no]["signal"][start+236:show_end+1], 'b')
        
        true_name = get_label_name_cn(ecg_data.test.y[i])
        pred_name = get_label_name_cn(pred_y[i])
        plt.title(u"Rec_no: %s, Start: %s, 人工: %s, 算法: %s" % (str(rec_no), str(start), true_name.decode("utf8"), pred_name.decode("utf8")), fontproperties=zhfont)
        plt.grid()
        plt.xticks([j for j in xrange(0, show_end - show_start + 1, 40)])
        true_name = get_label_name_en(ecg_data.test.y[i])
        pred_name = get_label_name_en(pred_y[i])
        if pred_y[i] == ecg_data.test.y[i]:
            file_name = "../cases/mit-bih/corr/" + "-".join([str(rec_no), str(start), true_name, pred_name])
        else:
            file_name = "../cases/mit-bih/incorr/" + "-".join([str(rec_no), str(start), true_name, pred_name])
        plt.savefig(file_name)
        plt.close(figure)
            

