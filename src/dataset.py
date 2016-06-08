import numpy as np

class numeric_dataset():
    def __init__(self):
        self.x = None
        self.y = None
        self.offset = 0
    
    def next_batch(self, batch_size):
        batch_x = []
        batch_y = []
        for i in xrange(batch_size):
            batch_x.append(self.x[self.offset])
            batch_y.append(self.y[self.offset])
            self.offset = self.offset + 1
            if self.offset >= self.x.shape[0]:
                self.offset = 0

        return (np.array(batch_x), np.array(batch_y))
        
    def print_summary(self):
        print "Sample num: %d, feature num: %d" % (self.x.shape[0], self.x.shape[1])
        each_label_num = {}
        for i in xrange(self.y.size):
            if self.y[i] not in each_label_num:
                each_label_num[self.y[i]] = 0
            each_label_num[self.y[i]] += 1
        print "Sample num for each label:"
        for y in each_label_num:
            print "Label %s: %d" % (str(y), each_label_num[y])
        
    
class ml_dataset():
    def __init__(self):
        self.train = None
        self.test = None
        
    def print_summary(self):
        print "TRAIN Summay"
        self.train.print_summary()
        print "\r\nTEST Summary"
        self.test.print_summary()
