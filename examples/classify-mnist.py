import os
os.chdir('..')

import sys
sys.path.insert(0, './python')
import caffe

import matplotlib
matplotlib.rcParams['backend'] = "Qt4Agg"
import numpy as np
import lmdb

MODEL_FILE = 'examples/mnist/lenet.prototxt'
PRETRAINED = 'examples/mnist/lenet_iter_10000.caffemodel'

net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
caffe.set_mode_cpu()
# Test self-made image
"""
img = caffe.io.load_image('examples/images/two_g.jpg', color=False)
img = img.astype(np.uint8)
out = net.forward_all(data=np.asarray([img.transpose(2,0,1)]))
print out['prob'][0]
"""
db_path = 'examples/mnist/mnist_test_lmdb'
lmdb_env = lmdb.open(db_path)
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
count = 0
correct = 0
for key, value in lmdb_cursor:
    print "Count:"
    print count
    count += 1
    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(value)
    label = int(datum.label)

    image = caffe.io.datum_to_array(datum)
    image = image.astype(np.uint8)
    out = net.forward_all(data=np.asarray([image]))
    predicted_label = out['prob'][0].argmax(axis=0)
    print out['prob']
    if label == predicted_label:
        correct += 1
    print("Label is class " + str(label) + ", predicted class is " + str(predicted_label))

print(str(correct) + " out of " + str(count) + " were classified correctly")
