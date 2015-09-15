import os
os.chdir('..')

import sys
sys.path.insert(0, 'python')
import caffe

from scipy.misc import imresize
from pylab import *
import lmdb

MODEL_FILE = 'examples/mnist/lenet.prototxt'
PRETRAINED = 'examples/mnist/lenet_iter_10000.caffemodel'


def rgb2gray(rgb):
    return 1-np.dot(rgb[..., :3], [0.299, 0.587, 0.144])[None, ...]

net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
caffe.set_mode_cpu()
if len(sys.argv) > 1 and sys.argv[1] == "-i":
    # Test self-made image
    f = sys.argv[2] if len(sys.argv) > 2 else 'examples/images/3.jpg'
    img = caffe.io.load_image(f, color=False).astype(np.uint8)
    img = rgb2gray(imresize(img, (28, 28)))
    imshow(img[0], cmap=cm.Greys_r)
    show()
    out = net.forward_all(data=np.asarray([img]))
    print out['prob'][0].argmax()
    sys.exit(0)

db_path = 'examples/mnist/mnist_test_lmdb'
lmdb_env = lmdb.open(db_path)
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
correct = 0
for i, (key, value) in enumerate(lmdb_cursor):
    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(value)
    label = int(datum.label)

    image = caffe.io.datum_to_array(datum).astype(np.uint8)
    out = net.forward_all(data=np.asarray([image]))
    predicted_label = out['prob'].argmax()
    if label == predicted_label:
        correct += 1
    print "n=%-5d accuracy=%.3f" % (i + 1, float(correct) / (i + 1))
