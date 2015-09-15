#!/usr/bin/python2
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
SHOW = False


plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def rgb2gray(rgb):
    return 1-np.dot(rgb[..., :3], [0.299, 0.587, 0.144])[None, ...]


# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, padsize=1, padval=0, t=None):
    data = data.copy()
    data -= data.min()
    data /= data.max()

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = data.squeeze()

    imshow(data)
    if t:
        title(t)
    if SHOW:
        show()


net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)

vis_square(net.params['conv1'][0].data.transpose(0, 2, 3, 1), t='conv1 filters')
vis_square(net.params['conv2'][0].data[:20].reshape(20**2, 5, 5), t='conv2 filters')

def forward(image):
    out = net.forward_all(data=np.asarray([image]))
    vis_square(net.blobs['conv1'].data[0, :36], padval=1, t='conv1')
    vis_square(net.blobs['pool1'].data[0, :36], padval=1, t='pool1')
    vis_square(net.blobs['conv2'].data[0], padval=1, t='conv2')
    vis_square(net.blobs['pool2'].data[0], padval=0.5, t='pool1')
    vis_square(net.blobs['ip1'].data[0, :400].reshape(1, 20, 20), padval=0.5, t='ip1')
    imshow(net.blobs['ip2'].data[0].reshape(1, 10))
    xticks(np.arange(0, 10, 1.0))
    title('ip2')
    if SHOW:
        show()
    return out

caffe.set_mode_cpu()
if len(sys.argv) > 1 and sys.argv[1] == "-i":
    # Test self-made image
    f = sys.argv[2] if len(sys.argv) > 2 else 'examples/images/3.jpg'
    img = caffe.io.load_image(f, color=False).astype(np.uint8)
    img = rgb2gray(imresize(img, (28, 28)))
    imshow(img[0], cmap=cm.Greys_r)
    show()
    out = forward(img)
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
    print "n=%-5d accuracy=%.3f label=%d predicted=%d prob=%s" %\
          (i + 1, float(correct) / (i + 1), label, predicted_label, out['prob'][0])
