import os
os.chdir('../..')

import sys
sys.path.insert(0, 'python')
import caffe

import time
import datetime
import logging
import flask
import werkzeug
import optparse
import tornado.wsgi
import tornado.httpserver
import numpy as np
import matplotlib.pyplot as plt
import mpld3
from PIL import Image
import cStringIO as StringIO
import urllib
import exifutil

REPO_DIRNAME = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../..')
UPLOAD_FOLDER = '/tmp/caffe_demos_uploads'
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'])

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Obtain the flask app object
app = flask.Flask(__name__)


@app.route('/')
def index():
    return flask.render_template('index.html', has_result=False)


@app.route('/classify_url', methods=['GET'])
def classify_url():
    imageurl = flask.request.args.get('imageurl', '')
    try:
        string_buffer = StringIO.StringIO(
            urllib.urlopen(imageurl).read())
        image = caffe.io.load_image(string_buffer)

    except Exception as err:
        # For any exception we encounter in reading the image, we will just
        # not continue.
        logging.info('URL Image open error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open image from URL.')
        )

    logging.info('Image: %s', imageurl)
    result = app.clf.classify_image(image)
    return flask.render_template(
        'index.html', has_result=True, result=result, imagesrc=imageurl)


@app.route('/classify_upload', methods=['POST'])
def classify_upload():
    try:
        # We will save the file to disk for possible data collection.
        imagefile = flask.request.files['imagefile']
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
            werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(UPLOAD_FOLDER, filename_)
        imagefile.save(filename)
        logging.info('Saving to %s.', filename)
        image = exifutil.open_oriented_im(filename)

    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open uploaded image.')
        )

    result = app.clf.classify_image(image)
    return flask.render_template(
        'index.html', has_result=True, result=result,
        imagesrc=embed_image_html(image)
    )


def embed_image_html(image):
    """Creates an image embedded in HTML base64 format."""
    image_pil = Image.fromarray((255 * image).astype('uint8'))
    image_pil = image_pil.resize((256, 256))
    string_buf = StringIO.StringIO()
    image_pil.save(string_buf, format='png')
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    return 'data:image/png;base64,' + data


def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS
    )


class LeNetClassifier(object):
    default_args = {
        'model_def_file': (
            '{}/examples/mnist/lenet.prototxt'.format(REPO_DIRNAME)),
        'pretrained_model_file': (
            '{}/examples/mnist/lenet_iter_10000.caffemodel'.format(REPO_DIRNAME))
    }
    for key, val in default_args.iteritems():
        if not os.path.exists(val):
            raise Exception(
                "File for {} is missing. Should be at: {}".format(key, val))
    default_args['image_dim'] = 28
    default_args['raw_scale'] = 28.

    def __init__(self, model_def_file, pretrained_model_file,
                 raw_scale, image_dim, gpu_mode):
        logging.info('Loading net and associated files...')
        if gpu_mode:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()
        self.net = caffe.Classifier(
            model_def_file, pretrained_model_file,
            image_dims=(image_dim, image_dim), raw_scale=raw_scale
        )

    def classify_image(self, image):
        try:
            starttime = time.time()
            scores = self.net.predict([rgb2gray(image)], oversample=True).flatten()
            endtime = time.time()

            predictions = (-scores).argsort()[:5]

            # In addition to the prediction text, we will also produce
            # the length for the progress bar visualization.
            meta = [
                (p, '%.5f' % scores[p]) for p in predictions
            ]
            logging.info('result: %s', str(meta))

            return True, meta, '%.3f' % (endtime - starttime), self.plots()

        except Exception as err:
            logging.info('Classification error: %s', err)
            return False, err.message

    def plots(self):
        return [
            # weights
            vis_square(self.net.params['conv1'][0].data.transpose(0, 2, 3, 1),
                       title='conv1 filters'),
            vis_square(self.net.params['conv2'][0].data[:20].reshape(20**2, 5, 5),
                       title='conv2 filters'),
            # activations
            vis_square(self.net.blobs['conv1'].data[0, :36], padval=1,
                       title='conv1'),
            vis_square(self.net.blobs['pool1'].data[0, :36], padval=1,
                       title='pool1'),
            vis_square(self.net.blobs['conv2'].data[0], padval=1,
                       title='conv2'),
            vis_square(self.net.blobs['pool2'].data[0], padval=0.5,
                       title='pool1'),
            vis_square(self.net.blobs['ip1'].data[0, :400].reshape(1, 20, 20), padval=0.5,
                       title='ip1'),
            vis_rec(self.net.blobs['ip2'].data[0].reshape(1, 10),
                    title='ip2')
        ]


def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    parser = optparse.OptionParser()
    parser.add_option(
        '-d', '--debug',
        help="enable debug mode",
        action="store_true", default=False)
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=5000)
    parser.add_option(
        '-g', '--gpu',
        help="use gpu mode",
        action='store_true', default=False)

    opts, args = parser.parse_args()
    LeNetClassifier.default_args.update({'gpu_mode': opts.gpu})

    # Initialize classifier + warm start by forward for allocation
    app.clf = LeNetClassifier(**LeNetClassifier.default_args)
    app.clf.net.forward()

    if opts.debug:
        app.run(debug=True, host='0.0.0.0', port=opts.port)
    else:
        start_tornado(app, opts.port)


# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, padsize=1, padval=0, title=None):
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

    plt.figure()
    plt.imshow(data)
    if title:
        plt.title(title)

    return title, mpld3.fig_to_html(plt.gcf())


def vis_rec(data, title=None):
    data = data.copy()
    data -= data.min()
    data /= data.max()

    plt.figure()
    plt.xticks(np.arange(0, 10, 1.0))
    plt.imshow(data)
    if title:
        plt.title(title)

    return title, mpld3.fig_to_html(plt.gcf())


def rgb2gray(rgb):
    return 1-np.dot(rgb[..., :3], [0.299, 0.587, 0.144])[..., None]


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    start_from_terminal(app)
