import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

import caffe

caffe.set_device(0)
caffe.set_mode_gpu()

testFile = open('../data/oct/test.txt')
lines = testFile.readlines()

# load net
net = caffe.Net('test.prototxt', 'models/OCT_Segmentation.caffemodel', caffe.TEST)

for i in range(len(lines)):
    imageName = lines[i].split('\n')[0]
    print(imageName)

    # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
    im = Image.open('../data/oct/originalImages/' + imageName)
    im = im.resize((256, 256))
    in_ = np.array(im, dtype=np.float32)
    in_ -= in_.mean()

    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, 1, *in_.shape)
    net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    net.forward()
    out = net.blobs['score_sem'].data[0].argmax(axis=0)

    [rows, cols] = out.shape;

    for countRow in range(rows - 1):
        for countCol in range(cols - 1):
            if out[countRow, countCol] == 1:
                out[countRow, countCol] = 255;

    outImage = Image.fromarray(np.uint8(out));
    outImage.save('results/' + imageName, 'JPEG');
