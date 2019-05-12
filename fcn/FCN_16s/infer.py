import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

import caffe

caffe.set_device(0)
caffe.set_mode_gpu()

np.set_printoptions(threshold=np.inf)
# the demo image is "2007_000129" from PASCAL VOC

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
im = Image.open('test.jpg')
in_ = np.array(im, dtype=np.float32)
in_ -= in_.mean()

# load net
net = caffe.Net('test.prototxt', 'models/OCT_Segmentation.caffemodel', caffe.TEST)
# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, 1, *in_.shape)
net.blobs['data'].data[...] = in_
# run net and take argmax for prediction
net.forward()

for i in range(0, 0):
    tmp_out = net.blobs['upscore16_sem'].data[0][i]
    plt.imshow(tmp_out, cmap='gray');
    plt.axis('off')
    plt.savefig('filters/' + str(i) + '.jpg')

out = net.blobs['score_sem'].data[0].argmax(axis=0)

[rows, cols] = out.shape;

for countRow in range(rows - 1):
	for countCol in range(cols - 1):
		if out[countRow, countCol] == 1:
			out[countRow, countCol] = 255;

outImage = Image.fromarray(np.uint8(out));
outImage.save('segmentation.jpg', 'JPEG');
