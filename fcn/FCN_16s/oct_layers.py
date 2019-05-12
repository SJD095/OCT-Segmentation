import caffe

import numpy as np
from PIL import Image
import scipy.io

import random

class OCTDataLayer(caffe.Layer):

    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.oct_dir = params['oct_dir']
        self.split = params['split']
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)

        if len(top) != 2:
            raise Exception("Need to define three tops: data, semantic label, and geometric label.")

        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        split_f  = '{}/{}.txt'.format(self.oct_dir, self.split)
        self.indices = open(split_f, 'r').read().splitlines()
        self.idx = 0

        if 'train' not in self.split:
            self.random = False

        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)

    def reshape(self, bottom, top):
        self.data = self.load_image(self.indices[self.idx])
        self.Imagelabel = self.load_label(self.indices[self.idx])
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.Imagelabel.shape)

    def forward(self, bottom, top):
        top[0].data[...] = self.data
        top[1].data[...] = self.Imagelabel
        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0

    def backward(self, top, propagate_down, bottom):
        pass

    def load_image(self, idx):
        im = Image.open('{}/originalImages/{}'.format(self.oct_dir, idx))
        in_ = np.array(im, dtype=np.float32)
        in_ -= in_.mean()
        in_ = in_.reshape(1, in_.shape[0], in_.shape[1])
        return in_

    def load_label(self, idx):
        label = scipy.io.loadmat('{}/Labels/{}'.format(self.oct_dir, idx))['L']
        label = label.astype(np.uint8)
        label -= 1
        label = label[np.newaxis, ...]
        return label.copy()
