import caffe
import surgery, score

import numpy as np
import os
import sys

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass

weights = 'models/OCT_Segmentation.caffemodel'

vgg_weights = '../ilsvrc-nets/vgg16-fcn.caffemodel'  
vgg_proto = '../ilsvrc-nets/VGG_ILSVRC_16_layers_deploy.prototxt'  

fcn_weights = 'models/siftflow-fcn16s-heavy.caffemodel'  
fcn_proto = 'train.prototxt'  

# init
caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)

#vgg_net=caffe.Net(vgg_proto,vgg_weights,caffe.TRAIN)  
#surgery.transplant(solver.net,vgg_net)  
#del vgg_net

#fcn_net=caffe.Net(fcn_proto,fcn_weights,caffe.TRAIN)  
#surgery.transplant(solver.net,fcn_net)  
#del fcn_net

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
#test = np.loadtxt('../data/oct/test.txt', dtype=str)

for _ in range(20):
    solver.step(1000)
    # N.B. metrics on the semantic labels are off b.c. of missing classes;
    # score manually from the histogram instead for proper evaluation
    #score.seg_tests(solver, False, test, layer='score_sem', gt='sem')
