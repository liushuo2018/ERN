import os
import sys
import argparse
import ConfigParser
Config = ConfigParser.ConfigParser()
# Configure here to set the data config files
Config.read("setting_data_train.ini")
caffe_root = Config.get('Caffe', 'CaffeRoot')

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--solver", help="indicate the model path")
parser.add_argument("--weight", help="indicate if it is training from a existing weight")
parser.add_argument("-i", "--idgpu", help="indicate the id of the GPU wanted to use")
args = parser.parse_args()

print "solving from file:" + args.solver
if args.weight is not None:
    print "solving with pre-trained weight: " + args.weight
else:
    print "solving from begining"

sys.path.insert(0, caffe_root + 'python')
sys.path.append("/home/lh/anaconda2/lib/python2.7/site-packages")  
sys.path.append("/home/lh/anaconda2/lib/python2.7/")  

import caffe
from caffe import draw
from caffe.proto import caffe_pb2

caffe.set_device(int(args.idgpu))
caffe.set_mode_gpu()

solver = caffe.SGDSolver(args.solver)

weights = args.weight
if weights is not None:
    solver.net.copy_from(weights)
# init

for _ in range(200):
    solver.step(1000)
    #score.seg_tests(solver, False, val, layer='score')
