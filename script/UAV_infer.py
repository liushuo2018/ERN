import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc
import test_performance as tp
import os
import sys
import time
import scipy.io as io

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="indicate the model path")
parser.add_argument("-w", "--weight", help="indicate if it is training from a existing weight")
parser.add_argument("-i", "--idgpu", help="the id of the GPU used to infer")
parser.add_argument('--name', help="the model name")
parser.add_argument('--overlap', help="bool, overlap or not", type=bool) #true: overlap | false: no-overlap
args = parser.parse_args()

import ConfigParser
Config = ConfigParser.ConfigParser()
# Configure here to set the data config files

Config.read("setting_data_train.ini")
caffe_root = Config.get('Caffe', 'CaffeRoot')

sys.path.insert(0, caffe_root + 'python')

import caffe

c_proto = args.model
c_model = args.weight
# load net
caffe.set_mode_gpu()
caffe.set_device(int(args.idgpu))
net = caffe.Net(c_proto,
                c_model,
                caffe.TEST)


# ------------------------------------------------------- #
# Modify the parameters and data path
# ------------------------------------------------------- #
# The mean
m_mu_rgb = np.array([73.83527, 74.98086, 108.47272])

# The data source folder
if args.overlap:
    data_root = '/media/lh/D/Data/Part1_split_test_overlap_50/'
else:
    data_root = '/media/lh/D/Data/Part1_split_test/'


rgb_root = data_root + 'rgb/'
edge_root = data_root + 'label_edge/'
tag_root = data_root + 'tag_png/'
tag_infer_root = data_root + 'tag_infer_' + args.name + '/'
edge_infer_root = data_root + 'edge_infer_' + args.name + '/'


# Define the number of labels
nlabels = 6
labels = [0, 1, 2, 3, 4, 5]

num_edge_labels = 2
edge_labels = [0, 1]

# flag to save the infer result or not | 0:not 1:save
save_image = 0
if save_image ==1 and not os.path.exists(tag_infer_root):
    os.mkdir(tag_infer_root)

# flag to save the infer edge result or not | 0:not 1:save
save_edge = 0
if save_edge == 1 and not os.path.exists(edge_infer_root): 
    os.mkdir(edge_infer_root)


mat_confusion_sum = np.zeros([nlabels, nlabels])

F_score_sum = np.zeros((6, 1))
acc_score_sum = 0
prec_score_sum = np.zeros((6, 1))
cnt = np.zeros((6, 1))
print 'Network initialised...'

num_files = len([f for f in os.listdir(rgb_root)
                if os.path.isfile(os.path.join(rgb_root, f))])
print 'infering total: ' + str(num_files) + ' patches / images...'

cnt = 0
sum_time = 0

for root, dirs, files in os.walk(rgb_root):
    for onefile in files:
        fname = rgb_root + onefile
        if onefile.startswith('UAV_'):
            im = caffe.io.load_image(fname)
            im_rgb = im
            start_time = time.time()
            # read the correspond semantic tag & edge tag
            im_tag = Image.open(tag_root + onefile[0:-4] + '.png')
            # edge_tag = Image.open(edge_root + onefile[0:-4] + '.png')
                        
            # transform the input image
            transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
            transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
            transformer.set_mean('data', m_mu_rgb)  # subtract the dataset-mean value in each channel
            transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
            transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR
            transformed_image = transformer.preprocess('data', im)
            in_ = transformed_image
            # shape for input (data blob is N x C x H x W), set data
            net.blobs['data'].reshape(1, *in_.shape)
            net.blobs['data'].data[...] = in_
            
            net.forward()
            end_time = time.time()
            sum_time += (end_time-start_time)
            cnt += 1

            out = net.blobs['score'].data[0].argmax(axis=0)

            model_name = args.name
            if model_name.find('ERN_E') == -1:
                edge_score = 'edge_deconv_score'
            else:
                edge_score = 'edge_score'
                
            
            
            
            if save_image == 1:

                # save the class as png image
                save_fname = tag_infer_root + onefile[0:-4] + '.png'
                scipy.misc.toimage(out, cmin=0, cmax=255).save(save_fname)
                # save the score in the same location same name as npy
                save_fname = tag_infer_root + onefile[0:-4] + '.npy'
                d_score = net.blobs['score'].data[0]
                np.save(save_fname, d_score)
                #io.savemat(save_fname, {'score': d_score})
                
            if save_edge == 1:
                edge_out = net.blobs[edge_score].data[0].argmax(axis=0)
                # save the class as png image
                save_fname = edge_infer_root + onefile[0:-4] + '.png'
                scipy.misc.toimage(edge_out, cmin=0, cmax=255).save(save_fname)
                # save the score in the same location same name as npy
                save_fname = edge_infer_root + onefile[0:-4] + '.npy'
                d_score = net.blobs[edge_score].data[0]
                np.save(save_fname, d_score)
                
                                

            out = np.array(out)
            tag_arr = np.array(im_tag)

            mat_confusion = tp.x_make_confusion_mat(tag_arr, out, labels, 255)
            mat_confusion_sum = mat_confusion + mat_confusion_sum
            # should not calculate in this way...

mat_prec = tp.x_calc_prec(mat_confusion_sum, labels)
mat_recall = tp.x_calc_recall(mat_confusion_sum, labels)
mat_fscore = tp.x_calc_f1score(mat_prec, mat_recall, labels)

s_prec = tp.x_calc_over_prec(mat_confusion_sum, labels)
s_recall = tp.x_calc_over_recall(mat_confusion_sum, labels)
s_fscore = tp.x_calc_over_fscore(s_prec, s_recall)

s_acc = tp.x_calc_over_acc(mat_confusion_sum, labels)

print "F1_score = " + str(mat_fscore)
print "overall accuracy = " + str(s_acc)
print "precision = " + str(mat_prec)
print "Confusion Matrix:"
print mat_confusion_sum
print "average time consuming per patch:"
print sum_time/cnt
print "total time for all test images:"
print sum_time
