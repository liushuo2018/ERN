"""
ERN - assemble UAV patches

Liu Yu, Liu Shuo

This module serve to assemble the patches together for later visualisation and post-processing

The procedure is like:

"""

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import argparse
import sys
import scipy.io
import scipy.misc
import test_performance as tp
import os
import time


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name', help="string, input the model name")
parser.add_argument('--overlap', help="overlap or not", type=bool) #true: overlap | false: no-overlap 
args = parser.parse_args()


# ------------------------------------------------------- #
# Modify the parameters and data path
# ------------------------------------------------------- #
patch_size = [256, 256]
labels = [0, 1, 2, 3, 4, 5]
N_class = 6
test_set = [11, 15, 28, 30, 34]

# overlap rate
overlap = [0.5, 0.5]

# assemble flag of edge
assemble_edge_flag = 0 # flag to assemble the edge or not | 0:not 1:save

# raw root, for reading the raw image size (w,h)
raw_root = '/media/lh/D/VH_Data/raw_test/top/'

# the infered patches folder, the assemble results folder
if args.overlap:
    rgb_infer_root = '/media/lh/D/VH_Data/test_Overlap/' + 'tag_infer_' + args.name +  '/'
    edge_infer_root = '/media/lh/D/VH_Data/test_Overlap/' + 'edge_infer_' + args.name + '/'
    assemble_root = '/media/lh/D/VH_Data/assemble/' + args.name + '_OL' + '/'   
    assemble_edge_root = '/media/lh/D/VH_Data/assemble/' + args.name + '_OL' + '_edge' +'/'
else:
    rgb_infer_root = '/media/lh/D/VH_Data/test/' + 'tag_infer_' + args.name + '/'
    edge_infer_root = '/media/lh/D/VH_Data/test/' + 'edge_infer_' + args.name  + '/'
    assemble_root = '/media/lh/D/VH_Data/assemble/' + args.name + '/'   
    assemble_edge_root = '/media/lh/D/VH_Data/assemble/' + args.name + '_edge' + '/'

if not os.path.exists(assemble_root):
    os.mkdir(assemble_root)
if not os.path.exists(assemble_edge_root):
    os.mkdir(assemble_edge_root)
# ------------------------------------------------------- #

def xf_VH_assemble_one(ind):
    """
    assemble the ind'th image to the original size

    - assuming no overlap

    :param ind:
    :param data_root: the folder path of the tmp result, say the patches

    """
    # load original image to infer the size of the image
    fname = raw_root + 'top_mosaic_09cm_area' + str(ind) + '.tif'
    im = Image.open(fname)
    n_im_size = im.size
    n_w = n_im_size[0] / patch_size[0] + 1
    n_h = n_im_size[1] / patch_size[1] + 1
    im_re = np.zeros((n_h * patch_size[0], n_w * patch_size[1]))

    # "s_pfile_root" indicate the splited patches' name
    s_pfile_root = 'VH_' + str(ind).zfill(2) + '_'
    pre_id = 0
    # iterate the height
    for ih in range(0, n_h):
        for iw in range(0, n_w):
            # rgb_infer_root indicate where the patches is stored
            fname = rgb_infer_root + s_pfile_root + str(pre_id).zfill(4) + str(0) + '.png'
            pre_id += 1
            im_pre = Image.open(fname)
            im_pre_np = np.asarray(im_pre)
            il = patch_size[1] * iw
            iu = patch_size[0] * ih
            im_re[iu:iu + patch_size[0], il:il + patch_size[1]] = im_pre_np
    im_re_c = im_re[0:n_im_size[1], 0:n_im_size[0]]
    return im_re_c


def xf_VH_assemble():
    """ Assemble the inference result for VH Data

    - assemble without overlap
    
    """
    cnt = 5
    sum_time = 0
    for root, dirs, files in os.walk(raw_root):
        for onefile in files:
            start_time = time.time()
            fname = raw_root + onefile
            l1 = len(onefile)
            l2 = len('top_mosaic_09cm_area')
            # onefile is like "top_mosaic_09cm_areaXX.tif"
            img_id = onefile[l2:-4] # 2 digits number
            # print img_id
            # read the original image
            im = Image.open(fname)
            # calculate the size of image
            n_im_size = im.size
            n_w = n_im_size[0] / patch_size[0] + 1
            n_h = n_im_size[1] / patch_size[1] + 1
            # build the new empty image
            im_re = np.zeros((n_h * patch_size[0], n_w * patch_size[1]))

            # "s_pfile_root" indicate the splited patches' name
            s_pfile_root = 'VH_' + img_id
            pre_id = 0

            if assemble_edge_flag == 1:
                im_re_edge = np.zeros((n_h * patch_size[0], n_w * patch_size[1]))
                pre_id_edge = 0

            # iterate the height
            for ih in range(0, n_h):
                for iw in range(0, n_w):
                    # assemble the rgb
                    fname = rgb_infer_root + s_pfile_root + '_' + str(pre_id).zfill(4) + str(0) + '.png'
                    pre_id += 1
                    im_pre = Image.open(fname)
                    im_pre_np = np.asarray(im_pre)
                    il = patch_size[1] * iw
                    iu = patch_size[0] * ih
                    im_re[iu:iu + patch_size[0], il:il + patch_size[1]] = im_pre_np
                    # assemble the edge
                    if assemble_edge_flag == 1:
                        fname_edge = edge_infer_root + s_pfile_root + '_' + str(pre_id_edge).zfill(4) + str(0) + '.png'
                        pre_id_edge += 1
                        im_pre_edge = Image.open(fname_edge)
                        im_pre_np_edge = np.asarray(im_pre_edge)
                        il_edge = patch_size[1] * iw
                        iu_edge = patch_size[0] * ih
                        im_re_edge[iu_edge:iu_edge + patch_size[0], il_edge:il_edge + patch_size[1]] = im_pre_np_edge

            end_time = time.time()
            sum_time += (end_time-start_time)
            
            # crop
            im_re_c = im_re[0:n_im_size[1], 0:n_im_size[0]]
            # save
            rgb_save = assemble_root + s_pfile_root + '.png'
            scipy.misc.toimage(im_re_c, cmin=0, cmax=255).save(rgb_save)

            if assemble_edge_flag == 1:
                im_re_c_edge = im_re_edge[0:n_im_size[1], 0:n_im_size[0]]
                edge_save = assemble_edge_root + s_pfile_root + '.png'
                scipy.misc.toimage(im_re_c_edge, cmin=0, cmax=255).save(edge_save)
    
    print "average stitching time consuming per tile:"
    print sum_time/cnt
    print "total time for stitching all test images:"
    print sum_time


def xf_VH_assemble_OverLap():
    """
    This function serve to assemble the inference result with overlap, to carry out overlap inference.

    method: select most vote for the class

    """
    for root, dirs, files in os.walk(raw_root):
        for onefile in files:
            fname = raw_root + onefile
            l1 = len(onefile)
            l2 = len('top_mosaic_09cm_area')
            # onefile is like "top_mosaic_09cm_areaXX.tif"
            img_id = onefile[l2:-4] # 2 digits number
            #print img_id
            # read the original image
            im = Image.open(fname, 'r')
            # calculate the size of image
            n_im_size = im.size
            n_w = int(n_im_size[0]/patch_size[0]/(1-overlap[0]) + 1)
            #print n_w
            if int(img_id) == 30:
                n_w += 1
            #print n_w
            n_h = int(n_im_size[1]/patch_size[1]/(1-overlap[1]) + 1)
            #print n_h
            # build the new empty image
            re_w = int(patch_size[0] + n_w*patch_size[0]*(1-overlap[0]))
            re_h = int(patch_size[1] + n_h*patch_size[1]*(1-overlap[1]))
            im_re = np.zeros((N_class, re_h, re_w))
            np_re = np.zeros((N_class, re_h, re_w))
            np_re_c = np.zeros((N_class, re_h, re_w))

            # "s_pfile_root" indicate the splited patches' name
            s_pfile_root = 'VH_' + img_id
            pre_id = 0

            if assemble_edge_flag == 1:
                im_re_edge = np.zeros((n_h * patch_size[0], n_w * patch_size[1]))
                pre_id_edge = 0

            # iterate the height

            #print(im_re.shape)
            #print(fname)

            iu = 0
            for ih in range(0, n_h):
                il = 0
                np_il = 0
                for iw in range(0, n_w):  
                    # use the data_root to indicate where the data is stored
                    fname = rgb_infer_root + s_pfile_root + '_' + str(pre_id).zfill(4) + str(0) + '.png'
                    fname_npy = rgb_infer_root + s_pfile_root + '_' + str(pre_id).zfill(4) + str(0) + '.npy'

                    pre_id += 1
                    im_pre = Image.open(fname)
                    im_pre_np = np.asarray(im_pre)
                    np_pre = np.load(fname_npy)
                    # [most vote count here]
                    for iih in range(0, patch_size[0]):
                        for iiw in range(0, patch_size[1]):
                            itag = im_pre_np[iih, iiw]
                            #print([itag, iu+iih, il+iiw])
                            im_re[itag, iu+iih, il+iiw] += 1
                    # [np array assemble here]
                    np_re[:, iu:iu + patch_size[0], il:il + patch_size[1]] += np_pre
                    np_re_c[:, iu:iu + patch_size[0], il:il + patch_size[1]] += 1   # adding up the counter
                    il = int(il + patch_size[1] - overlap[1] * patch_size[1])
                iu = int(iu + patch_size[0] - overlap[0] * patch_size[0])
            
            
            np_re_sum = np_re / np_re_c
            np_re_sum_c = np_re_sum[:, 0:n_im_size[1], 0:n_im_size[0]]

            im_re_max = np.argmax(np_re_sum, axis=0)
            im_re_c = im_re_max[0:n_im_size[1], 0:n_im_size[0]]
            rgb_save = assemble_root + s_pfile_root + '.png'
            scipy.misc.toimage(im_re_c, cmin=0, cmax=255).save(rgb_save)

if __name__ == '__main__': 
    # without overlap
    if args.overlap:
        xf_VH_assemble_OverLap()
    else:
        xf_VH_assemble()


    
