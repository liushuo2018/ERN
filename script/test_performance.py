import numpy as np
from PIL import Image
import os
import sys


def x_make_confusion_mat(y_gt, y_pred, labels, lignore):
    '''
    making the confusion matrix using the ground truth and prediction, the labels should be indicated.

    :param y_gt: ground truth matrix
    :param y_pred: predicted label matrix
    :param labels:
    :param lignore: ignore labels
    :return:
    '''
    nlabels = len(labels)
    y_gt = y_gt.flatten()
    y_pred = y_pred.flatten()
    mat_confusion = np.zeros([nlabels, nlabels])

    for clabel in labels:
        inds = np.where(y_pred == clabel)
        inds = inds[0]
        # print len(inds)
        for ind in inds:
            ilabel = y_gt[ind]
            if ilabel != lignore:
                mat_confusion[ilabel, clabel] = mat_confusion[ilabel, clabel] + 1
            else:
                mat_confusion[clabel, clabel] += 0
    return mat_confusion


def x_calc_prec(mat_confusion, labels):
    mat_prec = np.zeros([mat_confusion.shape[1], 1])
    for label in labels:
        tp = mat_confusion[label, label]
        fp = sum(mat_confusion[:, label])
        if fp>0:
            mat_prec[label] = tp / fp
        else:
            mat_prec[label] = 0
    return mat_prec


def x_calc_recall(mat_confusion, labels):
    mat_recall = np.zeros([mat_confusion.shape[1], 1])
    for label in labels:
        tp = mat_confusion[label, label]
        fn = sum(mat_confusion[label, :])
        if fn > 0:
            mat_recall[label] = tp / fn
        else:
            mat_recall[label] = 0
    return mat_recall


def x_calc_f1score(mat_prec, mat_reall, labels):
    mat_fscore = np.zeros([len(mat_reall), 1])
    for label in labels:
        prec = mat_prec[label]
        recall = mat_reall[label]
        if prec + recall > 0:
            mat_fscore[label] = 2 * prec * recall / (prec + recall)
        else:
            mat_fscore[label] = 0
    return mat_fscore


def x_calc_over_prec(mat_confusion, labels):
    s_tp = 0
    s_fp = 0
    for label in labels:
        tp = mat_confusion[label, label]
        fp = sum(mat_confusion[:, label])
        s_tp += tp
        s_fp += fp
    return s_tp/s_fp


def x_calc_over_recall(mat_confusion, labels):
    s_tp = 0
    s_fn = 0
    for label in labels:
        tp = mat_confusion[label, label]
        fn = sum(mat_confusion[label, :])
        s_tp += tp
        s_fn += fn
    s_recall = s_tp/s_fn
    return s_recall


def x_calc_over_fscore(prec, recall):
    return (2 * prec * recall / (prec + recall))


def x_calc_over_acc(mat_confusion, labels):
    s_all = sum(sum(mat_confusion))
    s_tp = 0
    for label in labels:
        tp = mat_confusion[label, label]
        s_tp += tp
    return s_tp / s_all

def xf_set_test_UAV(modelname):
    label_root = '/media/lh/D/Data/Partion1/test/label_png/'
    pred_root = '/media/lh/D/Data/Partion1/UAV_results/'

    labels = [0, 1, 2, 3, 4, 5]
    nlabels = 6

    mat_confusion_s = np.zeros([nlabels, nlabels])

    for root, dirs, files in os.walk(label_root):
        for onefile in files:
    
        	gt_fname = label_root + onefile
        	pred_fname = pred_root + modelname + '/' + 'UAV_' + onefile

       		im_gt = Image.open(gt_fname)
        	arr_gt = np.array(im_gt)
        	im_pred = Image.open(pred_fname)
        	arr_pred = np.array(im_pred)
        	mat_confusion = x_make_confusion_mat(arr_gt, arr_pred, labels=labels, lignore=255)
        	mat_confusion_s += mat_confusion

    mat_prec = x_calc_prec(mat_confusion_s, labels)
    mat_recall = x_calc_recall(mat_confusion_s, labels)
    mat_fscore = x_calc_f1score(mat_prec, mat_recall, labels)
    s_acc = x_calc_over_acc(mat_confusion_s, labels)

    print "confusion matrix:"
    print mat_confusion_s
    print 'Fscore:'
    print mat_fscore
    print 'Accuracy:'
    print mat_prec
    print 'Overall accuracy:'
    print s_acc

def xf_set_test_vh(modelname):
    label_root = '/media/lh/D/VH_Data/raw_test/gts_png/'
    pred_root = '/media/lh/D/VH_Data/assemble/'    
    gt_prefix = 'top_mosaic_09cm_area'
    pred_prefix = 'VH_'

    labels = [0, 1, 2, 3, 4, 5]
    test_set = [11, 15, 28, 30, 34]
    nlabels = 6

    mat_confusion_s = np.zeros([nlabels, nlabels])
    for cind in test_set:
        gt_fname = label_root + gt_prefix + str(cind) + '.png'
        pred_fname = pred_root + modelname + '/'+ pred_prefix + str(cind) + '.png'

        im_gt = Image.open(gt_fname)
        arr_gt = np.array(im_gt)
        im_pred = Image.open(pred_fname)
        arr_pred = np.array(im_pred)
        mat_confusion = x_make_confusion_mat(arr_gt, arr_pred, labels=labels, lignore=255)
        mat_confusion_s += mat_confusion

    mat_prec = x_calc_prec(mat_confusion_s, labels)
    mat_recall = x_calc_recall(mat_confusion_s, labels)
    mat_fscore = x_calc_f1score(mat_prec, mat_recall, labels)
    s_acc = x_calc_over_acc(mat_confusion_s, labels)

    print mat_confusion_s
    print 'Fscore:'
    print mat_fscore
    print 'Accuracy:'
    print mat_prec
    print 'Overall accuracy:'
    print s_acc

def xf_set_vh_shadow(modelname):
    label_root = '/media/lh/D/VH_Data/raw_test/gts_png/'
    shadow_root = '/media/lh/D/VH_Data/raw_test/shadow_png/'
    pred_root = '/media/lh/D/VH_Data/assemble/'    
    gt_prefix = 'top_mosaic_09cm_area'
    pred_prefix = 'VH_'

    labels = [0, 1, 2, 3, 4, 5]
    test_set = [11, 15, 28, 30, 34]
    nlabels = 6

    mat_confusion_s = np.zeros([nlabels, nlabels])
    for cind in test_set:
        gt_fname = label_root + gt_prefix + str(cind) + '.png'
        #print gt_fname
        pred_fname = pred_root + modelname + '/'+ pred_prefix + str(cind) + '.png'
        #print pred_fname
        shadow_fname = shadow_root + gt_prefix + str(cind) + '.png'
        #print shadow_fname

        im_gt = Image.open(gt_fname)
        arr_gt = np.array(im_gt)

        im_shadow = Image.open(shadow_fname)
        arr_shadow = np.array(im_shadow)

        im_pred = Image.open(pred_fname)
        arr_pred = np.array(im_pred)

        arr_gt = arr_gt.flatten()
        arr_shadow = arr_shadow.flatten()
        arr_pred = arr_pred.flatten()
        mat_confusion = np.zeros([nlabels, nlabels])

        lignore = 255

        for clabel in labels:
            inds = np.where(arr_pred == clabel)
            inds = inds[0]
            print len(inds)
            for ind in inds:
                ilabel = arr_gt[ind]
                shadowlabel = arr_shadow[ind]
                if ilabel != lignore and shadowlabel == 255:
                    mat_confusion[ilabel, clabel] = mat_confusion[ilabel, clabel] + 1
                else:
                    mat_confusion[clabel, clabel] += 0

        mat_confusion_s += mat_confusion

    mat_prec = x_calc_prec(mat_confusion_s, labels)
    mat_recall = x_calc_recall(mat_confusion_s, labels)
    mat_fscore = x_calc_f1score(mat_prec, mat_recall, labels)
    s_acc = x_calc_over_acc(mat_confusion_s, labels)

    print "confusion matrix:"
    print mat_confusion_s
    print 'Fscore:'
    print mat_fscore
    print 'Accuracy:'
    print mat_prec
    print 'Overall accuracy:'
    print s_acc


def main():
    modelname = raw_input('Enter the model name:')
    ###############################
    #test the UAV Image Dataset
    ###############################
    #xf_set_test_UAV(modelname)

    ###############################
    #test the ISPRS VH Dataset
    ###############################
    xf_set_test_vh(modelname)
    
    #################################
    # shadow-affected -regions for VH
    #################################
    #xf_set_vh_shadow(modelname)

if __name__ == '__main__':
	main()
	print 'Done!'
	
