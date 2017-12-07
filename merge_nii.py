#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 21:22:45 2017

@author: zhenggangxue
"""

import os
import numpy as np
import nibabel as nib


#这里是先写死了 test_frac = 0.2, 所以会去找 0，1，2，3，4 五个文件。
#最后会在“nii_generated”文件夹中生成四个文件，分别是“X_t1.nii.gz”,"X_flair.nii.gz","Y_test.nii.gz","Y_pred.nii.gz"
def merge_nii(data_root="data_240"):
    print("Start merge_nii().....")
    
    #create the directories for preceeded data storing
    if not os.path.exists(os.path.join(os.getcwd(), "nii_generated")):
        os.makedirs(os.path.join(os.getcwd(), "nii_generated"))
    
    X_test_t1_0 = nib.load(os.path.join(data_root, "X_test_t1_0.nii.gz")).get_data()
    X_test_t1_1 = nib.load(os.path.join(data_root, "X_test_t1_1.nii.gz")).get_data()
    X_test_t1_2 = nib.load(os.path.join(data_root, "X_test_t1_2.nii.gz")).get_data()
    X_test_t1_3 = nib.load(os.path.join(data_root, "X_test_t1_3.nii.gz")).get_data()
    X_test_t1_4 = nib.load(os.path.join(data_root, "X_test_t1_4.nii.gz")).get_data()
    X_test_t1 = np.concatenate((X_test_t1_0, X_test_t1_1, X_test_t1_2, X_test_t1_3, X_test_t1_4), axis=0)    
    X_test_t1 = X_test_t1.astype(float)
    img = nib.Nifti1Image(X_test_t1, np.eye(4))
    nib.save(img, os.path.join("nii_generated","X_t1.nii.gz"))
    
    X_test_flair_0 = nib.load(os.path.join(data_root, "X_test_flair_0.nii.gz")).get_data()
    X_test_flair_1 = nib.load(os.path.join(data_root, "X_test_flair_1.nii.gz")).get_data()
    X_test_flair_2 = nib.load(os.path.join(data_root, "X_test_flair_2.nii.gz")).get_data()
    X_test_flair_3 = nib.load(os.path.join(data_root, "X_test_flair_3.nii.gz")).get_data()
    X_test_flair_4 = nib.load(os.path.join(data_root, "X_test_flair_4.nii.gz")).get_data()
    X_test_flair= np.concatenate((X_test_flair_0, X_test_flair_1, X_test_flair_2, X_test_flair_3, X_test_flair_4),axis=0)
    X_test_flair = X_test_flair.astype(float)
    img = nib.Nifti1Image(X_test_flair, np.eye(4))
    nib.save(img, os.path.join("nii_generated","X_flair.nii.gz"))
    
    Y_test_0 = nib.load(os.path.join(data_root, "Y_test_0.nii.gz")).get_data()
    Y_test_1 = nib.load(os.path.join(data_root, "Y_test_1.nii.gz")).get_data()
    Y_test_2 = nib.load(os.path.join(data_root, "Y_test_2.nii.gz")).get_data()
    Y_test_3 = nib.load(os.path.join(data_root, "Y_test_3.nii.gz")).get_data()
    Y_test_4 = nib.load(os.path.join(data_root, "Y_test_4.nii.gz")).get_data()
    Y_test= np.concatenate((Y_test_0,Y_test_1,Y_test_2,Y_test_3,Y_test_4 ),axis=0)
    Y_test = Y_test.astype(float)
    img = nib.Nifti1Image(Y_test, np.eye(4))
    nib.save(img, os.path.join("nii_generated","Y_test.nii.gz"))
    
    Y_pred_0 = nib.load(os.path.join(data_root, "Y_pred_0.nii.gz")).get_data()
    Y_pred_1 = nib.load(os.path.join(data_root, "Y_pred_1.nii.gz")).get_data()
    Y_pred_2 = nib.load(os.path.join(data_root, "Y_pred_2.nii.gz")).get_data()
    Y_pred_3 = nib.load(os.path.join(data_root, "Y_pred_3.nii.gz")).get_data()
    Y_pred_4 = nib.load(os.path.join(data_root, "Y_pred_4.nii.gz")).get_data()
    Y_pred= np.concatenate((Y_pred_0,Y_pred_1,Y_pred_2,Y_pred_3,Y_pred_4 ),axis=0)
    Y_pred = Y_pred.astype(float)
    img = nib.Nifti1Image(Y_pred, np.eye(4))
    nib.save(img, os.path.join("nii_generated","Y_pred.nii.gz"))
    
    print("Finish merge_nii()")
    
if __name__ == '__main__':
    merge_nii()
    
    