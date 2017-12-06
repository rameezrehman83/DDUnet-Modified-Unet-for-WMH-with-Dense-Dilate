'''
goal: To generate the *npy files for u-net to train and test
'''



# In[]:

# ## 1 Test Part

# ### 1.1 Differnet sizes of images among hospitals
# * GE3T/100 : 
# ```
#     shape of GE3T/100/pre/FLAIR.nii.gz : (132, 256, 83)
#     shape of GE3T/100/pre/T1.nii.gz : (132, 256, 83)
#     shape of GE3T/100/wmh.nii.gz : (132, 256, 83)
#     shape of GE3T/100/pre/3DT1.nii.gz : (256, 256, 176)
# ```
# * Singapore/50:
# ```
#     shape of Singapore/50/pre/FLAIR.nii.gz : (256, 232, 48)
#     shape of Singapore/50/pre/T1.nii.gz : (256, 232, 48)
#     shape of Singapore/50/wmh.nii.gz : (256, 232, 48)
#     shape of Singapore/50/pre/3DT1.nii.gz : (256, 256, 192)
# ```
# * Utrecht/31:
# ```
#     shape of Utrecht/31/pre/FLAIR.nii.gz : (240, 240, 48)
#     shape of Utrecht/31/pre/T1.nii.gz : (240, 240, 48)
#     shape of Utrecht/31/wmh.nii.gz : (240, 240, 48)
#     shape of Utrecht/31/pre/3DT1.nii.gz : (256, 256, 192)
# ```

# ### 1.2 Check the shapes of all samples. Walk all directories.
# Figure out if samples' shape are same in each hospital.
# - enumerate all patients's image shape
#     we got all GE3T (132, 256, 83); Singapore (232, 256, 48), (256, 232, 48); Utrecht (240, 240, 48)


# ### 1.3 Construct a 4-d array to store all samples of one hospital for future corping or padding
# * we do GE3T first
# * maybe we should do the corp or padding first

# In[]:

import os
import numpy as np
import nibabel as nib

# In[5]:

# 不同医院的数据大小不一样，所以可能要存储成为三套分别的数据集, e.g.

# (X_train_GE3T.npy, Y_train_GE3T.npy, X_test_GE3T.npy, Y_test_GE3T.npy)
# (X_train_Singapore.npy, Y_train_Singapore.npy, X_test_Singapore.npy, Y_test_Singapore.npy)
# (X_train_Utrecht.npy, Y_train_Utrecht.npy, X_test_Utrecht.npy, Y_test_Utrecht.npy)

def create_raw_data(hospitals = ["GE3T", "Singapore", "Utrecht"],\
                test_frac = 0.2, \
                cut_margin = 1/8, \
                which_part_as_test = 0 \
                ):
    print("starting create_data for hospitals of ", hospitals, "in ./data")
    
    #create the directories for preceeded data storing
    if not os.path.exists(os.path.join(os.getcwd(), "data_raw")):
        os.makedirs(os.path.join(os.getcwd(), "data_raw"))

        
    data_root = os.path.join(os.getcwd(), "data")
    for hos in hospitals:
        
        hos_root = os.path.join(data_root, hos)
        # (X_train_GE3T.npy, Y_train_GE3T.npy, X_test_GE3T.npy, Y_test_GE3T.npy)
        
        X_each_hos_t1 = None
        X_each_hos_flair = None
        Y_each_hos = None
        
        for dd in os.listdir(hos_root):
            if not dd.startswith('.'):
                patient_t1 = nib.load(os.path.join(hos_root, dd, "pre", "T1.nii.gz")).get_data()
                patient_flair = nib.load(os.path.join(hos_root, dd, "pre", "FLAIR.nii.gz")).get_data()
                patient_label = nib.load(os.path.join(hos_root, dd, "wmh.nii.gz")).get_data()
                
                #1, cut margin
                nz = patient_t1.shape[2]
                beg = int(nz * cut_margin)
                end = int(nz - nz*cut_margin)
                
                patient_t1 = patient_t1[:,:,beg:end]
                patient_flair = patient_flair[:,:,beg:end]
                patient_label = patient_label[:,:,beg:end]
                
                # move the last axis to the first axis
                patient_t1 = np.moveaxis(patient_t1, -1, 0)
                patient_flair = np.moveaxis(patient_flair, -1, 0)
                patient_label = np.moveaxis(patient_label, -1, 0)
                
                if X_each_hos_t1 is None:
                    X_each_hos_t1 = patient_t1
                    X_each_hos_flair = patient_flair
                    Y_each_hos = patient_label
                else:
                    if patient_t1.shape[1:] == X_each_hos_t1.shape[1:]:
                        X_each_hos_t1 = np.concatenate((X_each_hos_t1, patient_t1), axis=0)
                        X_each_hos_flair = np.concatenate((X_each_hos_flair, patient_flair), axis=0)
                        Y_each_hos = np.concatenate((Y_each_hos, patient_label), axis=0)
                    
        # construct 2-channel form training set
        tmp = np.concatenate((X_each_hos_t1.reshape((X_each_hos_t1.shape[0]*X_each_hos_t1.shape[1]*X_each_hos_t1.shape[2],1)), \
                              X_each_hos_flair.reshape((X_each_hos_flair.shape[0]*X_each_hos_flair.shape[1]*X_each_hos_flair.shape[2],1))), \
                             axis=1)
        X_each_hos = tmp.reshape((X_each_hos_t1.shape[0],X_each_hos_t1.shape[1],X_each_hos_t1.shape[2],2))
        # delete other labels except 1
        Y_each_hos = np.where(Y_each_hos == 1, 1, 0)
        #2, split for test
        
        # choose 'test_frac' for test, and '1-test_frac' for train, without replace randomly
        # ndarray
        
#        mask_train = np.random.choice(X_each_hos.shape[0], int(X_each_hos.shape[0]* (1- test_frac)), replace = False)
#        # list
#        mask_test = [x for x in list(range(X_each_hos.shape[0])) if x not in mask_train]
#        
#        X_train = X_each_hos[mask_train,:,:,:]
#        Y_train = Y_each_hos[mask_train,:,:]
#        X_test = X_each_hos[mask_test,:,:,:]
#        Y_test = Y_each_hos[mask_test,:,:]
        
        mask_test = np.arange(which_part_as_test * int(X_each_hos.shape[0]*test_frac), \
                              (which_part_as_test+1) * int(X_each_hos.shape[0]*test_frac))
        
        mask_train = [x for x in list(range(X_each_hos.shape[0])) if x not in mask_test]
        
        X_train = X_each_hos[mask_train,:,:,:]
        Y_train = Y_each_hos[mask_train,:,:]
        X_test = X_each_hos[mask_test,:,:,:]
        Y_test = Y_each_hos[mask_test,:,:]
        
        np.save(os.path.join("data_raw", "X_train_"+hos+".npy"), X_train)
        np.save(os.path.join("data_raw", "Y_train_"+hos+".npy"), Y_train)
        np.save(os.path.join("data_raw", "X_test_"+hos+".npy"), X_test)
        np.save(os.path.join("data_raw", "Y_test_"+hos+".npy"), Y_test)
        
        print("X_train_"+hos+".npy "+"created: ", X_train.shape)
        print("Y_train_"+hos+".npy "+"created: ", Y_train.shape)
        print("X_test_"+hos+".npy "+"created: ", X_test.shape)
        print("Y_test_"+hos+".npy "+"created: ", Y_test.shape)

    print("create_data finished!")

# In[]:
#def get_crop_shape(target, refer):
#    # height, the 1 dimension
#
#    ch = (K.get_variable_shape(target)[1] - K.get_variable_shape(refer)[1])
#    assert (ch >= 0)
#    if ch % 2 != 0:
#        ch1, ch2 = int(ch/2), int(ch/2) + 1
#    else:
#        ch1, ch2 = int(ch/2), int(ch/2)
#        
#    cw = (K.get_variable_shape(target)[2] - K.get_variable_shape(refer)[2])
#    assert (cw >= 0)
#    if cw % 2 != 0:
#        cw1, cw2 = int(cw/2), int(cw/2) + 1
#    else:
#        cw1, cw2 = int(cw/2), int(cw/2)
#    
#    return (ch1, ch2), (cw1, cw2)
    
def resize_data(new_size = 240, hospitals = ["GE3T", "Singapore", "Utrecht"]):
    
    # mkdir
    if not os.path.exists(os.path.join(os.getcwd(), "data_"+str(new_size))):
        os.makedirs(os.path.join(os.getcwd(), "data_"+str(new_size)))

    X_train = None
    Y_train = None
    X_test = None
    Y_test = None
    
    '''
    X_train
    '''
    for hos in hospitals:
        X_train_raw = np.load(os.path.join("data_raw", "X_train_"+hos+".npy"))
        
        ############## height ###################
        ch = X_train_raw.shape[1] - new_size
        if ch <= 0: # need padded to new_size on height
            if -ch % 2 != 0:
                ch1, ch2 = int(-ch/2), int(-ch/2) + 1
            else:
                ch1, ch2 = int(-ch/2), int(-ch/2)
            X_train_raw = np.pad(X_train_raw, [(0,0), (ch1, ch2), (0, 0), (0,0)], mode='constant', constant_values=0)
        else:
            if ch % 2 != 0:
                ch1, ch2 = int(ch/2), int(ch/2) + 1
            else:
                ch1, ch2 = int(ch/2), int(ch/2)
            
            s = X_train_raw.shape[1]
            X_train_raw = X_train_raw[:,ch1:s-ch2,:,:]
        
        ############## weight ###################
        cw = X_train_raw.shape[2] - new_size
        if cw <= 0:
            if -cw % 2 != 0:
                cw1, cw2 = int(-cw/2), int(-cw/2)+1
            else:
                cw1, cw2 = int(-cw/2), int(-cw/2)
            X_train_raw = np.pad(X_train_raw, [(0,0), (0, 0), (cw1, cw2), (0,0)], mode='constant', constant_values=0)
        else:
            if cw % 2 != 0:
                cw1, cw2 = int(cw/2), int(cw/2) + 1
            else:
                cw1, cw2 = int(cw/2), int(cw/2)
            s = X_train_raw.shape[2]
            X_train_raw = X_train_raw[:,:,cw1:s-cw2,:]
            
        if X_train is None:
            X_train = X_train_raw
        else:
            X_train = np.concatenate((X_train, X_train_raw), axis = 0)
    np.save(os.path.join("data_"+str(new_size), "X_train.npy"), X_train)
    print("X_train saved! ",X_train.shape)
        
    '''
    Y_train
    '''
    for hos in hospitals:
        Y_train_raw = np.load(os.path.join("data_raw", "Y_train_"+hos+".npy"))
        
        ############## height ###################
        ch = Y_train_raw.shape[1] - new_size
        if ch <= 0: # need padded to new_size on height
            if -ch % 2 != 0:
                ch1, ch2 = int(-ch/2), int(-ch/2) + 1
            else:
                ch1, ch2 = int(-ch/2), int(-ch/2)
            Y_train_raw = np.pad(Y_train_raw, [(0,0), (ch1, ch2), (0, 0)], mode='constant', constant_values=0)
        else:
            if ch % 2 != 0:
                ch1, ch2 = int(ch/2), int(ch/2) + 1
            else:
                ch1, ch2 = int(ch/2), int(ch/2)
            
            s = Y_train_raw.shape[1]
            Y_train_raw = Y_train_raw[:,ch1:s-ch2,:]
        
        ############## weight ###################
        cw = Y_train_raw.shape[2] - new_size
        if cw <= 0:
            if -cw % 2 != 0:
                cw1, cw2 = int(-cw/2), int(-cw/2)+1
            else:
                cw1, cw2 = int(-cw/2), int(-cw/2)
            Y_train_raw = np.pad(Y_train_raw, [(0,0), (0, 0), (cw1, cw2)], mode='constant', constant_values=0)
        else:
            if cw % 2 != 0:
                cw1, cw2 = int(cw/2), int(cw/2) + 1
            else:
                cw1, cw2 = int(cw/2), int(cw/2)
            s = Y_train_raw.shape[2]
            Y_train_raw = Y_train_raw[:,:,cw1:s-cw2]
            
        if Y_train is None:
            Y_train = Y_train_raw
        else:
            Y_train = np.concatenate((Y_train, Y_train_raw), axis = 0)
    np.save(os.path.join("data_"+str(new_size), "Y_train.npy"), Y_train)
    print("Y_train saved! ",Y_train.shape)
    
    '''
    X_test
    '''
    for hos in hospitals:
        X_test_raw = np.load(os.path.join("data_raw", "X_test_"+hos+".npy"))
        
        ############## height ###################
        ch = X_test_raw.shape[1] - new_size
        if ch <= 0: # need padded to new_size on height
            if -ch % 2 != 0:
                ch1, ch2 = int(-ch/2), int(-ch/2) + 1
            else:
                ch1, ch2 = int(-ch/2), int(-ch/2)
            X_test_raw = np.pad(X_test_raw, [(0,0), (ch1, ch2), (0, 0), (0,0)], mode='constant', constant_values=0)
        else:
            if ch % 2 != 0:
                ch1, ch2 = int(ch/2), int(ch/2) + 1
            else:
                ch1, ch2 = int(ch/2), int(ch/2)
            
            s = X_test_raw.shape[1]
            X_test_raw = X_test_raw[:,ch1:s-ch2,:,:]
        
        ############## weight ###################
        cw = X_test_raw.shape[2] - new_size
        if cw <= 0:
            if -cw % 2 != 0:
                cw1, cw2 = int(-cw/2), int(-cw/2)+1
            else:
                cw1, cw2 = int(-cw/2), int(-cw/2)
            X_test_raw = np.pad(X_test_raw, [(0,0), (0, 0), (cw1, cw2), (0,0)], mode='constant', constant_values=0)
        else:
            if cw % 2 != 0:
                cw1, cw2 = int(cw/2), int(cw/2) + 1
            else:
                cw1, cw2 = int(cw/2), int(cw/2)
            s = X_test_raw.shape[2]
            X_test_raw = X_test_raw[:,:,cw1:s-cw2,:]
            
        if X_test is None:
            X_test = X_test_raw
        else:
            X_test = np.concatenate((X_test, X_test_raw), axis = 0)
    np.save(os.path.join("data_"+str(new_size), "X_test.npy"), X_test)
    print("X_test saved! ",X_test.shape)
        
    '''
    Y_test
    '''
    for hos in hospitals:
        Y_test_raw = np.load(os.path.join("data_raw", "Y_test_"+hos+".npy"))
        
        ############## height ###################
        ch = Y_test_raw.shape[1] - new_size
        if ch <= 0: # need padded to new_size on height
            if -ch % 2 != 0:
                ch1, ch2 = int(-ch/2), int(-ch/2) + 1
            else:
                ch1, ch2 = int(-ch/2), int(-ch/2)
            Y_test_raw = np.pad(Y_test_raw, [(0,0), (ch1, ch2), (0, 0)], mode='constant', constant_values=0)
        else:
            if ch % 2 != 0:
                ch1, ch2 = int(ch/2), int(ch/2) + 1
            else:
                ch1, ch2 = int(ch/2), int(ch/2)
            
            s = Y_test_raw.shape[1]
            Y_test_raw = Y_test_raw[:,ch1:s-ch2,:]
        
        ############## weight ###################
        cw = Y_test_raw.shape[2] - new_size
        if cw <= 0:
            if -cw % 2 != 0:
                cw1, cw2 = int(-cw/2), int(-cw/2)+1
            else:
                cw1, cw2 = int(-cw/2), int(-cw/2)
            Y_test_raw = np.pad(Y_test_raw, [(0,0), (0, 0), (cw1, cw2)], mode='constant', constant_values=0)
        else:
            if cw % 2 != 0:
                cw1, cw2 = int(cw/2), int(cw/2) + 1
            else:
                cw1, cw2 = int(cw/2), int(cw/2)
            s = Y_test_raw.shape[2]
            Y_test_raw = Y_test_raw[:,:,cw1:s-cw2]
            
        if Y_test is None:
            Y_test = Y_test_raw
        else:
            Y_test = np.concatenate((Y_test, Y_test_raw), axis = 0)
    np.save(os.path.join("data_"+str(new_size), "Y_test.npy"), Y_test)
    print("Y_test saved! ",Y_test.shape)
    
    print("Resizing to", new_size, "finished!")
    
# In[]:
def load_train_data(size=240):
#    expected the hospital name to decide which file to read
    X = np.load(os.path.join("data_"+str(size), "X_train.npy"))
    Y = np.load(os.path.join("data_"+str(size), "Y_train.npy"))
    return X, Y

def load_test_data(size=240):
    X = np.load(os.path.join("data_"+str(size), "X_test.npy"))
    Y = np.load(os.path.join("data_"+str(size), "Y_test.npy"))
    return X, Y
    
# In[]:
def test():
    create_raw_data(test_frac=0.2, which_part_as_test=0)
    resize_data()
    X_train, Y_train = load_train_data()
    print("X_train: ", X_train.shape)
    print("Y_train: ", Y_train.shape)

if __name__ == '__main__':
    create_raw_data(test_frac=0.2, which_part_as_test=0) # for frac=0.2, which_part_as_test can take from {0,1,2,3,4}
    resize_data()
