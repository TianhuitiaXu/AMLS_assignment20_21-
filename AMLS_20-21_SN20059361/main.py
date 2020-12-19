from B2 import CNN_B2 as B2
from B1 import CNN_B1 as B1
from A1 import CNN_A1 as A1
from A2 import landmarks_A2 as la
from A2 import SVM_A2 as svma2
import os
import numpy as np 
import pandas as pd 
from tqdm import tqdm
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
import tensorflow as tf
import time
# # ======================================================================================================================
# # Data preprocessing
# now_time = time.time()
global basedir, image_paths, target_size, data_B, image_B
data_basedir = './Datasets'
basedir_B = os.path.join(data_basedir,'cartoon_set')
images_dir_B = os.path.join(basedir_B,'img')
basedir_A = os.path.join(data_basedir,'celeba')
images_dir_A = os.path.join(basedir_A,'img')
labels_filename = 'labels.csv'

data_B = pd.read_csv(os.path.join(basedir_B, labels_filename),sep='\t')
data_A = pd.read_csv(os.path.join(basedir_A, labels_filename),sep='\t')
imageA_paths = [os.path.join(images_dir_A, l) for l in os.listdir(images_dir_A)]
# #read image
image_B = [] 
for i in tqdm(range(data_B.shape[0])):
    img = image.load_img(os.path.join(images_dir_B, str(i))+'.png', target_size=(100,100), color_mode='rgb')
    img = image.img_to_array(img)        
    image_B.append(img) 
image_B = np.array(image_B)
image_A = []
for i in tqdm(range(data_A.shape[0])):
    img = image.load_img(os.path.join(images_dir_A, str(i))+'.jpg', target_size=(109,89), color_mode='grayscale')
    img = image.img_to_array(img)        
    image_A.append(img)

##mean_nor
def mean_nor(x):
    x = (x - np.mean(x)) / np.std(x)
    return x, np.mean(x), np.std(x)

#min_max nor
def minmax_nor(x):
    diff = np.max(x)-np.min(x)
    x = (x - np.min(x)) / diff
    return x, np.min(x), diff

##central_crop
def center_crop(image,size):
    cropped_image = tf.image.central_crop(
      image, size)
    cropped_image = np.array(cropped_image)
    return cropped_image

def dataA1_preprocessing(data_A, image_A):
    # Y_A2=data_B['smiling']
    Y_A1=data_A['gender']
    Y_A1 = (np.array(Y_A1, dtype=float) + 1)/2
    image_A = center_crop(image_A,0.7)
    tr_x_a1, te_x_a1, tr_y_a1, te_y_a1 = train_test_split(image_A, Y_A1, train_size=0.8, random_state=0)
    tr_x_a1, cv_x_a1, tr_y_a1, cv_y_a1 = train_test_split(tr_x_a1, tr_y_a1, train_size=0.8, random_state=0)
    tr_x_a1, min, diff = minmax_nor(tr_x_a1)
    cv_x_a1 = (cv_x_a1 - min) / diff
    te_x_a1 = (te_x_a1 - min) / diff
    return tr_x_a1, cv_x_a1, te_x_a1, tr_y_a1, cv_y_a1, te_y_a1
tr_x_a1, cv_x_a1, te_x_a1, tr_y_a1, cv_y_a1, te_y_a1 = dataA1_preprocessing(data_A, image_A)

def dataA2_preprocessing(basedir_A,labels_filename,images_dir_A,imageA_paths):
    image_A2, Y_A2 = la.extract_features_labels(basedir_A,labels_filename,images_dir_A,imageA_paths)
    Y = np.array([Y_A2, -(Y_A2 - 1)]).T
    # Shuffle and split the data into training and test set
    #X, Y = shuffle(X,Y)
    x_train, x_test, y_train, y_test = train_test_split(image_A2, Y, train_size=0.8, random_state=0)
    x_train, x_cv, y_train, y_cv = train_test_split(x_train, y_train, train_size=0.8, random_state=0)
    return x_train, y_train, x_cv, y_cv, x_test, y_test
tr_x_a2, tr_y_a2, cv_x_a2, cv_y_a2, te_x_a2, te_y_a2 = dataA2_preprocessing(basedir_A,labels_filename,images_dir_A,imageA_paths)

def dataB_preprocessing(data_B, image_B):
    #obtain labels
    # data = pd.read_csv(os.path.join(basedir, labels_filename),sep='\t')
    Y_B2=data_B['eye_color']
    Y_B1=data_B['face_shape']

    image_B = center_crop(image_B,0.5)

    tr_x_b1, te_x_b1, tr_y_b1, te_y_b1 = train_test_split(image_B, Y_B1, train_size=0.8, random_state=0)
    tr_x_b1, cv_x_b1, tr_y_b1, cv_y_b1 = train_test_split(tr_x_b1, tr_y_b1, train_size=0.8, random_state=0)

    tr_x_b2, te_x_b2, tr_y_b2, te_y_b2 = train_test_split(image_B, Y_B2, train_size=0.8, random_state=0)
    tr_x_b2, cv_x_b2, tr_y_b2, cv_y_b2 = train_test_split(tr_x_b2, tr_y_b2, train_size=0.8, random_state=0)

    tr_x_b1, min, diff = minmax_nor(tr_x_b1)
    cv_x_b1 = (cv_x_b1 - min) / diff
    te_x_b1 = (te_x_b1 - min) / diff

    tr_x_b2, min_, diff_ = minmax_nor(tr_x_b2)
    cv_x_b2 = (cv_x_b2 - min_) / diff_
    te_x_b2 = (te_x_b2 - min_) / diff_

    datab1b2 = [tr_x_b1, cv_x_b1, te_x_b1, tr_y_b1, cv_y_b1, te_y_b1, tr_x_b2, cv_x_b2, te_x_b2, tr_y_b2, cv_y_b2, te_y_b2]

    return datab1b2 
datab1b2 = dataB_preprocessing(data_B, image_B)

# #read additional test data
# data_basedir_ = './dataset_AMLS_20-21_test'
# basedir_B_ = os.path.join(data_basedir_,'cartoon_set_test')
# images_dir_B_ = os.path.join(basedir_B_,'img')
# basedir_A_ = os.path.join(data_basedir_,'celeba_test')
# images_dir_A_ = os.path.join(basedir_A_,'img')
# labels_filename_ = 'labels.csv'
# data_B_ = pd.read_csv(os.path.join(basedir_B_, labels_filename),sep='\t')
# data_A_ = pd.read_csv(os.path.join(basedir_A_, labels_filename),sep='\t')
# imageA_paths_ = [os.path.join(images_dir_A_, l) for l in os.listdir(images_dir_A_)]
# # 
# image_B_ = [] 
# for i in tqdm(range(data_B_.shape[0])):
#     img = image.load_img(os.path.join(images_dir_B_, str(i))+'.png', target_size=(100,100), color_mode='rgb')
#     img = image.img_to_array(img)        
#     image_B_.append(img) 
# image_B_ = np.array(image_B_)
# image_A_ = []
# for i in tqdm(range(data_A_.shape[0])):
#     img = image.load_img(os.path.join(images_dir_A_, str(i))+'.jpg', target_size=(109,89), color_mode='grayscale')
#     img = image.img_to_array(img)        
#     image_A_.append(img)
# image_A_ = np.array(image_A_)
# y_a1 = data_A_['gender']
# y_a1 = (np.array(y_a1, dtype=float) + 1)/2
# y_a2 = data_A_['smiling']
# y_b1 = data_B_['face_shape']
# y_b2 = data_B_['eye_color']
# #A1
# image_A1_ = center_crop(image_A_,0.7)
# tr_x_a1, min_a1, diff = minmax_nor(tr_x_a1)
# image_A1_ = (image_A1_ - min_a1) / diff
# #A2
# image_A2_, y_a2 = la.extract_features_labels(basedir_A_,labels_filename_,images_dir_A_,imageA_paths_)
# y_a2 = np.array([y_a2, -(y_a2 - 1)]).T
# #B
# image_B_ = center_crop(image_B_,0.5)
# tr_x_b1, min_b1, diff = minmax_nor(datab1b2[0])
# image_B1_ = (image_B_ - min_b1) / diff
# tr_x_b2, min_, diff_ = minmax_nor(datab1b2[6])
# image_B2_ = (image_B_ - min_) / diff_
# data_train, data_val, data_test = data_preprocessing(args...)
# # ======================================================================================================================
# # Task A1
loss, acc_A1_train, val_loss, acc_A1_val, test_loss, acc_A1_test = A1.CNN_A1(tr_x_a1, cv_x_a1, te_x_a1, tr_y_a1, cv_y_a1, te_y_a1)
# loss, acc_A1_train, test_loss, acc_A1_test = A1.CNN_A1(tr_x_a1, cv_x_a1, image_A1_, tr_y_a1, cv_y_a1, y_a1)
# print(acc_A1_train,acc_A1_test)
# model_A1 = A1(args...)                 # Build model object.
# acc_A1_train = model_A1.train(args...) # Train model based on the training set (you should fine-tune your model based on validation set.)
# acc_A1_test = model_A1.test(args...)   # Test model based on the test set.
# Clean up memory/GPU etc...             # Some code to free memory if necessary.


# # ======================================================================================================================
# # Task A2
C = 3
Deg = 3
CO = 0.0
acc_A2_train, acc_A2_val, acc_A2_test=svma2.img_SVM(tr_x_a2.reshape(tr_x_a2.shape[0], 68*2), list(zip(*tr_y_a2))[0],
                                                    cv_x_a2.reshape(cv_x_a2.shape[0], 68*2), list(zip(*cv_y_a2))[0],
                                                    te_x_a2.reshape(te_x_a2.shape[0], 68*2), list(zip(*te_y_a2))[0],Deg,CO,C)
# acc_A2_train, acc_A2_val, acc_A2_test=svma2.img_SVM(tr_x_a2.reshape(tr_x_a2.shape[0], 68*2), list(zip(*tr_y_a2))[0],
#                              image_A2_.reshape((image_A2_.shape[0], 68*2)), list(zip(*y_a2))[0],Deg,CO,C)
# print(acc_A2_train, acc_A2_test)
# model_A2 = A2(args...)
# acc_A2_train = model_A2.train(args...)
# acc_A2_test = model_A2.test(args...)
# Clean up memory/GPU etc...


# # # ======================================================================================================================
# # # Task B1
a,b,c,d,e,f = B1.CNN_B1(datab1b2[0],datab1b2[1],datab1b2[2],datab1b2[3],datab1b2[4],datab1b2[5])
# a,b,c,d = B1.CNN_B1(datab1b2[0],datab1b2[1],image_B1_,datab1b2[3],datab1b2[4],y_b1)
loss_B1_tr = a
acc_B1_train = b
loss_B1_val = c
acc_B1_val = d
loss_B1_test = e
acc_B1_test = f
# print("train_loss:",a,"train_acc:",b,"test_loss",c,"test_acc",d)
# # model_B1 = B1(args...)
# # acc_B1_train = model_B1.train(args...)
# # acc_B1_test = model_B1.test(args...)
# # Clean up memory/GPU etc...


# # # ======================================================================================================================
# # # Task B2
aa,bb,cc,dd,ee,ff = B2.CNN_B2(datab1b2[6],datab1b2[7],datab1b2[8],datab1b2[9],datab1b2[10],datab1b2[11])
# a,b,c,d = B2.CNN_B2(datab1b2[6],datab1b2[7],image_B2_,datab1b2[9],datab1b2[10],y_b2)
loss_B2_tr = aa
acc_B2_train = bb
loss_B2_val = cc
acc_B2_val = dd
loss_B2_test = ee
acc_B2_test = ff
# print("train_loss:",a,"train_acc:",b,"test_loss:",c,"test_acc:",d)
# # model_B2 = B2(args...)
# # acc_B2_train = model_B2.train(args...)
# # acc_B2_test = model_B2.test(args...)
# # Clean up memory/GPU etc...


# # # ======================================================================================================================
# # ## Print out your results with following format:
print('TA1:{},{},{};TA2:{},{},{};TB1:{},{},{};TB2:{},{},{};'.format(acc_A1_train, acc_A1_val, acc_A1_test,
                                                        acc_A2_train, acc_A2_val, acc_A2_test,
                                                        acc_B1_train, acc_B1_val, acc_B1_test,
                                                        acc_B2_train, acc_B2_val, acc_B2_test))

# total_time = time.time()-now_time
# print(total_time)