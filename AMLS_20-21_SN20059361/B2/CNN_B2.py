import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from keras.preprocessing import image
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense 

import tensorflow as tf
from tensorflow.keras import Model
from sklearn.utils import shuffle
from tensorflow import keras
import time
#from B2 import landmarks as la
import matplotlib.pyplot as plt

# from keras import backend as K
# K.set_image_dim_ordering('th')

# global basedir, image_paths, target_size
# basedir = './AMLS_20-21_SNzcictxu'
# basedir = os.path.join(basedir,'Datasets')
# basedir = os.path.join(basedir,'cartoon_set')
# images_dir = os.path.join(basedir,'img')
# labels_filename = 'labels.csv'

# #nor
# def nor(x):
#     x = (x - np.mean(x)) / np.std(x)
#     return x

# #central_crop
# def center_crop(image):
#   cropped_image = tf.image.central_crop(
#       image, 0.5)
#   return cropped_image

# def data_preprocessing(data_B, image_B):
#     #obtain labels
#     # data = pd.read_csv(os.path.join(basedir, labels_filename),sep='\t')
#     Y_B2=data_B['eye_color']
#     Y_B1=data_B['face_shape']

#     # now_time = time.time()#起始时间

#     x_train, x_test, y_train, y_test = train_test_split(image_B, Y_B2, train_size=0.8, random_state=0)
#     x_train, x_cv, y_train, y_cv = train_test_split(x_train, y_train, train_size=0.8, random_state=0)

#     x_train = nor(x_train)
#     x_cv = nor(x_cv)
#     x_test = nor(x_test)

#     x_train = center_crop(x_train)
#     x_cv = center_crop(x_cv)
#     x_test = center_crop(x_test)

#     return x_train, x_cv, x_test, y_train, y_cv, y_test

# ############normalization##########################

# # x_train = nor(x_train)
# # x_cv = nor(x_cv)
# # x_test = nor(x_test)
# # # print(x_train.shape, x_cv.shape, x_test.shape)

# #min max normalize
# x_train = (x_train - np.min(x_train)) / (np.max(x_train) - np.min(x_train))
# x_cv = (x_cv - np.min(x_cv)) / (np.max(x_cv) - np.min(x_cv))
# x_test = (x_test - np.min(x_test)) / (np.max(x_test) - np.min(x_test))

# # x_train = x_train.reshape(x_train.shape[0],68,2,1)
# # x_cv = x_cv.reshape(x_cv.shape[0],68,2,1)
# # x_test = x_test.reshape(x_test.shape[0],68,2,1)

# # print(x_train.shape,x_cv.shape,x_test.shape)

# # x_train = x_train / 255.0
# # x_cv = x_cv / 255.0 #测试集不做增强
# # x_test = x_test / 255.0
# ######################################################

# # datagen = image.ImageDataGenerator(
    #horizontal_flip=True,    
# #     )
# #     vertical_flip=True
# #     rotation_range=1.2,
# #     zoom_range=0.2,
#     # horizontal_flip=True
#     # shear_range=0.2,
#     # width_shift_range=0.1,
#     # height_shift_range=0.1,
#     # zca_whitening = True,
#     # zca_epsilon = 1e-1
# # datagen_cv = image.ImageDataGenerator(
# #     zca_whitening=True,
# #     zca_epsilon = 1e-1)
# # datagen_te = image.ImageDataGenerator(
# #     zca_whitening=True,
# #     zca_epsilon = 1e-1)
# #datagen_0 = image.ImageDataGenerator(
#     #zca_whitening=False)

# #datagen.fit(x_train)
# # datagen_cv.fit(x_cv)
# # datagen_te.fit(x_test)
# #datagen_0.fit(x_train)

# x_train = np.array(x_train)
# print(x_train.shape)

# #crop
# # def random_crop(image):
# #   cropped_image = tf.image.random_crop(
# #       image, size=[x_train.shape[0],25,25,3])
# #   return tf.reshape(cropped_image, [res, res, 3])

# # x_train = random_crop(x_train)
# # x_train = np.array(x_train)
# # print(x_train.shape)

# # image show
# def show_image(x_train):
#     for x_batch in datagen_0.flow(x_train, batch_size=32, shuffle = True):
#         # create a grid of 4x4 images
#         fig, axes = plt.subplots(3, 3)
#         axes = axes.flatten()
#         for i in range(0, 9):
#             axes[i].imshow(x_batch[i])
#             # axes[i].set_xticks(())
#             # axes[i].set_yticks(())
#         plt.show()
#         # plt.tight_layout()
#         break
# # show_image(x_train)

# def show_augment_image(x_train, y_train, datagen):
#     # configure batch size and retrieve one batch of images
#     for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32, shuffle = True):
#         # create a grid of 4x4 images
#         fig, axes = plt.subplots(2, 2)
#         axes = axes.flatten()
#         for i in range(0, 4):
#             # print(x_batch[i].shape)
#             # axes[i].imshow(x_batch[i], cmap=plt.get_cmap('rgb'))
#             axes[i].imshow(x_batch[i])
#             # axes[i].set_xticks(())
#             # axes[i].set_yticks(())
#         plt.show()
#         # plt.tight_layout()
#         break
# #show_augment_image(x_train, y_train, datagen)
# # show_augment_image(x_cv, y_cv,datagen_cv)


# # # Pre-process data
# # scaler = MinMaxScaler() # This estimator scales and translates each feature individually such that it is in the given range on the training set, default between(0,1)
# # x_train = x_train.reshape(x_train.shape[0],50*50*3)
# # x_test = x_test.reshape(x_test.shape[0],50*50*3)
# # #x_test = x_test.reshape(x_test.shape[0],100*100*3)
# # x_train = scaler.fit_transform(x_train)
# # x_test = scaler.fit_transform(x_test)
# # #x_test = scaler.transform(x_test)

# # x_train = x_train/255.0
# # x_test = x_test/255.0

# # ########################CNN########################
def CNN_B2(x_train, x_cv, x_test, y_train, y_cv, y_test):
    class AlexNet(Model):
        def __init__(self):
            super(AlexNet, self).__init__()
            self.c1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')  # 卷积层
            #self.b1 = BatchNormalization()  # BN层
            # self.a1 = Activation('relu')  # 激活层
            self.p1 = MaxPool2D(pool_size=(2, 2), strides=2)  # 池化层
            #self.d1 = Dropout(0.2)  # dropout层
            
            self.c2 = Conv2D(filters=128, kernel_size=(3,3), activation='relu')
            #self.b2 = BatchNormalization()
            # self.a2 = Activation('relu')
            self.p2 = MaxPool2D(pool_size=(2,2), strides=2)
            
            self.c3 = Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu')
            #self.c4 = Conv2D(filters=384, kernel_size=(3,3), padding='same', activation='relu')
            #self.c5 = Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu')
            self.p3 = MaxPool2D(pool_size=(2,2), strides=2)

            self.flatten = Flatten()
            self.f1 = Dense(1024, activation='relu')
            self.d1 = Dropout(0.2)
            self.f2 = Dense(2048, activation='relu')
            self.d2 = Dropout(0.2)
            self.f3 = Dense(5, activation='softmax')
            
        def call(self, x):
            x = self.c1(x)
            #x = self.b1(x)
            #x = self.a1(x)
            x = self.p1(x)
            # x = self.d1(x)

            x = self.c2(x)
            #x = self.b2(x)
            #x = self.a2(x)
            x = self.p2(x)

            x = self.c3(x)

            #x = self.c4(x)

            #x = self.c5(x)
            x = self.p3(x)

            x = self.flatten(x)
            x = self.f1(x)
            x = self.d1(x)
            x = self.f2(x)
            x = self.d2(x)
            y = self.f3(x)
            return y
    # x_train, x_cv, x_test, y_train, y_cv, y_test = data_preprocessing(data_B, image_B)

    model = AlexNet()

    from tensorflow import keras
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    opt = keras.optimizers.Adam(learning_rate=0.00005)
    model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['sparse_categorical_accuracy'])

    epochs = 30
    model.compile(optimizer=opt,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['sparse_categorical_accuracy'])
    #itr = datagen.flow(x_train, y_train, batch_size=32)
    # itr_cv=datagen_cv.flow(x_cv, y_cv, batch_size=32)
    #history = model.fit(itr, epochs=epochs, validation_data=(x_cv,y_cv),validation_freq=1, callbacks=[callback])
    history = model.fit(x_train, y_train, batch_size=32, epochs=epochs, validation_data=(x_cv,y_cv), validation_freq=1, callbacks=[callback])
    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    results = model.evaluate(x_test, y_test)
    test_loss = results[0]
    test_acc = results[1]
    return loss[-1], acc[-1], val_loss[-1], val_acc[-1], test_loss, test_acc


   

# # # checkpoint_save_path = "./checkpoint/Baseline.ckpt"
# # # if os.path.exists(checkpoint_save_path + '.index'):
# # #     print('-------------load the model-----------------')
# # #     model.load_weights(checkpoint_save_path)

# # # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
# # #                                                  save_weights_only=True,
# # #                                                  save_best_only=True)

# # # #history = model.fit(x_train, y_train, batch_size=32, epochs=15, validation_data=(x_test, y_test), validation_freq=1,
# # #                     callbacks=[cp_callback])

# # #history = model.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_test, y_test), validation_freq=1, callbacks=[callback])

# # # model.summary()

# # # # print(model.trainable_variables)
# # # file = open('./weights.txt', 'w')
# # # for v in model.trainable_variables:
# # #     file.write(str(v.name) + '\n')
# # #     file.write(str(v.shape) + '\n')
# # #     file.write(str(v.numpy()) + '\n')
# # # file.close()

# # # # ###############################################    show   ###############################################

# # # 显示训练集和验证集的acc和loss曲线
# # acc = history.history['sparse_categorical_accuracy']
# # val_acc = history.history['val_sparse_categorical_accuracy']
# # loss = history.history['loss']
# # val_loss = history.history['val_loss']

# # plt.subplot(1, 2, 1)
# # plt.plot(acc, label='Training Accuracy')
# # plt.plot(val_acc, label='Validation Accuracy')
# # plt.title('Training and Validation Accuracy')
# # plt.legend()

# # plt.subplot(1, 2, 2)
# # plt.plot(loss, label='Training Loss')
# # plt.plot(val_loss, label='Validation Loss')
# # plt.title('Training and Validation Loss')
# # plt.legend()
# # plt.show()

# # total_time = time.time()-now_time
# # print(total_time)

# # print("Evaluate on tset data")
# # # itr_te=datagen_te.flow(x_test, y_test, batch_size=32)
# # # x_test, y_test = itr_te.next()
# # #results = model.evaluate(x_cv, np.array(list(zip(*y_test)))[0], batch_size=32)
# # results = model.evaluate(x_test, y_test)
# # #results = model.evaluate(itr_te)
# # print("test loss, test acc:", results)

# # ##########################################extra test data

# # # #data = pd.read_csv('./dataset_AMLS_20-21/celeba/labels.csv',sep='\t',header=0)
# # # data_ = pd.read_csv('./dataset_AMLS_20-21_test/cartoon_set_test/labels.csv',sep='\t')
# # # # Y_A0=data['gender']
# # # # Y_A1=data['smiling']
# # # y_B0=data_['eye_color']
# # # y_B1=data_['face_shape']
# # # # for i in range(Y_A0.shape[0]):
# # # #     if Y_A0[i] == -1:
# # # #         Y_A0[i] = 0
# # # #     else:
# # # #         Y_A0[i] = 1 #防止标签出现负数，如果这里负数的话后面计算loss就会出现nan
# # # # for i in range(Y_A1.shape[0]):
# # # #     if Y_A1[i] == -1:
# # # #         Y_A1[i] = 0
# # # #     else:
# # # #         Y_A1[i] = 1

# # # #图片读取
# # # train_image_ = [] 
# # # for i in tqdm(range(data_.shape[0])):
# # #     #img = image.load_img('./dataset_AMLS_20-21_test/celeba/img/'+str(i)+'.jpg', color_mode='rgb', target_size=(50,50), grayscale=False)
# # #     img_ = image.load_img('./dataset_AMLS_20-21_test/cartoon_set_test/img/'+str(i)+'.png', target_size=(100,100), grayscale=False, color_mode='rgb')
# # #     # img = img.resize((50,50))
# # #     img_ = image.img_to_array(img_)     
# # # #     img = img/255     
# # #     train_image_.append(img_) 
# # # X_ = np.array(train_image_)

# # # # Pre-process data
# # # scaler = MinMaxScaler() # This estimator scales and translates each feature individually such that it is in the given range on the training set, default between(0,1)
# # # X_ = X_.reshape(X_.shape[0],100*100*3)
# # # X_ = scaler.fit_transform(X_)

# # # X_ = X_.reshape(X_.shape[0],100,100,3)
# # # #X_ = X_/255.0
# # # # print(X_.shape, y_B0.shape)

# # # print("Evaluate on test data")
# # # results = model.evaluate(X_, y_B0, batch_size=32)
# # # print("test loss, test acc:", results)