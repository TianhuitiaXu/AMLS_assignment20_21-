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
#from A1 import landmarks_A as la
import matplotlib.pyplot as plt
#from matplotlib import pyplot

# # def get_data():

# #     X, y = la.extract_features_labels()
# #     Y = np.array([y, -(y - 1)]).T
# #     # Shuffle and split the data into training and test set
# #     # X, y = shuffle(X,y)
# #     x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=0)
# #     x_train, x_cv, y_train, y_cv = train_test_split(x_train, y_train, train_size=0.8, random_state=0)
# #     return x_train, y_train, x_cv, y_cv, x_test, y_test
# # x_train, y_train, x_cv, y_cv, x_test, y_test = get_data()


# # name = ['eye_color', 'face_shape']
# data = pd.read_csv('./dataset_AMLS_20-21/celeba/labels.csv',sep='\t',header=0)
# #data = pd.read_csv('./dataset_AMLS_20-21/cartoon_set/labels.csv',sep='\t')
# Y_A0=data['gender']
# Y_A1=data['smiling']
# # Y_B0=data['eye_color']
# # Y_B1=data['face_shape']
# for i in range(Y_A0.shape[0]):
#     if Y_A0[i] == -1:
#         Y_A0[i] = 0
#     else:
#         Y_A0[i] = 1 #防止标签出现负数，如果这里负数的话后面计算loss就会出现nan
# for i in range(Y_A1.shape[0]):
#     if Y_A1[i] == -1:
#         Y_A1[i] = 0
#     else:
#         Y_A1[i] = 1

# #图片读取
# train_image = [] 
# for i in tqdm(range(data.shape[0])):
#     img = image.load_img('./dataset_AMLS_20-21/celeba/img/'+str(i)+'.jpg', color_mode='grayscale', target_size=(108,89))
#     img = image.img_to_array(img)     
# #     img = img/255     
#     train_image.append(img) 
# X = np.array(train_image)

# now_time = time.time()#起始时间

# # X, Y_A1 = shuffle(X, Y_A1)
# x_train, x_test, y_train, y_test = train_test_split(X, Y_A0,random_state=0)
# x_train, x_cv, y_train, y_cv = train_test_split(x_train, y_train,random_state=0)
# def nor(x):
#     x = (x - np.mean(x)) / np.std(x)
#     return x
# x_train = nor(x_train)
# x_cv = nor(x_cv)
# x_test = nor(x_test)
# # x_train = x_train / 255.0
# # x_cv = x_cv / 255.0 #测试集不做增强
# # x_test = x_test / 255.0

# datagen = image.ImageDataGenerator(
#     zca_whitening = True,
#     zca_epsilon = 1e-1)
#     # rotation_range=1.2,
#     # shear_range=0.2,
#     # width_shift_range=0.1,
#     # height_shift_range=0.1,
#     # horizontal_flip=True)
# datagen_cv = image.ImageDataGenerator(
#     zca_whitening=True,
#     zca_epsilon = 1e-1)
# datagen_te = image.ImageDataGenerator(
#     zca_whitening=True,
#     zca_epsilon = 1e-1)
# # datagen_0 = image.ImageDataGenerator(
# #     horizontal_flip=False)
#     #    fill_mode='nearest'
#     # featurewise_center=True,
#     # featurewise_std_normalization=True,
#     # zca_whitening = True,
# # datagen = image.ImageDataGenerator(
# #     rescale=1./255,
# #     horizontal_flip=True)
# # # (std, mean, and principal components if ZCA whitening is applied)

# #landmark_data_reshape
# # x_train = x_train.reshape(x_train.shape[0],68,2,1)
# # x_cv = x_cv.reshape(768,68,2,1)
# # x_test = x_test.reshape(959,68,2,1)
# datagen.fit(x_train)
# datagen_cv.fit(x_cv)
# datagen_te.fit(x_test)
# # datagen_0.fit(x_train)
# #print(x_train.max())
# # def show_image(x_train):
# #     for x_batch in datagen_0.flow(x_train, batch_size=32, shuffle = False):
# #         # create a grid of 4x4 images
# #         fig, axes = plt.subplots(2, 2)
# #         axes = axes.flatten()
# #         for i in range(0, 4):
# #             axes[i].imshow(x_batch[i], cmap=plt.get_cmap('gray'))
# #             axes[i].set_xticks(())
# #             axes[i].set_yticks(())
# #         plt.show()
# #         plt.tight_layout()
# #         break
# # show_image(x_train)

# def show_augment_image(x_train, y_train,datagen):
#     # configure batch size and retrieve one batch of images
#     for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32, shuffle = True):
#         # create a grid of 4x4 images
#         fig, axes = plt.subplots(2, 2)
#         axes = axes.flatten()
#         for i in range(0, 4):
#             # print(x_batch[i].shape)
#             axes[i].imshow(x_batch[i], cmap=plt.get_cmap('gray'))
#             # axes[i].set_xticks(())
#             # axes[i].set_yticks(())
#         plt.show()
#         # plt.tight_layout()
#         break
# show_augment_image(x_train, y_train,datagen)

# #crop
# # def random_crop(image):
# #   cropped_image = tf.image.random_crop(
# #       image, size=[2812,22, 27, 1])
# #   return cropped_image

# # x_train = random_crop(x_train)
# # x_train = np.array(x_train)

# # # fits the model on batches with real-time data augmentation:

# #min max normalize
# # # x_train = (x_train - np.min(x_train)) / (np.max(x_train) - np.min(x_train))
# # # x_cv = (x_cv - np.min(x_cv)) / (np.max(x_cv) - np.min(x_cv))
# # # x_test = (x_test - np.min(x_test)) / (np.max(x_test) - np.min(x_test))

# # # #用sequential
# # # model = tf.keras.models.Sequential([
# # #     tf.keras.layers.Flatten(),
# # #     tf.keras.layers.Dense(128, activation='relu'),#第一层神经网络有128神经元
# # #     tf.keras.layers.Dense(2,activation='softmax')#使输出满足概率分布
# # # ])

# # 用类
# class MnistModel(Model):
#     def __init__(self):
#         super(MnistModel,self).__init__()
#         self.flatten = Flatten()
#         self.f1 = Dense(2048, activation='relu')
#         self.d1 = Dropout(0.2)
#         #self.b1 = BatchNormalization()  # BN层
#         self.f2 = Dense(4096,activation='relu')
#         self.d2 = Dropout(0.2)
#         #self.b2 = BatchNormalization()  # BN层
#         self.f3 = Dense(2, activation='softmax')

#     def call(self,x):
#         x = self.flatten(x)
#         x = self.f1(x)
#         x = self.d1(x)
#         #x = self.b1(x)
#         x = self.f2(x)
#         x = self.d2(x)
#         #x = self.b2(x)
#         y = self.f3(x)
#         return y

# model = MnistModel()
# callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)
# opt = keras.optimizers.Adam(learning_rate=0.00005)
# epochs = 100
# model.compile(optimizer=opt,
#              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
#              metrics=['sparse_categorical_accuracy'])
# itr = datagen.flow(x_train, y_train, batch_size=32)
# itr_cv=datagen_cv.flow(x_cv, y_cv, batch_size=32)
# # x_cv = itr_cv.next()
# # x, y = itr.next()#turn iterator to array
# # x = np.concatenate((x,x_train))
# # y = np.concatenate((y,y_train))
# #print(np.std(x))
# history = model.fit(itr, epochs=epochs, validation_data=itr_cv,validation_freq=1, callbacks=[callback])
# # #history = model.fit(x_train, np.array(list(zip(*y_train))[0]), batch_size=32, epochs=100, validation_data=(x_cv, np.array(list(zip(*y_cv))[0])),shuffle=True, validation_freq=1, callbacks=[callback])#每迭代一次执行一次测试机评测
# # model.summary()

# # 显示训练集和验证集的acc和loss曲线
# acc = history.history['sparse_categorical_accuracy']
# val_acc = history.history['val_sparse_categorical_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# plt.subplot(1, 2, 1)
# plt.plot(acc, label='Training Accuracy')
# plt.plot(val_acc, label='Validation Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.legend()
# plt.show()

# total_time = time.time()-now_time
# print(total_time)

# print("Evaluate on cross-validation data")
# itr_te=datagen_te.flow(x_test, y_test, batch_size=32)
# # x_test, y_test = itr_te.next()
# #results = model.evaluate(x_cv, np.array(list(zip(*y_test)))[0], batch_size=32)
# results = model.evaluate(itr_te)
# print("test loss, test acc:", results)