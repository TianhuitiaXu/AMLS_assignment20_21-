#import csv
import numpy as np
import pandas as pd
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn import svm
from sklearn.svm import SVC

#-------------read data_panda
#name = ['gender', 'smiling']
#data = pd.read_csv('./dataset_AMLS_20-21/celeba/labels.csv',sep='\t',header=0,names = name)
data = pd.read_csv('./dataset_AMLS_20-21/celeba/labels.csv',sep='\t')
y_1=data['gender']
y_2=data['smiling']
for i in range(y_1.shape[0]):
    if y_1[i] == -1:
        y_1[i] = 0.0
    else:
        y_1[i] = 1.0 #防止标签出现负数，如果这里负数的话后面计算loss就会出现nan
        
for i in range(y_2.shape[0]):
    if y_2[i] == -1:
        y_2[i] = 0.0
    else:
        y_2[i] = 1.0

# y_label = []
# for i in range(y_1.shape[0]):
#     y_label.append(y_1[i])
# y_label = np.array(y_label,dtype=float)

from tqdm import tqdm
from keras.preprocessing import image 
#图片读取
train_image = [] 
for i in tqdm(range(data.shape[0])):
    img = image.load_img('./dataset_AMLS_20-21/celeba/img/'+str(i)+'.jpg', color_mode='rgb', target_size=(89,109))
    #img = image.load_img('./dataset_AMLS_20-21/cartoon_set/img/'+str(i)+'.png', target_size=None, grayscale=True)
    #img = img.resize((50,50))
    img = image.img_to_array(img)     
#     img = img/255     
    train_image.append(img) 
X = np.array(train_image, dtype='float64')
X = X / 255.0
#print(type(X[0,0,0,0]))
#X, y_1 = shuffle(X,y_1)
x_train, x_test, y_train, y_test = train_test_split(X, y_1, train_size=0.8, random_state=0)
x_train, x_cv, y_train, y_cv = train_test_split(x_train, y_train, train_size=0.8, random_state=0)
now_time = time.time()#起始时间

#dataaugmentation
datagen = image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)    
    #     
# datagen_cv = image.ImageDataGenerator(
#     zca_whitening=True)
#    featurewise_center=True,
#     featurewise_std_normalization=True)
# # compute quantities required for featurewise normalization
# # (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(x_train)
# datagen_cv.fit(x_cv)
# fits the model on batches with real-time data augmentation:
# itr=datagen.flow(x_train, y=y_train, batch_size=3200)
# x, y = itr.next()
# print(type(x),type(y))

##logistic
def logRegrPredict(x_train, y_train, x_cv):
    # Build Logistic Regression Model
    logreg = LogisticRegression(penalty='l1',solver='liblinear',C=0.5,max_iter=100,random_state=0)
    # Train the model using the training sets
    # logreg.fit(x_train.reshape(x_train.shape[0],89*109*3), y_train)
    itr=datagen.flow(x_train, y=y_train, batch_size=3200)
    x, y = itr.next()#turn iterator to array
    x = np.concatenate((x,x_train))
    y = np.concatenate((y,y_train))
    logreg.fit(x.reshape(x.shape[0],89*109*3),y)
    print(x.shape)
    # itr_cv=datagen_cv.flow(x_cv, batch_size=800)
    # x_cv = itr_cv.next()
    y_pred= logreg.predict(x_cv.reshape(x_cv.shape[0],89*109*3))
    #print('Accuracy on test set: {:.2f}'.format(logreg.score(x_test, y_test)))
    return y_pred

#y_pred = logRegrPredict(x_train.reshape((x_train.shape[0], 89*109*3)), y_train, x_cv.reshape((x_cv.shape[0], 89*109*3)))
y_pred = logRegrPredict(x_train, y_train, x_cv)
# print(x_train[0,0,0,0],x_cv[0,0,0,0])
#print(y_pred)
#print(y_pred)
print('Accuracy on test set: '+str(accuracy_score(y_cv,y_pred)))
print(classification_report(y_cv,y_pred))#text report showing the main classification metrics

##SVM polyC=3 0.88A2; polyC=3 0.9343065693430657 A1
# def img_SVM(training_images, training_labels, test_images, test_labels,D,CO,C):
#     classifier = SVC(kernel='poly', degree=D, coef0=CO, C=C,random_state=0)
#     classifier.fit(training_images, training_labels)
#     pred = classifier.predict(test_images)
#     accuracy = accuracy_score(test_labels, pred)
#     #para = classifier.get_params()
# #     print("Accuracy:", accuracy_score(test_labels, pred))

#    # print(pred)
#     return accuracy, classifier
# pred_C = []
# C = [0.1,0.5,1,2,5,10]
# D = [1,3,5]
# CO = 0.0
# for i in range(6):
#     for j in range(3):  
#         pred,classifier=img_SVM(x_train.reshape((x_train.shape[0], 68*2)), y_train, x_cv.reshape((x_cv.shape[0], 68*2)), y_cv, D[j], CO,C[i])
#         #pred=img_SVM(X_train.reshape((3068, 68*2)), y_train, X_test.reshape((969, 68*2)), y_test)
#         pred_C.append(pred)
#print(pred_C)

# accuracy,classifier=img_SVM(x_train.reshape(x_train.shape[0], 89*109*3), y_train, x_cv.reshape((x_cv.shape[0], 89*109*3)), y_cv, 3, 0.0,1)
# print(accuracy)

# print("accuracy on test")
# pred_test = classifier.predict(x_test.reshape((x_test.shape[0], 89*109*3)))
# accu = accuracy_score(y_test, pred_test)
# print(accu)


total_time = time.time()-now_time
print(total_time)
# model.fit(datagen.flow(x_train, y_train, batch_size=32),
#           steps_per_epoch=len(x_train) / 32, epochs=epochs)