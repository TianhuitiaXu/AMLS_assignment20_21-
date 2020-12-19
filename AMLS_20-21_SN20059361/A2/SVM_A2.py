import numpy as np
import pandas as pd
from sklearn.metrics import classification_report,accuracy_score
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score

# def get_data():

#     X, y = la.extract_features_labels()
#     Y = np.array([y, -(y - 1)]).T
#     # Shuffle and split the data into training and test set
#     #X, Y = shuffle(X,Y)
#     x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=0)
#     x_train, x_cv, y_train, y_cv = train_test_split(x_train, y_train, train_size=0.8, random_state=0)
#     return x_train, y_train, x_cv, y_cv, x_test, y_test
# x_train, y_train, x_cv, y_cv, x_test, y_test = get_data()

# now_time = time.time()#起始时间

# print(x_train.shape, x_cv.shape, x_test.shape)

#dataprocessing
# x_train = x_train / 255.0
# x_cv = x_cv / 255.0 #测试集不做增强
# x_test = x_test / 255.0

# x_train = (x_train - np.min(x_train)) / (np.max(x_train) - np.min(x_train))
# x_cv = (x_cv - np.min(x_cv)) / (np.max(x_cv) - np.min(x_cv))
# x_test = (x_test - np.min(x_test)) / (np.max(x_test) - np.min(x_test))

##SVM polyC=3 0.88A2; polyC=3 0.9343065693430657 A1
def img_SVM(training_images, training_labels, val_images, val_labels, test_images, test_labels,D,CO,C):
    classifier = SVC(kernel='poly', degree=D, coef0=CO, C=C,random_state=0)
    classifier.fit(training_images, training_labels)
    pred_tr = classifier.predict(training_images)
    pred_cv = classifier.predict(val_images)
    pred_te = classifier.predict(test_images)
    accu_tr = accuracy_score(test_labels,pred_te)
    accu_cv = accuracy_score(val_labels,pred_cv)
    accu_te = accuracy_score(training_labels,pred_tr)
    return accu_tr, accu_cv, accu_te
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

# accuracy,classifier=img_SVM(x_train.reshape(x_train.shape[0], 68*2), list(zip(*y_train))[0], x_cv.reshape((x_cv.shape[0], 68*2)), list(zip(*y_cv))[0], 3, 0.0,3)
# print(accuracy)

# print("accuracy on test")
# pred_test = classifier.predict(x_test.reshape((x_test.shape[0], 68*2)))
# accu = accuracy_score(list(zip(*y_test))[0], pred_test)
# print(accu)

# total_time = time.time()-now_time
# print(total_time)

##logictic l1liblinear0.9 A2;l2liblinear0.91796875 A1
# def logRegrPredict(x_train, y_train, x_cv):
#     # Build Logistic Regression Model
#     logreg = LogisticRegression(penalty='l2',solver='liblinear',max_iter=100,random_state=0)
#     # Train the model using the training sets
#     logreg.fit(x_train, y_train)
#     y_pred= logreg.predict(x_cv)
#     #print('Accuracy on test set: {:.2f}'.format(logreg.score(x_test, y_test)))
#     return y_pred

# y_pred = logRegrPredict(x_train.reshape((x_train.shape[0], 68*2)), y_train, x_cv.reshape((x_cv.shape[0], 68*2)))
# #print(y_pred)
# print('Accuracy on test set: '+str(accuracy_score(y_cv,y_pred)))
# print(classification_report(y_cv,y_pred))#text report showing the main classification metrics
