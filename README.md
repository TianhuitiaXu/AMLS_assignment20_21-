# README
## 1. Organisation
   The project includes 4 main parts - A1, A2, B1, B2, each of which corresponds to a classification problem.

   The 'A1', ‘A2’, ‘B1’ and ‘B2’ folders corresponding to the four tasks in this project, which contain the code files for each task.

   In 'A1', there are 4.py files (except for the landmarks_A.py for feature extraction), each of which was tried to conduct a machine learning model for CeleA gender classification (A1) of the project:
   * 'LR_A1' : a logistic regression model.
   * 'SVM_A1' : a SVM model.
   * 'MLP_A1' : a MLP model which contains 3 fully connected layers
   * 'CNN_A1' : a convolutional neural network which contains 3 convolution layers and 3 fully connected layers
    
  In 'A2', there are 4 .py files, each of which was tried to conduct a machine learning model for CeleA facial expression (smiling) classification (A2) of the project:
   * 'LR_A2' : a logistic regression model.
   * 'SVM_A2' : a SVM model.
   * 'MLP_A2' : a MLP model which contains 3 fully connected layers
   * 'CNN_A2' : a convolutional neural network which contains 3 convolution layers and 3 fully connected layers
  
  In 'B1', there are 2 .py files, each of which was tried to conduct a machine learning model for Cartoon_set face shape classification (B1) of the project:
   * 'MLP_B1' : a MLP model which contains 3 fully connected layers
   * 'CNN_B1' : a convolutional neural network which contains 3 convolution layers and 3 fully connected layers
    
  In 'B2', there are 2 .py files, each of which was tried to conduct a machine learning model for Cartoon_set eye color classification (B2) of the project:
   * 'MLP_B2' : a MLP model which contains 3 fully connected layers
   * 'CNN_B2' : a convolutional neural network which contains 3 convolution layers and 3 fully connected layers
  
main.py conduct the data preprocessing and run the chosen model for each task and return the train_loss, train_acc, val_loss, val_acc, test_loss, test_acc for each task, respectively.

The estimated time of running main.py is 1465.82s on a laptop with 16.0 GB memory and NVIDIA GTX 1650 Ti, with CUDA.

## 2. Packages
Version:
  python 3.8.3, keras 2.3.1, tensorflow 2.2.0.
  
Packages:
  numpy, scipy, tensorflow, Keras, pandas, os, tqdm, scikit
    
