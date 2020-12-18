import os
import numpy as np
from keras.preprocessing import image
import cv2
import dlib
import pandas as pd
from pathlib import PurePath
#import defusedxml

# PATH TO ALL IMAGES
# global basedir, image_paths, target_size
# #basedir = './AMLS_assignment20_21-/dataset_AMLS_20-21'
# basedir = './dataset_AMLS_20-21'
# basedir = os.path.join(basedir,'celeba')
# #images_dir = basedir
# images_dir = os.path.join(basedir,'img')
# labels_filename = 'labels.csv'

detector = dlib.get_frontal_face_detector()
basedirA2 = './A2'
predictor = dlib.shape_predictor(os.path.join(basedirA2,'shape_predictor_68_face_landmarks.dat'))
def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def run_dlib_shape(image):
    # in this function we load the image, detect the landmarks of the face, and then return the image and the landmarks
    # load the input image, resize it, and convert it to grayscale
    resized_image = image.astype('uint8')

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    num_faces = len(rects)

    if num_faces == 0:
        return None, resized_image

    face_areas = np.zeros((1, num_faces))
    face_shapes = np.zeros((136, num_faces), dtype=np.int64)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        temp_shape = predictor(gray, rect)
        temp_shape = shape_to_np(temp_shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)],
        #   (x, y, w, h) = face_utils.rect_to_bb(rect)
        (x, y, w, h) = rect_to_bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h
    # find largest face and keep
    dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])

    return dlibout, resized_image


def extract_features_labels(basedir,labels_filename,images_dir,image_paths):
    """
    This funtion extracts the landmarks features for all images in the folder 'dataset/celeba'.
    It also extracts the gender label for each image.
    :return:
        landmark_features:  an array containing 68 landmark points for each image in which a face was detected
        gender_labels:      an array containing the gender label (male=0 and female=1) for each image in
                            which a face was detected
    """
    #print(images_dir)
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    target_size = None
    
    #data = pd.read_csv('./dataset_AMLS_20-21/celeba/labels.csv',sep='\t',header=0,names = name)
    data = pd.read_csv(os.path.join(basedir, labels_filename),sep='\t')
    gender = data['gender']
    smiling = data['smiling']
    #print(gender_labels[0:5])

#     labels_file = open(os.path.join(basedir, labels_filename), 'r')
#     lines = labels_file.readlines() 
# # from list get str get array symbols
# for i in range(len(rows)):
#     str0 = listToString(rows[i])
#     y = str0.split('\t')
#     y = np.array(y)
#     Y_A1[i,0] = y[2]
#     Y_A1[i,1] = y[3]
#     #str = str+str0
    #gender_labels = {listToString(line).split('\t')[0] : int(listToString(line).split('\t')[3]) for line in lines[1:]}
    #gender_labels = {listToString(line).split('\t')[3] for line in lines[1:]}
    
#     for line in lines[1:10]:
#         #print(line)
#         gender_labels.append(listToString(line).split('\t')[2])
#         a = listToString(line).split('\t')
#         print(np.array(a)[2])
#     #gender_labels = np.array(gender_labels)
                      
    if os.path.isdir(images_dir):
        all_features = []
        all_labels = []
        for img_path in image_paths:
            #print(img_path)
            file_name= PurePath(img_path.split('.')[1]).parts[-1]
            file_name=int(file_name)
            #print(file_name)
            # load image
            img = image.img_to_array(
                image.load_img(img_path,
                               target_size=target_size,
                               interpolation='bicubic'))
            #print(img)
            features, _ = run_dlib_shape(img)
            #print(features)
            if features is not None:
                all_features.append(features)
                #print(gender_labels[file_name])
                all_labels.append(smiling[file_name])
                #all_labels.append(gender_labels[file_name])
    #print(all_labels)

    landmark_features = np.array(all_features)
    gender_labels = (np.array(all_labels, dtype=float) + 1)/2 # simply converts the -1 into 0, so male=0 and female=1
    return landmark_features, gender_labels