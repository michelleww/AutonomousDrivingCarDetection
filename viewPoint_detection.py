import numpy as np
import cv2 as cv2
import glob as glob
import os
from math import ceil
from datetime import datetime
from skimage.feature import hog
import joblib
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

def get_resize_info(input_path):
    # loop through every training image to find the median shape
    name_list = glob.glob(input_path, recursive=True)
    shape_matrix = np.zeros((len(name_list), 2))
    for i in range(len(name_list)):
        img = cv2.imread(name_list[i], 0)
        shape_matrix[i, :] = img.shape
    median = np.ceil(np.median(shape_matrix, axis=0)).astype(int)
    return  median

def prepare_data(input_directory, resize):
    feature_list = []
    labels = []
    angles = ['30', '60', '90', '120', '150', '180', '210', '240', '270', '300', '330', '360']
    for angle in angles:
        directory = input_directory  + angle + '/'
        files = os.listdir(directory)
        # filter out unnecessary files
        files = [f for f in files if f.endswith('.jpg')]
        labels.extend([angle]*len(files))
        for f in files:
            img_path = directory + f
            img = cv2.imread(img_path, 0)
            #resize the image incase of the negative dimension error for hog 
            img_resize = cv2.resize(img, (int(resize[1]*1.25), int(resize[0]*1.25)))
            features= hog(img_resize)
            feature_list.append(features)
    return feature_list, labels

def train(resize_info, input_directory, svm_path):
    # pre-processing trainning images to get data and labels
    print('Pre-proceesing training images from ' + input_directory)
    data, labels = prepare_data(input_directory, resize_info)
    print('Done pre-processing!')


    X_train, X_test, y_train, y_test = train_test_split(data,labels,test_size=0.3)
    print('Start Training process...')
    svm = LinearSVC(random_state=0, dual= False, multi_class='ovr', class_weight = 'balanced', max_iter= 10000, C= 100.0)
    svm.fit(X_train, y_train)
    joblib.dump(svm, svm_path)
    print('Done training!')

    return data, labels, svm

def test_data_set(data, labels, svm):
    X_train, X_test, y_train, y_test = train_test_split(data,labels,test_size=0.3)

    result_list = []
    for test in X_test:
        predict_result = svm.predict(test.reshape(1,-1))
        result_list.append(predict_result)

    result_list = np.array(result_list).ravel()
    ground_true_list = np.array(y_test).ravel()

    accuracy =  (result_list == ground_true_list).mean()
    print("The accuracy is: " + str(accuracy))

def test_single_image(img_path, resize, svm_path):
    # extraxct hog feature from the testing image
    img = cv2.imread(img_path, 0)
    img = cv2.resize(img, (int(resize[1]*1.25), int(resize[0]*1.25)))
    features = hog(img)

    # restore the svm 
    svm = joblib.load(svm_path)

    # predit 
    prediction = svm.predict(features.reshape(1,-1))
    print('The prediction for the new data is: Class ' + str(prediction[0]))

if __name__ == '__main__':
    resize_info = get_resize_info('angle_classification/*/*.jpg')

    # training data
    data, labels, svm = train(resize_info, 'angle_classification/', 'svm.pkl')

    # testing data set
    test_data_set(data, labels, svm)

    # testing singel image
    test_single_image('001045_150_1.jpg', resize_info, 'svm.pkl')
    test_single_image('002110_300_1.jpg', resize_info, 'svm.pkl')
