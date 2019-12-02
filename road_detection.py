from skimage import io
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.segmentation import slic
from skimage.measure import perimeter
import os
# from keras.models import Sequential
# from keras.models import model_from_json
# from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import joblib
import mahotas as mt

def extract_feature(image, image_gt):
    # calculate image gradient using sobel
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(image_gray,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(image_gray,cv2.CV_64F,0,1,ksize=5)

    # super pixels for approximatly 1000 size
    img_seg = slic(image, n_segments = 1000, compactness=8, convert2lab=True, min_size_factor=0.3)
    # actual total number of labels
    num_labels = np.amax(img_seg)+1

    bins = np.linspace(0,1,num=21)

    features_list = []
    labels = []
    for label in range(num_labels):
        features = []
        roi = (img_seg == label)
        roi_idx = np.nonzero(roi)

        seg_center_x = np.mean(roi_idx[0])
        seg_center_y = np.mean(roi_idx[1])

        seg_size = np.shape(roi_idx)[1]

        seg_perimeter = perimeter(roi)

        seg_sobel_x = sobelx[roi]
        seg_sobel_y = sobely[roi]

        magnitude_mean = np.mean(np.sqrt(seg_sobel_x ** 2 + seg_sobel_y ** 2))

        direction_mean = np.mean(np.arctan2(np.absolute(seg_sobel_y), np.absolute(seg_sobel_x)))

        # add computed 2 D features
        features.extend([seg_size, seg_perimeter, np.mean(seg_sobel_x), np.mean(seg_sobel_y), magnitude_mean, direction_mean])
        # extract color features 
        for channel in range(3):
            sum = 0
            for idx in range(len(roi_idx[0])):
                i = roi_idx[0][idx]
                j = roi_idx[1][idx]
                sum += image[i, j, channel]
            features.append(sum/len(roi_idx[0]))

        # add texture feature     
        seg_rgb = image[roi]
        textures = mt.features.haralick(seg_rgb)
        features.extend(textures.mean(axis=0))

        # adding lable
        if image_gt[int(seg_center_x),int(seg_center_y),2] > 0:
            labels.append(1)
        else:
            labels.append(0)

        features_list.append(features)

    return features_list, labels

def extract_test_feature(image):
    # calculate image gradient using sobel
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(image_gray,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(image_gray,cv2.CV_64F,0,1,ksize=5)

    # super pixels for approximatly 1000 size
    img_seg = slic(image, n_segments = 1000, compactness=8, convert2lab=True, min_size_factor=0.3)
    # actual total number of labels
    num_labels = np.amax(img_seg)+1
    # print(la)
    # print(np.unique(img_seg))
    # print(num_labels)
    bins = np.linspace(0,1,num=21)

    features_list = []
    for label in range(num_labels):
        features = []
        roi = (img_seg == label)
        roi_idx = np.nonzero(roi)

        seg_center_x = np.mean(roi_idx[0])
        seg_center_y = np.mean(roi_idx[1])

        seg_size = np.shape(roi_idx)[1]

        seg_perimeter = perimeter(roi)

        seg_sobel_x = sobelx[roi]
        seg_sobel_y = sobely[roi]
        magnitude_mean = np.mean(np.sqrt(seg_sobel_x ** 2 + seg_sobel_y ** 2))

        direction_mean = np.mean(np.arctan2(np.absolute(seg_sobel_y), np.absolute(seg_sobel_x)))

        # add computed 2 D features
        features.extend([seg_size, seg_perimeter, np.mean(seg_sobel_x), np.mean(seg_sobel_y), magnitude_mean, direction_mean])

        # extract color features 
        for channel in range(3):
            sum = 0
            for idx in range(len(roi_idx[0])):
                i = roi_idx[0][idx]
                j = roi_idx[1][idx]
                sum += image[i, j, channel]

            features.append(sum/len(roi_idx[0]))
        # add texture feature     
        seg_rgb = image[roi]
        textures = mt.features.haralick(seg_rgb)
        features.extend(textures.mean(axis=0))
        features_list.append(features)

    return features_list, img_seg

def extract_trainings():

    img_name = [name for name in os.listdir('train/image_left') if name.endswith('.jpg') or name.endswith('.png')]
    # all data features
    data_features = []
    data_labels = []
    
    print('Start Reading training images...')
    # Loop through all images
    for name in img_name:
 
        print('Loading image ' + name)
    
        # Load image and ground truth image
        img_path = 'train/image_left/' + name
        img_gt_name = name.replace('_', '_road_', 1).replace('jpg', 'png')
        img_gt_path = 'train/gt_image_left/' + img_gt_name

        img = cv2.imread(img_path)
        img_gt = cv2.imread(img_gt_path)

        features, labels= extract_feature(img, img_gt)

        # add features for a single image
        data_features.extend(features)
        data_labels.extend(labels)

    # Output Data to file
    np.save('training_data_features', data_features)
    np.save('training_data_labels', data_labels)

    # Display Message
    print('Data Successfully Saved!')
    return data_features, data_labels

# def training(data, labels):

#     # fix random seed for reproducibility
#     seed = 7
#     np.random.seed(seed)

#     # split into test set and training set
#     X_train, y_train = data,labels

#     # Reshaping the array to 4-dims so that it can work with the Keras API
#     X_train = X_train.reshape(len(X_train), 28, 28, 1)

#     # Making sure that the values are float so that we can get decimal points after division
#     X_train = X_train.astype('float32')

#     input_shape = (28, 28, 1)
#     # create model
#     model = Sequential()
#     #add model layers

#     model.add(Conv2D(32, kernel_size=5, activation='relu', input_shape=input_shape))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Conv2D(64, kernel_size=5, activation='relu', input_shape=input_shape))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
#     model.add(Dense(1024, activation='relu'))
#     model.add(Dropout(0.2))
#     model.add(Dense(10, activation='softmax'))

#     # Compile model
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#     # Fit the model
#     history = model.fit(X_train, y_train, validation_split=0.33, epochs=40, batch_size=100,  verbose=2)

#     # save the model for future usage
#     model_json = model.to_json()
#     with open("road_detection_cnn_model.json", "w") as json_file:
#         json_file.write(model_json)
#     # serialize weights to HDF5
#     model.save_weights('road_detection_cnn_model.h5')
#     print("Saved model to disk")

#     # summarize history for accuracy
#     plt.plot(history.history['accuracy'])
#     plt.plot(history.history['val_accuracy'])
#     plt.title('model accuracy')
#     plt.ylabel('accuracy')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'validation'], loc='upper left')
#     plt.show()
#     # summarize history for loss
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('model loss')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'validation'], loc='upper left')
#     plt.show()

def train(data, labels, svm_path):
    X_train, X_test, y_train, y_test = train_test_split(data,labels,test_size=0.3)
    print('Start Training process...')
    svm = LinearSVC(random_state=0, dual= False, multi_class='ovr', class_weight = 'balanced', max_iter= 10000, C= 100.0)
    svm.fit(X_train, y_train)
    joblib.dump(svm, svm_path)
    print('Done training!')

def test_single_data(image):
    data, labels = [], []
    if os.path.isfile('training_data_features.npy') and os.path.isfile('training_data_labels.npy'):
        data, labels = load_training_data()
    else: 
        data, labels = extract_trainings()


    if not os.path.isfile('road_detection_svm.pkl'):
        train(data, labels, 'road_detection_svm.pkl')

    test_data, img_seg = extract_test_feature(image)
    svm = joblib.load('road_detection_svm.pkl')
 
    # evaluate loaded model on test data
    predictions = svm.predict(test_data)

    print(predictions)
    return predictions, img_seg

def load_training_model(name):
    # load json and create model
    json_file = open(name +'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(name + '.h5')
    print("Loaded model from disk")
    return loaded_model

def load_training_data():
    # Load Data
    data = np.load('training_data_features.npy', allow_pickle=True)
    labels = np.load('training_data_labels.npy', allow_pickle=True)
    
    # Display Message
    print('Training Data Successfully Loaded!')
    return data, labels

def load_testing_data():
    # Load Data
    data = np.load('testing_data_features.npy', allow_pickle=True)
    
    # Display Message
    print('Testing Data Successfully Loaded!')
    return data

if __name__ == '__main__':
    test = cv2.imread('test/image_left/umm_000061.jpg')
    test_single_data(test)