from skimage import io
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import joblib
import mahotas as mt
import matplotlib.colors as mcolors
import webcolors
from compute3d import get_3d_locations
from skimage.color import rgb2lab, lab2rgb
import copy
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout

def extract_feature(image, image_gt, image_right, calib_path):
    # calculate image gradient using sobel
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(image_gray,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(image_gray,cv2.CV_64F,0,1,ksize=5)

    image_lab = rgb2lab(copy.deepcopy(image)/255)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # compute 3D locations
    locations_3D = get_3d_locations(image, image_right, calib_path)

    # super pixels for approximatly 1000 size
    img_seg = slic(image, n_segments = 1000, compactness=8, convert2lab=True, min_size_factor=0.3)
    # actual total number of labels
    num_labels = np.amax(img_seg)+1

    features_list = []
    labels = []
    for label in range(num_labels):
        features = []
        roi = (img_seg == label)
        roi_idx = np.nonzero(roi)

        seg_center_x = np.mean(roi_idx[0])
        seg_center_y = np.mean(roi_idx[1])

        seg_sobel_x = sobelx[roi]
        seg_sobel_y = sobely[roi]

        magnitude_mean = np.mean(np.sqrt(seg_sobel_x ** 2 + seg_sobel_y ** 2))

        direction_mean = np.mean(np.arctan2(np.absolute(seg_sobel_y), np.absolute(seg_sobel_x)))

        # add computed 2 D features
        features.extend([np.mean(seg_sobel_x), np.mean(seg_sobel_y), magnitude_mean, direction_mean])
        
        # extract color features  and 3D feature
        x, y, z = 0, 0, 0
        flag = True
        rgb_mean = []
        lab_mean = []
        hsv_mean = []
        size = len(roi_idx[0])

        for channel in range(3):
            sum_rgb = 0
            sum_lab = 0
            sum_hsv = 0
            for idx in range(size):
                i = roi_idx[0][idx]
                j = roi_idx[1][idx]
                sum_rgb += image[i, j, channel]
                sum_lab += image_lab[i,j,channel]
                sum_hsv += image_hsv[i,j,channel]
                if flag:
                    if idx == size - 1:
                        flag = False
                    item_idx = image.shape[1] * i + j
                    x += locations_3D[item_idx, 0]
                    y += locations_3D[item_idx, 1]
                    z += locations_3D[item_idx, 2]
            rgb_mean.append(sum_rgb/size)
            lab_mean.append(sum_lab/size)
            hsv_mean.append(sum_hsv/size)        
        features.extend(rgb_mean)
        features.extend(lab_mean)
        features.extend(hsv_mean)
        features.extend([x/size, y/size, z/size])

        # add texture feature     
        seg_rgb = image[roi]
        textures = mt.features.haralick(seg_rgb)
        features.append(np.mean(textures))

        # adding lable
        if image_gt[int(seg_center_x),int(seg_center_y),2] > 0:
            labels.append(1)
        else:
            labels.append(0)

        features_list.append(features)

    return features_list, labels

def extract_test_feature(image, image_right, calib_path):
    # calculate image gradient using sobel
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(image_gray,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(image_gray,cv2.CV_64F,0,1,ksize=5)

    image_lab = rgb2lab(copy.deepcopy(image)/255)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # compute 3D locations
    locations_3D = get_3d_locations(image, image_right, calib_path)

    # super pixels for approximatly 1000 size
    img_seg = slic(image, n_segments = 1000, compactness=8, convert2lab=True, min_size_factor=0.3)
    # actual total number of labels
    num_labels = np.amax(img_seg)+1

    features_list = []
    for label in range(num_labels):
        features = []
        roi = (img_seg == label)
        roi_idx = np.nonzero(roi)

        seg_sobel_x = sobelx[roi]
        seg_sobel_y = sobely[roi]
        magnitude_mean = np.mean(np.sqrt(seg_sobel_x ** 2 + seg_sobel_y ** 2))

        direction_mean = np.mean(np.arctan2(np.absolute(seg_sobel_y), np.absolute(seg_sobel_x)))

        # add computed 2 D features
        features.extend([np.mean(seg_sobel_x), np.mean(seg_sobel_y), magnitude_mean, direction_mean])

        # extract color features  and 3D feature
        x, y, z = 0, 0, 0
        flag = True
        rgb_mean = []
        lab_mean = []
        hsv_mean = []
        size = len(roi_idx[0])

        for channel in range(3):
            sum_rgb = 0
            sum_lab = 0
            sum_hsv = 0
            for idx in range(size):
                i = roi_idx[0][idx]
                j = roi_idx[1][idx]
                sum_rgb += image[i, j, channel]
                sum_lab += image_lab[i,j,channel]
                sum_hsv += image_hsv[i,j,channel]
                if flag:
                    if idx == size - 1:
                        flag = False
                    item_idx = image.shape[1] * i + j
                    x += locations_3D[item_idx, 0]
                    y += locations_3D[item_idx, 1]
                    z += locations_3D[item_idx, 2]
            rgb_mean.append(sum_rgb/size)
            lab_mean.append(sum_lab/size)
            hsv_mean.append(sum_hsv/size)        
        features.extend(rgb_mean)
        features.extend(lab_mean)
        features.extend(hsv_mean)
        features.extend([x/size, y/size, z/size])

        # add texture feature     
        seg_rgb = image[roi]
        textures = mt.features.haralick(seg_rgb)
        features.append(np.mean(textures))

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
        img_right_path = 'train/image_right/' + name
        calib_path= ('train/calib/'+ name).replace('jpg', 'txt')

        img = cv2.imread(img_path)
        img_gt = cv2.imread(img_gt_path)
        img_right = cv2.imread(img_right_path)

        features, labels= extract_feature(img, img_gt, img_right, calib_path)

        # add features for a single image
        data_features.extend(features)
        data_labels.extend(labels)

    # Output Data to file
    np.save('training_data_features', data_features)
    np.save('training_data_labels', data_labels)

    # Display Message
    print('Data Successfully Saved!')
    return data_features, data_labels

def test_single_data(image_path):
    data, labels = [], []

    if os.path.isfile('training_data_features.npy') and os.path.isfile('training_data_labels.npy'):
        data, labels = load_training_data()
    else: 
        data, labels = extract_trainings()

    image = cv2.imread(image_path)
    image_right = cv2.imread(image_path.replace('image_left', 'image_right'))
    calib_path = image_path.replace('image_left', 'calib').replace('.jpg', '.txt')
    test_data, img_seg = extract_test_feature(image, image_right, calib_path)

    # test_data = np.array(test_data).astype('float32')
    # test_data = np.nan_to_num(test_data)

    # # Making sure that the values are float so that we can get decimal points after division
    # data = data.astype('float32')
    # data = np.nan_to_num(data)

    if os.path.isfile('road_detection_cnn_model.json'):
        model = load_training_model('road_detection_cnn_model')
    else:
        model = training(data, labels)

    predictions = model.predict(test_data)

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

def get_segmentation(predictions, img, img_seg):
    num_labels = np.amax(img_seg)+1
    blank = np.zeros((img.shape[0],img.shape[1]))
    for label in range(num_labels):
        pre = predictions[label]
        if pre == 1:
            roi = (img_seg == label)
            roi_idx = np.nonzero(roi)
            for idx in range(len(roi_idx[0])):
                i = roi_idx[0][idx]
                j = roi_idx[1][idx]
                blank[i,j] = 10
    return blank

# see tut7 for reference
def visualize_segementation(imgLeft, gt_mask):
    unique, counts = np.unique(gt_mask, return_counts=True)
    obj_ids = np.unique(gt_mask)
    
    number_object = obj_ids.shape[0]

    # norm ids
    count = 0
    for o_id in obj_ids:
        gt_mask[gt_mask == o_id] = count
        count += 1

    base_COLORS = []

    for key, value in mcolors.CSS4_COLORS.items():
        rgb = webcolors.hex_to_rgb(value)
        base_COLORS.append([rgb.blue, rgb.green, rgb.red])
    base_COLORS = np.array(base_COLORS)

    np.random.seed(99)
    base_COLORS = np.random.permutation(base_COLORS)

    colour_id = np.array([(id) % len(base_COLORS) for id in range(number_object)])

    COLORS = base_COLORS[colour_id]
    COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")
    gt_mask = np.int_(gt_mask)
    mask = COLORS[gt_mask]

    output = ((0.4 * imgLeft) + (0.6 * mask)).astype("uint8")
    
    fig=plt.figure(figsize=(10, 10))
    plt.imshow(output[:,:,::-1])
    plt.show()



def training(data, labels):

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    # split into test set and training set
    X_train, y_train = data,labels

    input_shape = (len(X_train), 17)
    # create model
    model = Sequential()
    #add model layers

    model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(64, kernel_size=5, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    history = model.fit(X_train, y_train, validation_split=0.33, epochs=40, batch_size=32,  verbose=2)

    # save the model for future usage
    model_json = model.to_json()
    with open("road_detection_cnn_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights('road_detection_cnn_model.h5')
    print("Saved model to disk")

    return model

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

if __name__ == '__main__':
    test_path = 'test/image_left/umm_000061.jpg'
    test = cv2.imread('test/image_left/umm_000061.jpg')
    predictions, img_seg = test_single_data(test_path)
    gt_mask_1 = get_segmentation(predictions, test, img_seg)
    visualize_segementation(test, gt_mask_1)
